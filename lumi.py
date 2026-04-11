# CLAUDE VERSION - GEMINI KNOWING THE API
import os
import time
import datetime
import warnings
import logging
import streamlit as st
from google import genai
from google.genai.types import GenerateContentConfig  # ← correct API usage
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Suppress noisy warnings ───────────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_DIR     = ".chroma_store"
RETRIEVAL_K    = 6      # always fetch top-6 chunks, no threshold gate
MAX_RETRIES    = 2      # streaming retry attempts before non-streaming fallback

# ── System instruction (passed via GenerateContentConfig, NOT in the prompt) ──
#
# Key findings from Gemini 2.5 Flash docs:
#   1. system_instruction is processed at a higher authority level than contents.
#      Embedding it in the prompt string treats it as user text → ignored.
#   2. Gemini 2.5 Flash is fine-tuned for instruction following with SHORT,
#      direct instructions. Long verbose prompts with headers hurt compliance.
#   3. For RAG grounding, Google's own recommended pattern is to explicitly
#      state "rely ONLY on the provided context" — Gemini respects this hard.
#   4. Language detection works reliably when stated as the FIRST sentence
#      in system_instruction, before any other rule.
#
SYSTEM_INSTRUCTION = """\
You are Lumi, the virtual concierge of Hotel Ducado, a luxury 5-star hotel.
Always reply in the exact same language the guest uses. No exceptions.
Answer only using the hotel information provided in the user message.
If the information does not contain the answer, apologize briefly and suggest the guest contact reception.
Be warm, concise, and elegant. Never repeat the guest's question. Maximum 3 short paragraphs."""

# ── User-turn template (context + question go here, not in system_instruction) ─
#
# Placing context in contents (not system_instruction) is correct:
# the model is designed to treat contents as the working material to reason over.
#
USER_TEMPLATE = """\
HOTEL INFORMATION:
{context}

GUEST MESSAGE:
{question}"""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Lumi — Hotel Ducado",
    page_icon="✨",
    layout="centered",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 3rem; padding-bottom: 5rem; }
    .stChatInputContainer {
        border-radius: 16px;
        border: 1px solid #444;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def write_log(message: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando modelo de embeddings…")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"},
    )


def load_all_documents() -> list:
    documents = []

    pdf_folder = "documentos_hotel"
    if os.path.exists(pdf_folder):
        for file in sorted(os.listdir(pdf_folder)):
            if file.endswith(".pdf"):
                try:
                    loader = PyMuPDFLoader(os.path.join(pdf_folder, file))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source_type"] = "manual"
                        doc.metadata["filename"]    = file
                    documents.extend(docs)
                    write_log(f"Loaded PDF: {file} ({len(docs)} pages)")
                except Exception as e:
                    write_log(f"Error loading {file}: {e}")

    csv_path = os.path.join("dades_hotel", "clients.csv")
    if os.path.exists(csv_path):
        try:
            loader = CSVLoader(file_path=csv_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_type"] = "client_data"
                doc.metadata["filename"]    = "clients.csv"
            documents.extend(docs)
            write_log(f"Loaded CSV: {len(docs)} rows")
        except Exception as e:
            write_log(f"Error loading clients.csv: {e}")

    return documents


@st.cache_resource(show_spinner="Indexando base de conocimiento…")
def load_vector_db(_embedding_model):
    """
    Fast path → load persisted Chroma index (~0.3 s).
    Slow path → build + persist (~8-12 s, first run only).
    Delete .chroma_store/ to force a rebuild after updating PDFs or CSV.
    """
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        write_log("Loading persisted Chroma index.")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_embedding_model,
        )

    write_log("Building Chroma index from scratch…")
    documents = load_all_documents()
    if not documents:
        write_log("ERROR: No documents found.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    write_log(f"Split into {len(chunks)} chunks.")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=_embedding_model,
        persist_directory=CHROMA_DIR,
    )
    db.persist()
    write_log("Chroma index built and persisted.")
    return db


@st.cache_resource(show_spinner=False)
def load_ai_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ══════════════════════════════════════════════════════════════════════════════
# RAG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_context(vector_db, query: str) -> list:
    """
    Always return the top RETRIEVAL_K most similar chunks, no score filtering.
    Chroma ranks. Gemini judges. Score filtering here only causes false negatives.
    Deduplicates chunks whose first 80 characters are identical.
    """
    results = vector_db.similarity_search_with_score(query, k=RETRIEVAL_K)

    seen, deduplicated = set(), []
    for doc, score in results:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            deduplicated.append((doc, score))

    return deduplicated


def format_context(chunks_with_scores: list) -> str:
    """Label each chunk with its source file for model traceability."""
    parts = []
    for i, (doc, _score) in enumerate(chunks_with_scores, 1):
        source = doc.metadata.get("filename", "hotel data")
        parts.append(f"[Source {i} — {source}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALL  — correct Gemini API usage with system_instruction
# ══════════════════════════════════════════════════════════════════════════════

def call_gemini(ai_client, user_content: str, placeholder) -> str | None:
    """
    Calls Gemini 2.5 Flash with system_instruction passed via GenerateContentConfig.
    This is the correct API usage: system_instruction is processed at a higher
    authority level than contents and cannot be overridden by user input.

    Streams tokens into an st.empty() placeholder (▌ cursor effect).
    Retries up to MAX_RETRIES times on streaming errors, then falls back to
    a single non-streaming call. Never uses st.write_stream() — it can drop
    the Streamlit websocket on slow or long responses.
    """
    config = GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
    )

    # ── Streaming attempts ────────────────────────────────────────────────────
    for attempt in range(MAX_RETRIES):
        full_text = ""
        try:
            response = ai_client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=user_content,
                config=config,
            )
            for chunk in response:
                if chunk.text:
                    full_text += chunk.text
                    placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)
            return full_text

        except Exception as e:
            write_log(f"Streaming attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5)

    # ── Non-streaming fallback ────────────────────────────────────────────────
    write_log("Falling back to non-streaming call.")
    try:
        response  = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=config,
        )
        full_text = response.text or ""
        placeholder.markdown(full_text)
        return full_text
    except Exception as e:
        write_log(f"Non-streaming fallback also failed: {e}")
        placeholder.error("Error de conexión. Por favor, inténtelo de nuevo.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def handle_user_message(pregunta: str, vector_db, ai_client) -> str | None:
    write_log(f"Query: {pregunta}")

    with st.spinner(""):
        chunks = retrieve_context(vector_db, pregunta)

    write_log(
        f"Retrieved {len(chunks)} chunks | "
        f"scores: {[round(s, 3) for _, s in chunks]}"
    )

    # Context goes into contents, not system_instruction
    user_content = USER_TEMPLATE.format(
        context=format_context(chunks),
        question=pregunta,
    )

    placeholder = st.empty()
    full_text   = call_gemini(ai_client, user_content, placeholder)

    if full_text:
        write_log(f"Response ({len(full_text)} chars): {full_text[:150]}")

    return full_text


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

hora = datetime.datetime.now().hour
if   6  <= hora < 14: saludo, icono = "Buenos días",   "☀️"
elif 14 <= hora < 20: saludo, icono = "Buenas tardes", "🌤️"
else:                  saludo, icono = "Buenas noches",  "🌙"

with st.sidebar:
    st.markdown("### ✨ Hotel Ducado")
    st.caption("Asistente virtual")
    st.divider()
    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Lumi está disponible 24 h para atender sus consultas.")

embedding_model = load_embedding_model()
vector_db       = load_vector_db(embedding_model)
ai_client       = load_ai_client()

if vector_db is None:
    st.error(
        "No se han encontrado documentos. "
        "Comprueba que las carpetas 'documentos_hotel/' y 'dades_hotel/' "
        "existen y contienen archivos, luego reinicia la aplicación."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown(f"""
        <div style='margin-top:18vh; margin-bottom:20vh; text-align:center;'>
            <h2 style='font-family:"Lora",serif; font-weight:400; font-size:2.6rem;
                       color:var(--text-color); letter-spacing:-0.5px;'>
                <span style='font-size:2.2rem; vertical-align:middle;
                             margin-right:12px;'>{icono}</span>
                {saludo}, soy Lumi
            </h2>
            <p style='font-family:"Inter",sans-serif; font-size:1.1rem;
                      color:var(--text-color); opacity:0.7; margin-top:-10px;'>
                ¿En qué puedo ayudarle hoy?
            </p>
        </div>
    """, unsafe_allow_html=True)

AVATARS = {"user": "👤", "assistant": "✨"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])

if pregunta := st.chat_input("Escribe tu consulta aquí…"):

    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar="✨"):
        result = handle_user_message(pregunta, vector_db, ai_client)

    if result:
        st.session_state.messages.append({"role": "assistant", "content": result})
