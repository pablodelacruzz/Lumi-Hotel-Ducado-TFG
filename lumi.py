import os
import time
import datetime
import warnings
import logging
import streamlit as st
from google import genai
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

CHROMA_DIR          = ".chroma_store"
RELEVANCE_THRESHOLD = 1.2    # only block truly unrelated chunks (cosine distance)
RETRIEVAL_K         = 8      # candidates fetched from Chroma
MAX_CONTEXT_CHUNKS  = 5      # max chunks forwarded to Gemini
MAX_RETRIES         = 2      # retry attempts on streaming failure

# Language rule placed FIRST and echoed at the END — Gemini cannot miss it.
SYSTEM_PROMPT = """\
CRITICAL LANGUAGE RULE — apply this before anything else:
Detect the language of the guest message and reply in THAT EXACT SAME LANGUAGE.
Spanish message → Spanish reply.
English message → English reply.
French message → French reply.
Catalan message → Catalan reply.
Never translate. Never switch language mid-response.

You are Lumi, the virtual concierge of Hotel Ducado — a luxury 5-star property.
Assist guests with warmth, elegance, and precision.

CONTENT RULES:
1. Answer ONLY using the HOTEL INFORMATION provided below.
   Never invent services, prices, room types, or policies.
2. Be warm, concise, and elegant — like a 5-star front-desk professional.
3. If the context partially answers the question, share what you know
   and gracefully note that the front desk can confirm the rest.
4. If the context does not answer the question at all, apologise briefly
   and invite the guest to contact reception.
5. Use short flowing paragraphs for conversational replies.
   Only use bullet lists when listing multiple items (room types, amenities).
6. Do NOT repeat the guest's question back to them.
7. Keep answers under 3 paragraphs unless more detail is clearly needed.

HOTEL INFORMATION:
{context}

REMINDER — the guest wrote in a specific language. Reply in THAT SAME LANGUAGE.
Guest message: {question}"""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the first Streamlit call)
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
# LOGGING  (never crashes the app)
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
    """Load PDFs and CSV, tagging each document with source metadata."""
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
    Fast path → load persisted Chroma index from disk  (~0.3 s).
    Slow path → build from scratch and persist          (~8-12 s, first run only).

    To force a full rebuild (e.g. after updating PDFs):
    delete the .chroma_store/ folder and restart the app.
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
    1. Fetch top RETRIEVAL_K chunks from Chroma.
    2. Filter out chunks above RELEVANCE_THRESHOLD (truly unrelated).
    3. Deduplicate near-identical chunks.
    4. Safety net: if everything was filtered, return the top-3 anyway
       so the LLM can decide relevance itself.
    """
    results = vector_db.similarity_search_with_score(query, k=RETRIEVAL_K)

    seen, filtered = set(), []
    for doc, score in results:
        if score > RELEVANCE_THRESHOLD:
            continue
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            filtered.append((doc, score))
        if len(filtered) >= MAX_CONTEXT_CHUNKS:
            break

    # Safety net — never send empty context to the LLM
    if not filtered:
        seen2 = set()
        for doc, score in results[:3]:
            key = doc.page_content[:80]
            if key not in seen2:
                seen2.add(key)
                filtered.append((doc, score))

    return filtered


def format_context(chunks_with_scores: list) -> str:
    """Structure retrieved chunks so the model can distinguish sources."""
    parts = []
    for i, (doc, _score) in enumerate(chunks_with_scores, 1):
        source = doc.metadata.get("filename", "hotel data")
        parts.append(f"[Source {i} — {source}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════════════════════════════════════

def call_gemini(ai_client, prompt: str, placeholder) -> str | None:
    """
    Tries streaming first (live token-by-token effect).
    On failure retries up to MAX_RETRIES times, then falls back to a single
    non-streaming call. Uses st.empty() placeholder — no st.write_stream,
    which can cause Streamlit connection drops on long responses.
    Returns the full response text or None on total failure.
    """
    # ── Streaming attempts ────────────────────────────────────────────────────
    for attempt in range(MAX_RETRIES):
        full_text = ""
        try:
            response = ai_client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            for chunk in response:
                if chunk.text:
                    full_text += chunk.text
                    # ▌ cursor makes it feel live without using write_stream
                    placeholder.markdown(full_text + "▌")
            placeholder.markdown(full_text)  # final render, remove cursor
            return full_text

        except Exception as e:
            write_log(f"Streaming attempt {attempt + 1} error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5)

    # ── Non-streaming fallback ────────────────────────────────────────────────
    write_log("All streaming attempts failed — falling back to non-streaming.")
    try:
        response  = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        full_text = response.text or ""
        placeholder.markdown(full_text)
        return full_text
    except Exception as e:
        write_log(f"Non-streaming fallback failed: {e}")
        placeholder.error("Error de conexión. Por favor, inténtelo de nuevo en unos instantes.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def handle_user_message(pregunta: str, vector_db, ai_client) -> str | None:
    write_log(f"Query: {pregunta}")

    # Retrieval (spinner is subtle — just a small indicator, no jarring text)
    with st.spinner(""):
        chunks = retrieve_context(vector_db, pregunta)

    write_log(
        f"Retrieved {len(chunks)} chunks | "
        f"scores: {[round(s, 3) for _, s in chunks]}"
    )

    prompt    = SYSTEM_PROMPT.format(
        context=format_context(chunks),
        question=pregunta,
    )
    placeholder = st.empty()
    full_text   = call_gemini(ai_client, prompt, placeholder)

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✨ Hotel Ducado")
    st.caption("Asistente virtual")
    st.divider()
    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Lumi está disponible 24 h para atender sus consultas.")

# ── Load resources ────────────────────────────────────────────────────────────
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

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Welcome screen (only shown before first message) ─────────────────────────
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

# ── Render conversation history ───────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if pregunta := st.chat_input("Escribe tu consulta aquí…"):

    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar="✨"):
        result = handle_user_message(pregunta, vector_db, ai_client)

    if result:
        st.session_state.messages.append({"role": "assistant", "content": result})
