import os
import datetime
import warnings
import logging
import streamlit as st
from google import genai
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Suppress noisy warnings ──────────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ── Page config ───────────────────────────────────────────────────────────────
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
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_DIR        = ".chroma_store"   # persisted index — survives restarts
RELEVANCE_THRESHOLD = 0.45            # cosine distance cutoff (lower = more similar)
RETRIEVAL_K       = 8                 # candidates fetched from Chroma
MAX_CONTEXT_CHUNKS = 5               # max chunks passed to Gemini

SYSTEM_PROMPT = """\
You are Lumi, the virtual concierge of Hotel Ducado — a luxury 5-star property.
Your role is to assist guests with warmth, elegance, and absolute precision.

LANGUAGE RULE (non-negotiable):
  Respond in the EXACT language the guest uses. If they write in Spanish → reply in Spanish.
  English → English. French → French. Never mix languages.

RESPONSE RULES:
  1. Answer ONLY using the hotel information provided in the CONTEXT section below.
     Never invent services, prices, room types, or policies.
  2. Be warm, concise, and elegant — like a 5-star front-desk professional.
  3. If the context partially answers the question, give what you know and
     gracefully acknowledge what you cannot confirm.
  4. If the context does not answer the question, apologise briefly and invite
     the guest to contact the front desk for further assistance.
  5. Use short, flowing paragraphs for conversational replies.
     Only use lists when the question explicitly calls for one (e.g. room types, amenities).
  6. Do NOT repeat the guest's question back to them.
  7. Keep answers under 3 paragraphs unless a longer answer is clearly needed.

CONTEXT:
{context}

Guest message: {question}"""


# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES  (built once per process, survive Streamlit reruns)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Cargando modelo de embeddings…")
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"},
    )


def load_all_documents() -> list:
    """Load PDFs + CSV, tagging each chunk with source metadata."""
    documents = []

    pdf_folder = "documentos_hotel"
    if os.path.exists(pdf_folder):
        for file in sorted(os.listdir(pdf_folder)):
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(os.path.join(pdf_folder, file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_type"] = "manual"
                    doc.metadata["filename"]    = file
                documents.extend(docs)

    csv_path = os.path.join("dades_hotel", "clients.csv")
    if os.path.exists(csv_path):
        loader = CSVLoader(file_path=csv_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = "client_data"
            doc.metadata["filename"]    = "clients.csv"
        documents.extend(docs)

    return documents


@st.cache_resource(show_spinner="Indexando base de conocimiento…")
def load_vector_db(_embedding_model):
    """
    Fast path  → load persisted Chroma index from disk  (~0.3 s).
    Slow path  → build from scratch and persist          (~8–12 s first run).
    """
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_embedding_model,
        )

    documents = load_all_documents()
    if not documents:
        st.warning("No se encontraron documentos en 'documentos_hotel/' ni 'dades_hotel/'.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=_embedding_model,
        persist_directory=CHROMA_DIR,
    )
    db.persist()
    return db


@st.cache_resource(show_spinner=False)
def load_ai_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ══════════════════════════════════════════════════════════════════════════════
# RAG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def filter_and_deduplicate(results: list, threshold: float, max_k: int) -> list:
    """
    1. Keep only chunks below the relevance threshold.
    2. If nothing passes, fall back to the top-2 candidates.
    3. Deduplicate near-identical chunks (same leading 80 chars).
    4. Return at most max_k items.
    """
    relevant = [(doc, score) for doc, score in results if score < threshold]
    if not relevant:
        relevant = results[:2]  # graceful fallback

    seen, deduplicated = set(), []
    for doc, score in relevant[:max_k]:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            deduplicated.append((doc, score))

    return deduplicated


def is_context_sufficient(chunks_with_scores: list, threshold: float) -> bool:
    """True only when at least one chunk is clearly on-topic."""
    if not chunks_with_scores:
        return False
    best_score = min(score for _, score in chunks_with_scores)
    return best_score < threshold


def format_context(chunks_with_scores: list) -> str:
    """
    Structure context blocks so the model can distinguish sources.
    Each block is labelled and separated by a visual divider.
    """
    parts = []
    for i, (doc, score) in enumerate(chunks_with_scores, 1):
        source = doc.metadata.get("filename", "hotel data")
        parts.append(f"[Source {i} — {source}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING
# ══════════════════════════════════════════════════════════════════════════════

def stream_parser(response):
    """Yield text chunks from a Gemini streaming response, skipping empty ones."""
    for chunk in response:
        if chunk.text:
            yield chunk.text


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def write_log(message: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def handle_user_message(pregunta: str, vector_db, ai_client) -> str | None:
    """
    Full RAG + LLM pipeline for one conversational turn.
    Returns the assistant's full response text, or None on error.
    """
    write_log(f"Query: {pregunta}")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    with st.spinner("Consultando los registros del hotel…"):
        raw_results = vector_db.similarity_search_with_score(pregunta, k=RETRIEVAL_K)
        relevant    = filter_and_deduplicate(raw_results, RELEVANCE_THRESHOLD, MAX_CONTEXT_CHUNKS)

    # ── Insufficient context guard ────────────────────────────────────────────
    if not is_context_sufficient(relevant, RELEVANCE_THRESHOLD):
        msg = (
            "Mis disculpas, no encuentro esa información en mis registros. "
            "Nuestro equipo en recepción estará encantado de ayudarle con cualquier consulta."
        )
        st.markdown(msg)
        write_log(f"Fallback (no relevant context). Best score: "
                  f"{min(s for _,s in raw_results):.3f}" if raw_results else "no results")
        return msg

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = SYSTEM_PROMPT.format(
        context=format_context(relevant),
        question=pregunta,
    )

    # ── Generate & stream ─────────────────────────────────────────────────────
    try:
        response   = ai_client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        full_text  = st.write_stream(stream_parser(response))
        write_log(f"Response ({len(full_text)} chars): {full_text[:120]}…")
        return full_text

    except Exception as e:
        write_log(f"ERROR: {e}")
        st.error("Error de conexión. Por favor, inténtelo de nuevo.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# Time-aware greeting
hora = datetime.datetime.now().hour
if 6 <= hora < 14:
    saludo, icono = "Buenos días",   "☀️"
elif 14 <= hora < 20:
    saludo, icono = "Buenas tardes", "🌤️"
else:
    saludo, icono = "Buenas noches", "🌙"

# Sidebar
with st.sidebar:
    st.markdown("### ✨ Hotel Ducado")
    st.caption("Asistente virtual")
    st.divider()
    if st.button("Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Lumi está disponible 24h para atender sus consultas.")

# Load resources
embedding_model = load_embedding_model()
vector_db       = load_vector_db(embedding_model)
ai_client       = load_ai_client()

if vector_db is None:
    st.error("No se han encontrado documentos del hotel. "
             "Asegúrate de que las carpetas 'documentos_hotel/' y 'dades_hotel/' existen.")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome screen (only shown before first message)
if len(st.session_state.messages) == 0:
    st.markdown(f"""
        <div style='margin-top:18vh; margin-bottom:20vh; text-align:center;'>
            <h2 style='font-family:"Lora",serif; font-weight:400; font-size:2.6rem;
                       color:var(--text-color); letter-spacing:-0.5px;'>
                <span style='font-size:2.2rem; vertical-align:middle; margin-right:12px;'>{icono}</span>
                {saludo}, soy Lumi
            </h2>
            <p style='font-family:"Inter",sans-serif; font-size:1.1rem;
                      color:var(--text-color); opacity:0.7; margin-top:-10px;'>
                ¿En qué puedo ayudarle hoy?
            </p>
        </div>
    """, unsafe_allow_html=True)

AVATARS = {"user": "👤", "assistant": "✨"}

# Render conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])

# Chat input
if pregunta := st.chat_input("Escribe tu consulta aquí…"):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)

    # Generate and stream assistant response
    with st.chat_message("assistant", avatar="✨"):
        result = handle_user_message(pregunta, vector_db, ai_client)

    if result:
        st.session_state.messages.append({"role": "assistant", "content": result})
