import os
import time
import datetime
import warnings
import logging
import streamlit as st
from google import genai
from google.genai.types import GenerateContentConfig  # Ús correcte de l'API de Gemini
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Treure Warnings de terminal ───────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS I CONFIGURACIÓ RAG
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_DIR     = ".chroma_store"
RETRIEVAL_K    = 4      # Portem els 4 millors fragments (com que ara són més grans, 4 és ideal)
MAX_RETRIES    = 2      # Intents de streaming abans de fer fallback

# ── Instruccions de Sistema (L'ànima de la Lumi) ──────────────────────────────
SYSTEM_INSTRUCTION = """\
You are Lumi, the virtual concierge of Hotel Ducado, a luxury 5-star hotel.
CRITICAL RULES:
1. Always reply in the EXACT same language the guest uses. No exceptions.
2. Answer ONLY using the hotel information provided in the user message. Do not invent details or prices.
3. If the information does not contain the answer, apologize politely and suggest the guest contact the physical reception desk.
4. Tone: Be exceptionally warm, empathetic, and elegant. Act like a high-end concierge. Elaborate slightly to sound conversational and helpful, but do not exceed 3 paragraphs. Never repeat the guest's question."""

# ── Plantilla de l'usuari (Aquí van les dades del RAG) ────────────────────────
USER_TEMPLATE = """\
HOTEL INFORMATION:
{context}

GUEST MESSAGE:
{question}"""

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓ DE LA PÀGINA I DISSENY (ESTIL APPLE)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Lumi — Hotel Ducado", page_icon="✨", layout="centered")

st.markdown("""
<style>
    html, body, [class*="css"] { 
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Helvetica Neue", sans-serif !important; 
        letter-spacing: -0.015em;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 3rem; padding-bottom: 5rem; }
    .stChatInputContainer {
        border-radius: 20px;
        border: 1px solid #444;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONS BÀSIQUES I CACHE
# ══════════════════════════════════════════════════════════════════════════════

def write_log(message: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception:
        pass

@st.cache_resource(show_spinner="Carregant model d'intel·ligència...")
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
                    write_log(f"PDF Carregat: {file} ({len(docs)} pàgines)")
                except Exception as e:
                    write_log(f"Error carregant {file}: {e}")

    csv_path = os.path.join("dades_hotel", "clients.csv")
    if os.path.exists(csv_path):
        try:
            loader = CSVLoader(file_path=csv_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_type"] = "client_data"
                doc.metadata["filename"]    = "clients.csv"
            documents.extend(docs)
        except Exception as e:
            write_log(f"Error carregant clients.csv: {e}")

    return documents

@st.cache_resource(show_spinner="Indexant la base de coneixement...")
def load_vector_db(_embedding_model):
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        write_log("Carregant base de dades Chroma existent.")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=_embedding_model)

    write_log("Creant base de dades Chroma des de zero...")
    documents = load_all_documents()
    if not documents:
        return None

    # TALLAT DE TEXT OPTIMITZAT: Fragments més grans per no perdre context (ex: Planxa)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  
        chunk_overlap=128, 
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    
    db = Chroma.from_documents(documents=chunks, embedding=_embedding_model, persist_directory=CHROMA_DIR)
    write_log("Base de dades creada i guardada.")
    return db

@st.cache_resource(show_spinner=False)
def load_ai_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def retrieve_context(vector_db, query: str) -> list:
    results = vector_db.similarity_search_with_score(query, k=RETRIEVAL_K)
    seen, deduplicated = set(), []
    for doc, score in results:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            deduplicated.append((doc, score))
    return deduplicated

def format_context(chunks_with_scores: list) -> str:
    parts = []
    for i, (doc, _score) in enumerate(chunks_with_scores, 1):
        source = doc.metadata.get("filename", "hotel data")
        parts.append(f"[Source {i} — {source}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)

# ══════════════════════════════════════════════════════════════════════════════
# TRUCADA A GEMINI (AMB PROTECCIÓ ANTI-CRASH)
# ══════════════════════════════════════════════════════════════════════════════

def call_gemini(ai_client, user_content: str, placeholder) -> str | None:
    config = GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)

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
            write_log(f"Error de streaming (intent {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5)

    write_log("Intentant mètode sense streaming (fallback)...")
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
        placeholder.error(f"Error de connexió: {e}")
        return None

def handle_user_message(pregunta: str, vector_db, ai_client) -> str | None:
    write_log(f"Consulta: {pregunta}")
    with st.spinner("Pensant..."):
        chunks = retrieve_context(vector_db, pregunta)
    
    user_content = USER_TEMPLATE.format(
        context=format_context(chunks),
        question=pregunta,
    )
    placeholder = st.empty()
    return call_gemini(ai_client, user_content, placeholder)

# ══════════════════════════════════════════════════════════════════════════════
# INTERFÍCIE D'USUARI (FRONT-END)
# ══════════════════════════════════════════════════════════════════════════════

hora = datetime.datetime.now().hour
if   6  <= hora < 14: saludo, icono = "Buenos días",   "☀️"
elif 14 <= hora < 20: saludo, icono = "Buenas tardes", "🌤️"
else:                 saludo, icono = "Buenas noches", "🌙"

embedding_model = load_embedding_model()
vector_db       = load_vector_db(embedding_model)
ai_client       = load_ai_client()

if vector_db is None:
    st.error("No s'han trobat documents a la carpeta 'documentos_hotel/'.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown(f"""
        <div style='margin-top:18vh; margin-bottom:20vh; text-align:center;'>
            <h2 style='font-weight:600; font-size:2.6rem; color:var(--text-color); letter-spacing:-0.04em;'>
                <span style='font-size:2.2rem; vertical-align:middle; margin-right:12px;'>{icono}</span>{saludo}, soy Lumi
            </h2>
            <p style='font-weight:400; font-size:1.1rem; color:var(--text-color); opacity:0.6; margin-top:-10px; letter-spacing:-0.01em;'>
                ¿En qué puedo asistirle hoy?
            </p>
        </div>
    """, unsafe_allow_html=True)

AVATARS = {"user": "👤", "assistant": "✨"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])

if pregunta := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar="✨"):
        result = handle_user_message(pregunta, vector_db, ai_client)

    if result:
        st.session_state.messages.append({"role": "assistant", "content": result})
