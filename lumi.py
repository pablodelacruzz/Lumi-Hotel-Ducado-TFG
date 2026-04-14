import os
import time
import datetime
import warnings
import logging
import re   # Per extreure les dades ocultes de la resposta (metadades)
import streamlit as st
from google import genai
from google.genai.types import GenerateContentConfig
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Configuració de seguretat i silenci de warnings ──────────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS I CONFIGURACIÓ DEL SISTEMA
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_DIR     = ".chroma_store_v2" 
RETRIEVAL_K    = 4      
MAX_RETRIES    = 2      

# Diccionari de Seguretat per a les habitacions (Tokens)
ROOM_TOKENS = {
    "tk_a1b2c": "Hab_201",
    "tk_x9y8z": "Hab_202",
    "tk_m4n5p": "Hab_203"
}

# ── Instruccions de Sistema (Prompt Engineering) ──────────────────────────────
# ARREGLAT: Hem tornat a posar la plantilla HTML exacta perquè Gemini sàpiga com fer-ho.
SYSTEM_INSTRUCTION = """\
You are Lumi, the virtual concierge of Hotel Ducado, a luxury 5-star hotel.
CRITICAL RULES:
1. Always reply in the EXACT same language the guest uses. No exceptions.
2. Answer ONLY using the hotel information provided in the user message. Do not invent details or prices.
3. If the information does not contain the answer, apologize politely and suggest the guest contact the physical reception desk.
4. Tone: Be exceptionally warm, empathetic, and elegant. Act like a high-end concierge. Elaborate slightly to sound conversational and helpful, but do not exceed 3 paragraphs. Never repeat the guest's question.
5. MANDATORY METADATA: You MUST start your response with an exact HTML comment containing the language and category.
EXAMPLE FORMAT: Replace "ES" with the 2-letter code of the user's language. 
Replace "MENJAR" with the most fitting category: MENJAR, SERVEIS, HORARIS, NORMES, PROBLEMA, ALTRES.
After the HTML comment, start a new line and write your natural response to the guest."""

USER_TEMPLATE = """\
HOTEL INFORMATION:
{context}

GUEST MESSAGE:
{question}"""

# ══════════════════════════════════════════════════════════════════════════════
# DISSENY I ESTIL (INTERFÍCIE APPLE)
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
# LOGS I ANALÍTICA
# ══════════════════════════════════════════════════════════════════════════════

def write_log(message: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception:
        pass

def log_analytics(room: str, question: str, response_text: str):
    """Guarda un registre de text pla per a l'anàlisi de negoci."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lang = "Desconegut"
    category = "Desconeguda"
    
    try:
        # ARREGLAT: Hem tornat a posar l'expressió regular completa!
        match = re.search(r'', response_text, re.IGNORECASE)
        if match and len(match.groups()) >= 2:
            lang = match.group(1).strip()
            category = match.group(2).strip()
    except Exception as e:
        write_log(f"Error parsejant metadades: {e}")
        
    linia_log = f"[{ts}] | Habitació: {room} | Idioma: {lang} | Categoria: {category} | Pregunta: {question}\n"
    
    # També ho traiem per consola perquè es vegi al "Manage App" de Streamlit Cloud
    print(linia_log)
        
    try:
        with open("log_consultes.txt", "a", encoding="utf-8") as f:
            f.write(linia_log)
    except Exception as e:
        write_log(f"Error guardant log_consultes: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# LÒGICA RAG (EMBEDDINGS I VECTOR DB)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Preparant intel·ligència...")
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
        except Exception:
            pass
    return documents

@st.cache_resource(show_spinner="Indexant manuals de l'hotel...")
def load_vector_db(_embedding_model):
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=_embedding_model)

    documents = load_all_documents()
    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  
        chunk_overlap=128, 
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    
    db = Chroma.from_documents(documents=chunks, embedding=_embedding_model, persist_directory=CHROMA_DIR)
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
# GENERACIÓ DE RESPOSTA (GEMINI)
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
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.5)

    try:
        response  = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=config,
        )
        full_text = response.text or ""
        placeholder.markdown(full_text)
        return full_text
    except Exception:
        placeholder.error("Error de connexió.")
        return None

def handle_user_message(pregunta: str, vector_db, ai_client) -> str | None:
    with st.spinner("Pensant..."):
        chunks = retrieve_context(vector_db, pregunta)
    
    user_content = USER_TEMPLATE.format(
        context=format_context(chunks),
        question=pregunta,
    )
    placeholder = st.empty()
    return call_gemini(ai_client, user_content, placeholder)

# ══════════════════════════════════════════════════════════════════════════════
# EXECUCIÓ PRINCIPAL (APP)
# ══════════════════════════════════════════════════════════════════════════════

# 1. Identificació de l'habitació via URL
query_params = st.query_params
url_token = query_params.get("key", "default")
current_room = ROOM_TOKENS.get(url_token, "Recepció / General")

# 2. Salutació segons l'hora
hora = datetime.datetime.now().hour
if   6  <= hora < 14: saludo, icono = "Buenos días",   "☀️"
elif 14 <= hora < 20: saludo, icono = "Buenas tardes", "🌤️"
else:                 saludo, icono = "Buenas noches", "🌙"

# 3. Càrrega de recursos
embedding_model = load_embedding_model()
vector_db       = load_vector_db(embedding_model)
ai_client       = load_ai_client()

if vector_db is None:
    st.error("No s'han trobat documents.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Pantalla de benvinguda
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

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])

# 5. Entrada de l'usuari
if pregunta := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar="✨"):
        result = handle_user_message(pregunta, vector_db, ai_client)

    if result:
        st.session_state.messages.append({"role": "assistant", "content": result})
        # Registrar dades de negoci
        log_analytics(current_room, pregunta, result)

# ══════════════════════════════════════════════════════════════════════════════
# PANELL D'ADMINISTRACIÓ (NOMÉS PER A LA REUNIÓ AMB L'HÈCTOR)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 Panell de Control (Tech)")
    st.caption(f"📍 Habitació detectada: **{current_room}**")
    st.divider()
    
    st.markdown("#### Registre de Consultes (Text Pla)")
    if st.button("🔄 Refrescar Registre"):
        try:
            with open("log_consultes.txt", "r", encoding="utf-8") as f:
                contingut_log = f.read()
            if contingut_log.strip() == "":
                st.info("L'arxiu de registre existeix, però està buit.")
            else:
                st.text_area("Contingut de 'log_consultes.txt':", value=contingut_log, height=400)
        except FileNotFoundError:
            st.warning("L'arxiu encara no s'ha creat. Fes una pregunta al xat per generar-lo.")
