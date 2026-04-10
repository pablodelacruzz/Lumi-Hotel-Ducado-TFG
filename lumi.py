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

# treure Warnings de terminal
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

st.set_page_config(page_title="Lumi - Hotel Ducado", page_icon="✨", layout="centered")

# Disseny web
st.markdown("""
<style>
    /* Tipografia nativa del sistema (San Francisco en Apple, Roboto en Android) */
    html, body, {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif !important;
        letter-spacing: -0.015em;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 5rem;
    }
    
    .stChatInputContainer {
        border-radius: 20px; /* Corba més suau estil iOS */
        border: 1px solid #444; 
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.08);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def iniciar_sistema():
    # Llegim la clau des dels "secrets" de Streamlit en lloc de posar-la al codi
    API_KEY = st.secrets["GEMINI_API_KEY"] 
    ai_client = genai.Client(api_key=API_KEY)
    
    documents = []
    
    pdf_folder = "documentos_hotel"
    if os.path.exists(pdf_folder):
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(os.path.join(pdf_folder, file))
                documents.extend(loader.load())
                
    csv_path = os.path.join("dades_hotel", "clients.csv")
    if os.path.exists(csv_path):
        loader = CSVLoader(file_path=csv_path, encoding="utf-8")
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma.from_documents(documents=chunks, embedding=embedding_model) 
    
    return db, ai_client

vector_db, model = iniciar_sistema()

def write_log(message):
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}\n")

def stream_parser(response):
    for chunk in response:
        try:
            # Intentem extreure el text del paquet
            if chunk.text:
                yield chunk.text
        except Exception:
            # Si el paquet ve buit o amb dades internes de Google, l'ignorem i continuem
            continue

# Estètica depenent de l'hora
hora_actual = 16 #datetime.datetime.now().hour #23 #16

if 6 <= hora_actual < 14:
    saludo = "Buenos días"
    icono_tiempo = "☀️"
elif 14 <= hora_actual < 20:
    saludo = "Buenas tardes"
    icono_tiempo = "🌤️"
else:
    saludo = "Buenas noches"
    icono_tiempo = "🌙"

# interfície del xat
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown(f"""
        <div style='margin-top: 18vh; margin-bottom: 20vh; text-align: center;'>
            <h2 style='font-weight: 600; font-size: 2.6rem; color: var(--text-color); letter-spacing: -0.04em;'>
                <span style='font-size: 2.2rem; vertical-align: middle; margin-right: 12px;'>{icono_tiempo}</span>{saludo}, soy Lumi
            </h2>
            <p style='font-weight: 400; font-size: 1.1rem; color: var(--text-color); opacity: 0.6; margin-top: -10px; letter-spacing: -0.01em;'>¿En qué puedo asistirle hoy?</p>
        </div>
    """, unsafe_allow_html=True)


avatars = {"user": "👤", "assistant": "✨"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatars.get(message["role"])):
        st.markdown(message["content"])

# lógica resposta IA
pregunta = st.chat_input("Escribe tu consulta aquí...")

if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)
        
    write_log(f"Consulta: {pregunta}")

    results = vector_db.similarity_search_with_score(pregunta, k=3) 
    context_recuperat = [doc.page_content for doc, score in results]
    
    with st.chat_message("assistant", avatar="✨"):
        if not context_recuperat:
            resposta_buida = "Mis disculpas, no encuentro esta información en mis registros. Por favor, consulte con nuestro equipo en la recepción física para que puedan ayudarle."
            st.markdown(resposta_buida)
            st.session_state.messages.append({"role": "assistant", "content": resposta_buida})
        else:
            context_text = "\n".join(context_recuperat)
            prompt_final = f"""You are Lumi, the virtual assistant and customer service AI for Hotel Ducado. 
Your mission is to assist guests with the utmost politeness, warmth, and empathy, acting as a Premium virtual assistant for a 5-star hotel.

GOLDEN RULE OF IDENTITY: You are a neutral AI. DO NOT identify with any human gender. 
NEVER say "I am the receptionist" or use gendered adjectives to describe yourself. 
Always use neutral language for your role: "I am Lumi, the virtual assistant of the Hotel", "I am here to help you", or "I am the reception intelligence".

STRICT INSTRUCTIONS:
1. ALWAYS respond in the exact language and tone that the guest uses to ask the question.
2. Use ONLY the information provided in the SYSTEM INFORMATION below. Do not invent or assume any details outside of this context.
3. Craft polite, direct, and elegant responses. 
4. If the answer is not contained in the SYSTEM INFORMATION, apologize politely and ask the guest to contact the physical reception desk.

SYSTEM INFORMATION:
{context_text}

GUEST QUESTION:
{pregunta}"""
            
            try:
                response = model.models.generate_content_stream(
                    model='gemini-2.5-flash',
                    contents=prompt_final
                )
                text_complet = st.write_stream(stream_parser(response))
                
                st.session_state.messages.append({"role": "assistant", "content": text_complet})
                write_log(f"Resposta: {text_complet}")
            except Exception as e:
                st.error("Error de conexión. Por favor, inténtelo de nuevo.")
                write_log(f"ERROR: {e}")
