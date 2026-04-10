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
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

st.set_page_config(page_title="Lumi - Hotel Ducado", page_icon="✨", layout="centered")

# Disseny web
st.markdown("""
<style>
    /* Tipografia nativa del sistema (San Francisco en Apple, Roboto en Android) */
    html, body {
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
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

API_KEY = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def iniciar_sistema():
    # Llegim la clau de api
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
        f.write(f" {message}\n")

def stream_parser(response):
    for chunk in response:
        yield chunk.text

# Estètica depenent de l'hora
hora_actual = datetime.datetime.now().hour #23 #16

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
            
            prompt_final = f"""Ets la Lumi, l'assistent virtual i la intel·ligència d'atenció al client de l'Hotel Ducado. 
La teva missió és atendre els hostes amb la màxima amabilitat, calidesa i empatia, com si fossis un assistent virtual Premium d'un hotel de 5 estrelles.

REGLA D'OR D'IDENTITAT: Ets una IA neutral. NO t'identifiquis amb un gènere humà. 
MAI diguis "sóc la recepcionista" o "estic encantada". 
Fes servir sempre un llenguatge neutre o inclusiu per referir-te al teu càrrec: "Sóc la Lumi, assistent virtual de l'Hotel", "Estic aquí per ajudar-li", "Sóc la intel·ligència de recepció".

INSTRUCCIONS:
1. Respon SEMPRE en l'idioma i el to en què el client et pregunti.
2. Fes servir NOMÉS la informació que et dono aquí sota (prové dels manuals i del csv de clients).
3. Elabora respostes amables, directes i elegants. No et repeteixis a cada frase.
4. Si no saps la resposta amb aquesta informació, disculpa't amablement.

INFORMACIÓ DEL SISTEMA:
{context_text}

PREGUNTA DEL CLIENT:
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
