import os
import time
import datetime
import warnings
import logging
import re
import streamlit as st
from google import genai
from google.genai.types import GenerateContentConfig
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Treure Warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# CONSTANTS I CONFIG
CHROMA_DIR     = ".chroma_store_v2" 
RETRIEVAL_K    = 4      
MAX_RETRIES    = 3      # HEM PUJAT ELS REINTENTS A 3 PER EVITAR TALLS

# Codi habitacions URL 
ROOM_TOKENS = {
    # P1
    "tk_1q2w3": "Hab_101",
    "tk_4e5r6": "Hab_102",
    "tk_7t8y9": "Hab_103",
    "tk_0u1i2": "Hab_104",
    "tk_3o4p5": "Hab_105",
    "tk_6a7s8": "Hab_106",
    "tk_9d0f1": "Hab_107",
    "tk_2g3h4": "Hab_108",
    "tk_5j6k7": "Hab_109",
    "tk_8l9z0": "Hab_110",
    "tk_1x2c3": "Hab_111",

    # P2
    "tk_a1b2c": "Hab_201",  
    "tk_x9y8z": "Hab_202",  
    "tk_m4n5p": "Hab_203", 
    "tk_v4b5n": "Hab_204",
    "tk_m6q7w": "Hab_205",
    "tk_e8r9t": "Hab_206",
    "tk_y1u2i": "Hab_207",
    "tk_o3p4a": "Hab_208",
    "tk_s5d6f": "Hab_209",
    "tk_g7h8j": "Hab_210",
    "tk_k9l1z": "Hab_211",

    # P3
    "tk_x2c3v": "Hab_301",
    "tk_b4n5m": "Hab_302",
    "tk_q6w7e": "Hab_303",
    "tk_r8t9y": "Hab_304",
    "tk_u1i2o": "Hab_305",
    "tk_p3a4s": "Hab_306",
    "tk_d5f6g": "Hab_307",
    "tk_h7j8k": "Hab_308",
    "tk_l9z1x": "Hab_309",
    "tk_c2v3b": "Hab_310",
    "tk_n4m5q": "Hab_311",

    # P4
    "tk_w1e2r": "Hab_401",
    "tk_t3y4u": "Hab_402",
    "tk_i5o6p": "Hab_403",
    "tk_a7s8d": "Hab_404",
    "tk_f9g1h": "Hab_405",
    "tk_j2k3l": "Hab_406",
    "tk_z4x5c": "Hab_407",
    "tk_v6b7n": "Hab_408",
    "tk_m8q9w": "Hab_409",
    "tk_e1r2t": "Hab_410",
    "tk_y3u4i": "Hab_411",

    # P5
    "tk_o5p6a": "Hab_501",
    "tk_s7d8f": "Hab_502",
    "tk_g9h1j": "Hab_503",
    "tk_k2l3z": "Hab_504",
    "tk_x4c5v": "Hab_505",
    "tk_b6n7m": "Hab_506",
    "tk_q8w9e": "Hab_507",
    "tk_r1t2y": "Hab_508",
    "tk_u3i4o": "Hab_509",
    "tk_p5a6s": "Hab_510",
    "tk_d7f8g": "Hab_511",

    # P6
    "tk_h9j1k": "Hab_601",
    "tk_l2z3x": "Hab_602",
    "tk_c4v5b": "Hab_603",
    "tk_n6m7q": "Hab_604",
    "tk_w8e9r": "Hab_605",
    "tk_t1y2u": "Hab_606",
    "tk_i3o4p": "Hab_607",
    "tk_a5s6d": "Hab_608",
    "tk_f7g8h": "Hab_609",
    "tk_j9k1l": "Hab_610",
    "tk_z2x3c": "Hab_611",

    # P7
    "tk_v4b9n": "Hab_701",
    "tk_m6q2w": "Hab_702",
    "tk_e8r4t": "Hab_703",
    "tk_y1u6i": "Hab_704",
    "tk_o3p8a": "Hab_707",
    "tk_s5d0f": "Hab_708",
    "tk_g7h2j": "Hab_710",
    "tk_k9l4z": "Hab_711",

    # SUITE PRESIDENCIAL
    "tk_suite8": "Hab_801"
}

# PROMPT GEMINI (NO TOCAR!)
SYSTEM_INSTRUCTION = """\
You are Lumi, the virtual concierge of Hotel Ducado, a luxury 5-star hotel.
CRITICAL RULES:
1. Always reply in the EXACT same language the guest uses. No exceptions.
2. Answer ONLY using the hotel information provided in the user message. Do not invent details or prices.
3. If the information does not contain the answer, apologize politely and suggest the guest contact the physical reception desk.
4. Tone: Be exceptionally warm, empathetic, and elegant. Max 3 paragraphs.
5. METADATA (CRITICAL): At the VERY END of your entire response, you MUST append the language and category wrapped in double pipes.
FORMAT EXACTLY LIKE THIS: ||ES, MENJAR||
Language codes: ES, EN, CA, FR, DE...
Categories allowed: MENJAR, SERVEIS, HORARIS, NORMES, PROBLEMA, ALTRES."""

USER_TEMPLATE = """\
HOTEL INFORMATION:
{context}

GUEST MESSAGE:
{question}"""


# DISSENY WEB
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


# ESCRIURE LOGS AL .txt
def write_log(message: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception:
        pass

def log_analytics(room: str, question: str, response_text: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lang = "Desconegut"
    category = "Desconeguda"
    
    try:
        match = re.search(r'\|\|(.*?)\|\|', response_text)
        if match:
            dades = match.group(1).split(',')
            if len(dades) >= 2:
                lang = dades[0].strip().upper()
                category = dades[1].strip().upper()
    except Exception as e:
        pass
        
    linia_log = f"[{ts}] | Habitació: {room} | Idioma: {lang} | Categoria: {category} | Pregunta: {question}\n"
        
    try:
        with open("log_consultes.txt", "a", encoding="utf-8") as f:
            f.write(linia_log)
    except Exception as e:
        pass


# RAG INICIAL BASE
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = splitter.split_documents(documents)
    return Chroma.from_documents(documents=chunks, embedding=_embedding_model, persist_directory=CHROMA_DIR)

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
        parts.append(f"[Fragment {i}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# --- GENERATOR A GEMINI (AMB SISTEMA DE REINTENTS ROBUST) ---
def call_gemini(ai_client, user_content: str, placeholder) -> str | None:
    config = GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
    
    # Bucle de reintents per evitar errors de connexió
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
                    # Ocultem les dades || mentre escriu
                    display_text = full_text.split("||")[0].strip()
                    placeholder.markdown(display_text + "▌")
            
            # Resultat final net
            display_text = full_text.split("||")[0].strip()
            placeholder.markdown(display_text)
            
            return full_text # Retornem el text amb els || per extreure al log
            
        except Exception as e:
            # Si estem en el primer o segon intent, esperem 2 segons i ho tornem a provar
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
                continue
            else:
                placeholder.error("Error de connexió amb Gemini. Si us plau, torna a provar-ho.")
                return None

def handle_user_message(pregunta: str, vector_db, ai_client) -> str | None:
    with st.spinner("Pensant..."):
        chunks = retrieve_context(vector_db, pregunta)
    
    user_content = USER_TEMPLATE.format(context=format_context(chunks), question=pregunta)
    placeholder = st.empty()
    return call_gemini(ai_client, user_content, placeholder)


# EXECUCIÓ PRINCIPAL (APP)
query_params = st.query_params
url_token = query_params.get("key", "default")

# PANELL ADMIN HOTEL
is_admin = (url_token == "admin")
current_room = "Administració" if is_admin else ROOM_TOKENS.get(url_token, "Recepció / General")

hora = datetime.datetime.now().hour
if   6  <= hora < 14: saludo, icono = "Buenos días",   "☀️"
elif 14 <= hora < 20: saludo, icono = "Buenas tardes", "🌤️"
else:                 saludo, icono = "Buenas noches", "🌙"

embedding_model = load_embedding_model()
vector_db       = load_vector_db(embedding_model)
ai_client       = load_ai_client()

if vector_db is None:
    st.error("No s'han trobat documents.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Netejar l'historial del xat perquè l'usuari no vegi mai els ||
def clean_message(text):
    return text.split("||")[0].strip()

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
        st.markdown(clean_message(message["content"]))

if pregunta := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar="✨"):
        result = handle_user_message(pregunta, vector_db, ai_client)

    if result:
        st.session_state.messages.append({"role": "assistant", "content": result})
        log_analytics(current_room, pregunta, result)


# PANELL PER L'HOTEL i ADMIN PANEL + DASHBOARD
if is_admin:
    import pandas as pd
    import plotly.express as px

    st.write("---")
    st.markdown("### 🔒 Business Intelligence Hotel ")
    st.caption("Aquest panell només és visible per a direcció a través d'un enllaç segur.")

    # Dashboard
    if os.path.exists("log_consultes.txt"):
        raw_logs = []
        with open("log_consultes.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(" | ")
                if len(parts) >= 4:
                    try:
                        data_str = parts[0].replace("[", "").replace("]", "").strip()
                        hab = parts[1].split(": ")[1].strip()
                        idioma = parts[2].split(": ")[1].strip()
                        cat = parts[3].split(": ")[1].strip()
                        preg = parts[4].split(": ", 1)[1].strip() if len(parts) > 4 else ""
                        
                        raw_logs.append({
                            "Data": data_str,
                            "Habitació": hab,
                            "Idioma": idioma,
                            "Categoria": cat,
                            "Pregunta": preg
                        })
                    except Exception:
                        pass 

        df = pd.DataFrame(raw_logs)

        if not df.empty:
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df['Hora'] = df['Data'].dt.hour

            # KPIs
            st.markdown("#### 📊 KPIs de Negoci")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            total_consultes = len(df)
            upsell_df = df[df["Categoria"].isin(["MENJAR", "SERVEIS"])]
            problemes_df = df[df["Categoria"] == "PROBLEMA"]
            
            taxa_upsell = (len(upsell_df) / total_consultes) * 100 if total_consultes > 0 else 0
            
            kpi1.metric("Interaccions Totals", total_consultes)
            kpi2.metric("Oportunitats Upselling", f"{len(upsell_df)}", f"{taxa_upsell:.1f}% del total")
            kpi3.metric("Alertes (Problemes)", len(problemes_df), "Risc", delta_color="inverse")
            kpi4.metric("Idiomes Detectats", df["Idioma"].nunique())

            st.write("---")

            # gràfics
            col_chart1, col_chart2 = st.columns([2, 1]) 
            
            with col_chart1:
                st.markdown("**🕒 Volum de Peticions per Franja Horària**")
                st.caption("Optimització de torns de Recepció i Room Service.")
                
                hora_counts = df['Hora'].value_counts().sort_index().reset_index()
                hora_counts.columns = ['Hora del dia', 'Nombre de Consultes']
                totes_les_hores = pd.DataFrame({'Hora del dia': range(24)})
                hora_counts = pd.merge(totes_les_hores, hora_counts, on='Hora del dia', how='left').fillna(0)
                
                fig_line = px.area(hora_counts, x='Hora del dia', y='Nombre de Consultes', 
                                   template="plotly_white", color_discrete_sequence=['#1f77b4'])
                fig_line.update_xaxes(tickmode='linear', dtick=2)
                st.plotly_chart(fig_line, use_container_width=True)

            with col_chart2:
                st.markdown("**Distribució de la Demanda**")
                st.caption("Focus d'interès dels clients.")
                
                fig_donut = px.pie(df, names="Categoria", hole=0.5, 
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                fig_donut.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_donut, use_container_width=True)

            # --- TAULA D'ALERTES EN DIRECTE ---
            if not problemes_df.empty:
                st.markdown("#### 🚨 Registre d'Incidències Crítiques")
                st.error("S'han detectat interaccions categoritzades com a PROBLEMA. Cal revisió.")
                st.dataframe(problemes_df[['Data', 'Habitació', 'Idioma', 'Pregunta']].sort_values(by='Data', ascending=False).head(5), use_container_width=True)
            else:
                st.success("✨ Cap queixa o incidència reportada recentment.")

        else:
            st.info("No hi ha dades suficients per generar els gràfics.")
    else:
        st.info("Encara no hi ha dades al registre. Fes una consulta per començar.")

    st.write("---")

    # --- 2. GESTIÓ DE FITXERS (DESCARREGAR I PUJAR BACKUPS) ---
    st.markdown("#### 📂 Gestió de Dades")
    col_down, col_up = st.columns(2)
    
    with col_down:
        st.markdown("**Exportar Històric**")
        if os.path.exists("log_consultes.txt"):
            with open("log_consultes.txt", "r", encoding="utf-8") as f:
                log_data = f.read()
            st.download_button(
                label="📥 Descarregar Log de Consultes (.txt)",
                data=log_data,
                file_name="log_consultes.txt",
                mime="text/plain"
            )
        else:
            st.write("No hi ha dades per descarregar.")

    with col_up:
        st.markdown("**Importar Dades (Backups)**")
        uploaded_file = st.file_uploader("📤 Pujar arxiu de logs (.txt)", type=["txt"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("⚠️ Sobreescriure dades actuals"):
                with open("log_consultes.txt", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("Base de dades actualitzada! Recarregant el panell...")
                time.sleep(1.5)
                st.rerun()
