import os
import time
import datetime
import warnings
import logging
import streamlit as st
from google import genai
from google.genai.types import GenerateContentConfig
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ─────────────────────────────────────────────
# CLEAN LOGS
# ─────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

CHROMA_DIR = ".chroma_store"
RETRIEVAL_K = 6

# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────
def detect_language(text: str) -> str:
    text = text.lower()

    if any("\u4e00" <= c <= "\u9fff" for c in text):
        return "zh"
    if any("\u3040" <= c <= "\u30ff" for c in text):
        return "ja"
    if any("\uac00" <= c <= "\ud7af" for c in text):
        return "ko"
    if any(w in text for w in ["què", "servei", "habitació"]):
        return "ca"
    if any(w in text for w in ["qué", "servicio", "habitación"]):
        return "es"

    return "en"

# ─────────────────────────────────────────────
# TRANSLATE TO ENGLISH (FOR RETRIEVAL)
# ─────────────────────────────────────────────
def translate_to_english(ai_client, text: str) -> str:
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Translate to English (only translation): {text}",
            config={"temperature": 0}
        )
        return response.text.strip()
    except:
        return text

# ─────────────────────────────────────────────
# SYSTEM PROMPT (STRICT LANGUAGE CONTROL)
# ─────────────────────────────────────────────
SYSTEM_INSTRUCTION = """\
You are Lumi, virtual concierge of Hotel Ducado.

CRITICAL RULE:
You MUST ALWAYS respond in the exact same language as the guest message.

- Catalan → Catalan
- Spanish → Spanish
- English → English
- Chinese → Chinese
- ANY language → same language

The hotel context may be in English or Spanish.
IGNORE its language. Use it only as knowledge.

If answer is not found → apologize briefly and suggest reception.

Style:
Elegant, warm, concise. Max 3 paragraphs.
"""

USER_TEMPLATE = """\
LANGUAGE: {lang}

HOTEL CONTEXT:
{context}

GUEST MESSAGE:
{question}
"""

# ─────────────────────────────────────────────
# LOAD EMBEDDINGS
# ─────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"},
    )

# ─────────────────────────────────────────────
# LOAD VECTOR DB
# ─────────────────────────────────────────────
@st.cache_resource
def load_db(embeddings):
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    documents = []
    for file in os.listdir("documentos_hotel"):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join("documentos_hotel", file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    db.persist()
    return db

# ─────────────────────────────────────────────
# RETRIEVAL (MULTI QUERY)
# ─────────────────────────────────────────────
def retrieve_context(db, question, ai_client):
    queries = [question]

    en = translate_to_english(ai_client, question)
    if en != question:
        queries.append(en)

    all_chunks = []
    for q in queries:
        results = db.similarity_search(q, k=RETRIEVAL_K)
        all_chunks.extend(results)

    # deduplicate
    seen = set()
    final = []
    for doc in all_chunks:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            final.append(doc)

    return final[:6]

# ─────────────────────────────────────────────
# FORMAT CONTEXT
# ─────────────────────────────────────────────
def format_context(chunks):
    return "\n\n---\n\n".join([c.page_content for c in chunks])

# ─────────────────────────────────────────────
# CALL GEMINI
# ─────────────────────────────────────────────
def call_gemini(client, prompt, placeholder):
    config = GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)

    full = ""
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config,
    )

    for chunk in response:
        if chunk.text:
            full += chunk.text
            placeholder.markdown(full + "▌")

    placeholder.markdown(full)
    return full

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Lumi", page_icon="✨")

embeddings = load_embeddings()
db = load_db(embeddings)
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if pregunta := st.chat_input("Ask something..."):

    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    lang = detect_language(pregunta)

    chunks = retrieve_context(db, pregunta, client)

    prompt = USER_TEMPLATE.format(
        context=format_context(chunks),
        question=pregunta,
        lang=lang
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        respuesta = call_gemini(client, prompt, placeholder)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})
