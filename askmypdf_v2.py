# ============================================================
# 🧠 AskMyPDF v3 — Enterprise-grade RAG Chatbot with OCR (PaddleOCR)
# Author: Shlok Salunke
# ✅ FINAL FIX — LangChain & PaddleX import patch (stable for Streamlit Cloud)
# ============================================================

import os, sys, types, importlib.util, json, uuid, re
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
import cv2

# ============================================================
# 🧩 UNIVERSAL RUNTIME PATCH (fixes all LangChain import errors)
# ============================================================
try:
    # Safe import detection
    lcd = None
    lts = None
    lcm = None

    if importlib.util.find_spec("langchain_core.documents"):
        import langchain_core.documents as lcd
    if importlib.util.find_spec("langchain_text_splitters"):
        import langchain_text_splitters as lts
    if importlib.util.find_spec("langchain_core.memory"):
        import langchain_core.memory as lcm

    # Patch: langchain.docstore.document
    if lcd:
        module_docstore = types.ModuleType("langchain.docstore.document")
        module_docstore.Document = lcd.Document
        sys.modules["langchain.docstore.document"] = module_docstore

    # Patch: langchain.text_splitter
    if lts:
        sys.modules["langchain.text_splitter"] = lts

    # Patch: langchain.memory
    if lcm:
        sys.modules["langchain.memory"] = lcm

    print("✅ LangChain compatibility patch applied successfully!")

except Exception as e:
    print(f"⚠️ Patch setup encountered issue: {e}")

# ============================================================
# 🧩 PaddleOCR (import only after patching!)
# ============================================================
from paddleocr import PaddleOCR

# ============================================================
# 🧠 LangChain & Mistral components
# ============================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# ============================================================
# 🔑 Load Environment Variables
# ============================================================
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# ============================================================
# ⚙️ Streamlit Page Config
# ============================================================
st.set_page_config(page_title="📄 AskMyPDF v3", layout="wide")
st.markdown("""
<h1 style='text-align:center;'>🧠 AskMyPDF v3</h1>
<h4 style='text-align:center;color:gray;'>Enterprise-grade RAG chatbot with OCR (PaddleOCR), auto tone detection, and smart prompt templates.</h4>
""", unsafe_allow_html=True)

# ============================================================
# 🧹 Helper Functions
# ============================================================
def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

def clean_output(text):
    if isinstance(text, list):
        text = " ".join([doc.page_content for doc in text])
    text = re.sub(r'[\n`*•#\-\d]+\.*\s*', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_image_pdf(pdf_path, threshold_chars=15):
    """Check if PDF is likely scanned (no extractable text)."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs or all(len(p.page_content.strip()) < threshold_chars for p in docs):
            return True
        return False
    except Exception:
        return True

# ============================================================
# 🧾 OCR Extraction using PaddleOCR
# ============================================================
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

def ocr_extract(pdf_path, dpi=300):
    """OCR extraction for scanned/image PDFs using PaddleOCR."""
    pages = convert_from_path(pdf_path, dpi=dpi)
    docs = []
    for i, page in enumerate(pages):
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = ocr_engine.ocr(img, cls=True)
        text_lines = []
        for res in result:
            for box, (text, conf) in res:
                if conf > 0.6:
                    text_lines.append(text)
        page_text = "\n".join(text_lines)
        metadata = {"page": i + 1, "source": pdf_path}
        docs.append(Document(page_content=page_text, metadata=metadata))
    return docs

# ============================================================
# 🧩 Load and Index PDF (OCR + Embedding)
# ============================================================
def load_and_index(pdf_path, chunk_size=1000, chunk_overlap=150):
    """Load, clean, split, and embed PDF (with OCR fallback)."""
    if is_image_pdf(pdf_path):
        st.info("🔍 Detected scanned PDF → applying OCR. This may take a while...")
        pages = ocr_extract(pdf_path)
    else:
        pages = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)
    cleaned = [Document(page_content=clean_text(c.page_content), metadata=c.metadata) for c in chunks]

    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    vector_db = FAISS.from_documents(cleaned, embeddings)
    return vector_db

# ============================================================
# 🧠 Auto Tone & Style Detection
# ============================================================
def detect_tone_and_style(query):
    query_lower = query.lower()
    academic_keywords = ["academic", "research", "paper", "theory", "elaborate", "explain in detail"]
    formal_keywords = ["explain", "discuss", "describe", "analyze", "define"]
    casual_keywords = ["pls", "please", "yaar", "buddy", "bro", "short", "quick", "simple"]
    friendly_keywords = ["hi", "hello", "thanks", "cool", "awesome"]

    concise_keywords = ["short", "brief", "summary", "summarize"]
    detailed_keywords = ["detailed", "explain fully", "in depth", "complete"]
    bullet_keywords = ["points", "bullets", "list"]

    tone = "Friendly"
    style = "Concise"

    if any(k in query_lower for k in academic_keywords):
        tone, style = "Academic", "Detailed"
    elif any(k in query_lower for k in formal_keywords):
        tone = "Formal"
    elif any(k in query_lower for k in casual_keywords):
        tone = "Casual"
    elif any(k in query_lower for k in friendly_keywords):
        tone = "Friendly"

    if any(k in query_lower for k in detailed_keywords):
        style = "Detailed"
    elif any(k in query_lower for k in bullet_keywords):
        style = "Bulleted"
    elif any(k in query_lower for k in concise_keywords):
        style = "Concise"

    return tone, style

# ============================================================
# 💬 System Prompt (Industry-Grade)
# ============================================================
SYSTEM_PROMPT = """
You are **AskMyPDF**, an intelligent Retrieval-Augmented Generation (RAG) assistant designed to understand, analyze, and summarize information from PDF documents provided by the user.

Your key roles:
1. **Context Awareness:** Always use the retrieved document chunks as your main source of truth. Avoid making up facts.
2. **Clarity & Brevity:** Respond in clear, human-like natural language. Avoid overly technical jargon unless requested.
3. **Tone Adaptation:** Match the user's tone automatically (formal, friendly, academic, or concise).
4. **Transparency:** If the answer is missing, say “The document doesn’t contain explicit information about that.”
5. **Citation:** Include page numbers or metadata like `(Page 5)` if available.
6. **Focus:** Only answer questions related to the uploaded PDF unless asked otherwise.
7. **Formatting:** Use bullet points, numbering, or short paragraphs for readability.
8. **Privacy:** Never store, reveal, or infer personal data outside the session.
"""

RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "tone", "style"],
    template="""
{system_prompt}

User Tone Detected: {tone}
Preferred Response Style: {style}

Context Extracted from PDF:
---------------------------
{context}

User Question:
--------------
{question}

Assistant Task:
---------------
- Understand the question intent.
- Use only the context to answer.
- Maintain the {tone} tone and {style} style.
- If missing info, state that politely.
- Ensure readability and structure.

Final Answer:
"""
)

# ============================================================
# 🔗 Build RAG Chain
# ============================================================
def build_chain(vector_db, tone="Friendly", style="Concise", temperature=0.2):
    llm = ChatMistralAI(api_key=api_key, model="mistral-small-latest", temperature=temperature)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT_TEMPLATE.partial(system_prompt=SYSTEM_PROMPT)}
    )
    return chain

# ============================================================
# 💾 Streamlit Session State
# ============================================================
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

st.session_state.setdefault("chat_memory", [])
st.session_state.setdefault("vector_db", None)
st.session_state.setdefault("rag_chain", None)
st.session_state.setdefault("pdf_loaded", False)
st.session_state.setdefault("current_file", None)

user_folder = os.path.join("chats", st.session_state.user_id)
os.makedirs(user_folder, exist_ok=True)

# ============================================================
# 📤 PDF Upload Section
# ============================================================
if not st.session_state.pdf_loaded:
    uploaded_pdf = st.file_uploader("📄 Upload a PDF", type="pdf")
    if uploaded_pdf:
        temp_path = f"temp_{st.session_state.user_id}.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.read())

        with st.spinner("⚙️ Processing your PDF..."):
            try:
                st.session_state.vector_db = load_and_index(temp_path, chunk_size=1500, chunk_overlap=200)
                st.session_state.pdf_loaded = True
                st.success("✅ PDF processed successfully! You can now start chatting.")
            except Exception as e:
                st.error(f"❌ Error loading PDF: {e}")
        os.remove(temp_path)
    else:
        st.info("📎 Please upload a PDF file to begin.")
        st.stop()

# ============================================================
# 💬 Chat Interface
# ============================================================
for entry in st.session_state.chat_memory:
    st.chat_message("🧑").write(entry["user"])
    st.chat_message("🤖").write(entry["bot"])

user_question = st.chat_input("Ask something about your PDF...")
if user_question:
    with st.spinner("🤖 Thinking..."):
        try:
            tone, style = detect_tone_and_style(user_question)
            st.info(f"🧠 Detected Tone: **{tone}** | Style: **{style}**")

            basic_inputs = {"hi", "hello", "hey", "who are you", "help"}
            if user_question.lower().strip() in basic_inputs:
                bot_answer = "👋 Hey! I’m AskMyPDF v3 — your intelligent RAG assistant that adapts to your tone and analyzes your PDFs."
            else:
                st.session_state.rag_chain = build_chain(st.session_state.vector_db, tone=tone, style=style)
                response = st.session_state.rag_chain.invoke({
                    "question": user_question,
                    "tone": tone,
                    "style": style
                })
                bot_answer = response["answer"] if isinstance(response, dict) else response
                bot_answer = clean_output(bot_answer)

            st.chat_message("🧑").write(user_question)
            st.chat_message("🤖").write(bot_answer)

            st.session_state.chat_memory.append({
                "user": user_question, "bot": bot_answer, "tone": tone, "style": style
            })

            chat_file = st.session_state.current_file or f"chat_{uuid.uuid4().hex[:8]}.json"
            with open(os.path.join(user_folder, chat_file), "w") as f:
                json.dump(st.session_state.chat_memory, f, indent=2)
            st.session_state.current_file = chat_file
        except Exception as e:
            st.error(f"⚠️ Failed to generate response: {e}")
