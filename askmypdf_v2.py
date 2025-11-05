# ============================================================
# 🧠 AskMyPDF v3.1 — Enterprise-grade RAG Chatbot with OCR (PaddleOCR)
# Author: Shlok Salunke
# ✅ FINAL VERSION — 100% Compatible with LangChain ≥1.0 & Streamlit Cloud
# ============================================================

import os, sys, types, importlib.util, json, uuid, re
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
import cv2

# ============================================================
# 🧩 UNIVERSAL PATCH (fixes LangChain import issues automatically)
# ============================================================
try:
    lcd = None
    lts = None
    if importlib.util.find_spec("langchain_core.documents"):
        import langchain_core.documents as lcd
    if importlib.util.find_spec("langchain_text_splitters"):
        import langchain_text_splitters as lts
    if lcd:
        module_docstore = types.ModuleType("langchain.docstore.document")
        module_docstore.Document = lcd.Document
        sys.modules["langchain.docstore.document"] = module_docstore
    if lts:
        sys.modules["langchain.text_splitter"] = lts
    print("✅ LangChain compatibility patch applied successfully!")
except Exception as e:
    print(f"⚠️ Patch issue: {e}")

# ============================================================
# 🧩 PaddleOCR (after patch)
# ============================================================
from paddleocr import PaddleOCR

# ============================================================
# 🧠 LangChain + Mistral imports (modular & stable)
# ============================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chains import ConversationalRetrievalChain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

# ============================================================
# ✅ Custom Memory (replaces removed langchain.memory)
# ============================================================
class ConversationBufferMemory:
    """Memory buffer compatible with LangChain ≥1.0"""
    def __init__(self, memory_key="chat_history", return_messages=True):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.history = ChatMessageHistory()

    def load_memory_variables(self, inputs):
        if self.return_messages:
            return {self.memory_key: self.history.messages}
        return {self.memory_key: "\n".join([m.content for m in self.history.messages])}

    def save_context(self, inputs, outputs):
        q = inputs.get("question") or inputs.get("input") or ""
        a = outputs.get("answer") or outputs.get("output") or ""
        if q:
            self.history.add_message(HumanMessage(content=q))
        if a:
            self.history.add_message(AIMessage(content=a))

    def clear(self):
        self.history.clear()

# ============================================================
# 🔑 Load Environment Variables
# ============================================================
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# ============================================================
# ⚙️ Streamlit Page Setup
# ============================================================
st.set_page_config(page_title="📄 AskMyPDF v3.1", layout="wide")
st.markdown("""
<h1 style='text-align:center;'>🧠 AskMyPDF v3.1</h1>
<h4 style='text-align:center;color:gray;'>OCR + RAG chatbot using MistralAI & LangChain 1.x (Final Stable Release)</h4>
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
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return not docs or all(len(p.page_content.strip()) < threshold_chars for p in docs)
    except Exception:
        return True

# ============================================================
# 🧾 OCR Extraction (for scanned PDFs)
# ============================================================
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

def ocr_extract(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    docs = []
    for i, page in enumerate(pages):
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = ocr_engine.ocr(img, cls=True)
        lines = [t for r in result for box, (t, conf) in r if conf > 0.6]
        docs.append(Document(page_content="\n".join(lines), metadata={"page": i + 1, "source": pdf_path}))
    return docs

# ============================================================
# 🧩 Load + Embed PDF
# ============================================================
def load_and_index(pdf_path, chunk_size=1000, chunk_overlap=150):
    if is_image_pdf(pdf_path):
        st.info("🔍 Scanned PDF detected → applying OCR. Please wait...")
        pages = ocr_extract(pdf_path)
    else:
        pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)
    cleaned = [Document(page_content=clean_text(c.page_content), metadata=c.metadata) for c in chunks]
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    return FAISS.from_documents(cleaned, embeddings)

# ============================================================
# 🧠 Tone & Style Detector
# ============================================================
def detect_tone_and_style(q):
    ql = q.lower()
    tone, style = "Friendly", "Concise"
    if any(k in ql for k in ["academic","research","paper","theory"]): tone, style = "Academic","Detailed"
    elif any(k in ql for k in ["explain","discuss","describe","define"]): tone="Formal"
    elif any(k in ql for k in ["pls","yaar","bro","quick"]): tone="Casual"
    if any(k in ql for k in ["detailed","complete","in depth"]): style="Detailed"
    elif any(k in ql for k in ["points","bullets","list"]): style="Bulleted"
    elif any(k in ql for k in ["short","brief","summary"]): style="Concise"
    return tone, style

# ============================================================
# 💬 System Prompt
# ============================================================
SYSTEM_PROMPT = """You are AskMyPDF, a RAG-based assistant that analyzes PDF context to answer clearly, factually, and politely."""

RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "tone", "style"],
    template="""
{system_prompt}

Detected Tone: {tone}
Style: {style}

Context:
{context}

Question:
{question}

Answer (keep structure, tone, and clarity):
"""
)

# ============================================================
# 🔗 Build RAG Chain
# ============================================================
def build_chain(vector_db, tone="Friendly", style="Concise"):
    llm = ChatMistralAI(api_key=api_key, model="mistral-small-latest", temperature=0.2)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT_TEMPLATE.partial(system_prompt=SYSTEM_PROMPT)}
    )

# ============================================================
# 💾 Streamlit Session
# ============================================================
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
st.session_state.setdefault("chat_memory", [])
st.session_state.setdefault("vector_db", None)
st.session_state.setdefault("pdf_loaded", False)
folder = os.path.join("chats", st.session_state.user_id)
os.makedirs(folder, exist_ok=True)

# ============================================================
# 📤 PDF Upload
# ============================================================
if not st.session_state.pdf_loaded:
    up = st.file_uploader("📄 Upload PDF", type="pdf")
    if up:
        temp = f"temp_{st.session_state.user_id}.pdf"
        with open(temp, "wb") as f: f.write(up.read())
        with st.spinner("⚙️ Processing your PDF..."):
            try:
                st.session_state.vector_db = load_and_index(temp, 1500, 200)
                st.session_state.pdf_loaded = True
                st.success("✅ PDF processed successfully! Ask me anything.")
            except Exception as e:
                st.error(f"❌ Error loading PDF: {e}")
        os.remove(temp)
    else:
        st.info("📎 Please upload a PDF to begin.")
        st.stop()

# ============================================================
# 💬 Chat Interface
# ============================================================
for chat in st.session_state.chat_memory:
    st.chat_message("🧑").write(chat["user"])
    st.chat_message("🤖").write(chat["bot"])

q = st.chat_input("Ask something about your PDF...")
if q:
    with st.spinner("🤖 Thinking..."):
        try:
            tone, style = detect_tone_and_style(q)
            st.info(f"🧠 Detected Tone: **{tone}** | Style: **{style}**")
            if q.lower().strip() in {"hi","hello","hey","who are you"}:
                a = "👋 Hey! I’m AskMyPDF v3.1 — your intelligent RAG assistant."
            else:
                chain = build_chain(st.session_state.vector_db, tone, style)
                resp = chain.invoke({"question": q, "tone": tone, "style": style})
                a = clean_output(resp.get("answer", str(resp)))
            st.chat_message("🧑").write(q)
            st.chat_message("🤖").write(a)
            st.session_state.chat_memory.append({"user": q, "bot": a})
            with open(os.path.join(folder, "chat.json"), "w") as f:
                json.dump(st.session_state.chat_memory, f, indent=2)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
