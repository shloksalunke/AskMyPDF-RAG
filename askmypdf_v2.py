# ============================================================
# 🧠 AskMyPDF v3 — Enterprise-grade RAG Chatbot with OCR (PaddleOCR)
# Author: Shlok Salunke
# ✅ FINAL READY-TO-DEPLOY — Python 3.11 + LangChain 1.x Compatible
# ============================================================

import os, sys, types, importlib.util, json, uuid, re
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
import cv2

# ============================================================
# 🧩 UNIVERSAL RUNTIME PATCH (fixes all LangChain import issues)
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
    print(f"⚠️ Patch setup encountered issue: {e}")

# ============================================================
# 🧩 PaddleOCR (import after patching)
# ============================================================
from paddleocr import PaddleOCR

# ============================================================
# 🧠 LangChain & Mistral components — Stable Imports
# ============================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# ✅ Universal Memory Handler (LangChain ≥ 1.0)
# ============================================================
try:
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

    class ConversationBufferMemory:
        """Fully compatible memory class for LangChain ≥ 1.0"""
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

except Exception as e:
    print(f"⚠️ Using fallback memory handler: {e}")
    from langchain_core.messages import HumanMessage, AIMessage
    class ConversationBufferMemory:
        def __init__(self, memory_key="chat_history", return_messages=True):
            self.memory_key = memory_key
            self.return_messages = return_messages
            self.messages = []

        def load_memory_variables(self, inputs):
            return {self.memory_key: self.messages}

        def save_context(self, inputs, outputs):
            q = inputs.get("question") or ""
            a = outputs.get("answer") or ""
            if q:
                self.messages.append(HumanMessage(content=q))
            if a:
                self.messages.append(AIMessage(content=a))

        def clear(self):
            self.messages = []

# ============================================================
# 🔗 Chain, Document, and PromptTemplate imports
# ============================================================
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
<h4 style='text-align:center;color:gray;'>Enterprise-grade RAG chatbot with OCR (PaddleOCR), tone detection, and smart prompts.</h4>
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
# 🧾 OCR Extraction using PaddleOCR
# ============================================================
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

def ocr_extract(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    docs = []
    for i, page in enumerate(pages):
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = ocr_engine.ocr(img, cls=True)
        text_lines = [t for r in result for box, (t, conf) in r if conf > 0.6]
        docs.append(Document(page_content="\n".join(text_lines),
                             metadata={"page": i+1, "source": pdf_path}))
    return docs

# ============================================================
# 🧩 Load and Index PDF (OCR + Embedding)
# ============================================================
def load_and_index(pdf_path, chunk_size=1000, chunk_overlap=150):
    if is_image_pdf(pdf_path):
        st.info("🔍 Detected scanned PDF → applying OCR. This may take a while...")
        pages = ocr_extract(pdf_path)
    else:
        pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)
    cleaned = [Document(page_content=clean_text(c.page_content), metadata=c.metadata) for c in chunks]
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    return FAISS.from_documents(cleaned, embeddings)

# ============================================================
# 🧠 Tone & Style Detection
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
SYSTEM_PROMPT = """You are AskMyPDF, a smart RAG assistant analyzing PDFs.
Use retrieved chunks as truth. Keep answers clear, structured, and tone-matched.
If missing info, say: “The document doesn’t contain explicit information about that.”"""

RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "tone", "style"],
    template="""
{system_prompt}

Tone: {tone}
Style: {style}

Context:
{context}

Question:
{question}

Answer (tone & style maintained):
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
        llm=llm, retriever=retriever, memory=memory,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT_TEMPLATE.partial(system_prompt=SYSTEM_PROMPT)}
    )

# ============================================================
# 💾 Streamlit Session State
# ============================================================
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
st.session_state.setdefault("chat_memory", [])
st.session_state.setdefault("vector_db", None)
st.session_state.setdefault("pdf_loaded", False)

user_folder = os.path.join("chats", st.session_state.user_id)
os.makedirs(user_folder, exist_ok=True)

# ============================================================
# 📤 PDF Upload
# ============================================================
if not st.session_state.pdf_loaded:
    up = st.file_uploader("📄 Upload a PDF", type="pdf")
    if up:
        tmp = f"temp_{st.session_state.user_id}.pdf"
        with open(tmp, "wb") as f: f.write(up.read())
        with st.spinner("⚙️ Processing your PDF..."):
            try:
                st.session_state.vector_db = load_and_index(tmp, 1500, 200)
                st.session_state.pdf_loaded = True
                st.success("✅ PDF processed successfully!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        os.remove(tmp)
    else:
        st.info("📎 Please upload a PDF to begin.")
        st.stop()

# ============================================================
# 💬 Chat Interface
# ============================================================
for m in st.session_state.chat_memory:
    st.chat_message("🧑").write(m["user"])
    st.chat_message("🤖").write(m["bot"])

q = st.chat_input("Ask something about your PDF...")
if q:
    with st.spinner("🤖 Thinking..."):
        try:
            tone, style = detect_tone_and_style(q)
            st.info(f"🧠 Tone: **{tone}** | Style: **{style}**")
            base = {"hi","hello","hey","who are you","help"}
            if q.lower().strip() in base:
                a = "👋 Hey! I’m AskMyPDF v3 — your PDF-smart RAG assistant."
            else:
                chain = build_chain(st.session_state.vector_db, tone, style)
                resp = chain.invoke({"question": q, "tone": tone, "style": style})
                a = resp.get("answer", str(resp))
                a = clean_output(a)
            st.chat_message("🧑").write(q)
            st.chat_message("🤖").write(a)
            st.session_state.chat_memory.append({"user": q, "bot": a})
            with open(os.path.join(user_folder, "chat.json"), "w") as f:
                json.dump(st.session_state.chat_memory, f, indent=2)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
