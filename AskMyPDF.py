import streamlit as st
import os, json, uuid
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

# Load API key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Page setup
st.set_page_config(page_title="📄 AskMyPDF", layout="wide")
st.markdown("""
<h1 style='text-align: center; font-size: 3em;'>🧠 AskMyPDF</h1>
<h4 style='text-align: center; color: gray;'>Chat with your PDF. 100% Private. Local Only.</h4>
""", unsafe_allow_html=True)

# Session ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_folder = os.path.join("chats", st.session_state.user_id)
os.makedirs(user_folder, exist_ok=True)

# Sidebar
st.sidebar.markdown("### ☰ Menu")

# New Chat Button
if st.sidebar.button("➕ Start New Chat"):
    st.session_state.chat_memory = []
    st.session_state.current_file = None
    st.session_state.vector_db = None
    st.session_state.rag_chain = None
    st.session_state.pdf_loaded = False
    st.rerun()

# Chat list per user
def get_saved_chats():
    if not os.path.exists(user_folder):
        return []
    return sorted([f for f in os.listdir(user_folder) if f.endswith(".json")], reverse=True)

selected_chat = st.sidebar.selectbox("🗂️ Previous Chats", ["New Chat"] + get_saved_chats())

# Load chat if exists
if selected_chat != "New Chat":
    path = os.path.join(user_folder, selected_chat)
    if os.path.exists(path):
        with open(path, "r") as f:
            st.session_state.chat_memory = json.load(f)
        st.session_state.current_file = selected_chat

# Chat download
if st.session_state.get("chat_memory"):
    chat_json = json.dumps(st.session_state.chat_memory, indent=2)
    st.sidebar.download_button("💾 Download This Chat", chat_json, file_name="askmypdf_chat.json", mime="application/json")

# Init vars (💡 include current_file)
st.session_state.setdefault("chat_memory", [])
st.session_state.setdefault("vector_db", None)
st.session_state.setdefault("rag_chain", None)
st.session_state.setdefault("pdf_loaded", False)
st.session_state.setdefault("current_file", None)

@st.cache_data
def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

# Upload PDF
if not st.session_state.pdf_loaded:
    uploaded_pdf = st.file_uploader("📄 Upload a PDF to get started", type="pdf")

    if uploaded_pdf:
        with st.spinner("⏳ Processing PDF..."):
            temp_path = f"temp_{st.session_state.user_id}.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_pdf.read())

            try:
                pages = PyPDFLoader(temp_path).load()
                if not pages:
                    st.error("❌ PDF has no readable content.")
                    st.stop()

                chunks = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(pages)[:200]
                cleaned = [Document(page_content=clean_text(c.page_content), metadata=c.metadata) for c in chunks]

                embeddings = MistralAIEmbeddings(model="mistral-embed")
                st.session_state.vector_db = FAISS.from_documents(cleaned, embeddings)

                llm = ChatMistralAI(api_key=api_key, model="mistral-small-latest", temperature=0)
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
                st.session_state.rag_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm, retriever=retriever, memory=memory
                )

                os.remove(temp_path)
                st.session_state.pdf_loaded = True
                st.success("✅ PDF Loaded! Ask your first question below.")
            except Exception as e:
                st.error(f"❌ Failed to load PDF: {str(e)}")
                st.stop()
    else:
        st.markdown("""
        ## 👋 Welcome to AskMyPDF  
        Upload a PDF above to start chatting  
        - 🔐 100% Private  
        - 🧠 Fully local & secure  
        """)
        st.stop()

# Show chat history
for entry in st.session_state.chat_memory:
    st.chat_message("🧑").write(entry["user"])
    st.chat_message("🤖").write(entry["bot"])

# Chat
user_question = st.chat_input("Ask something about your PDF...")
if user_question:
    if not st.session_state.rag_chain:
        st.error("❗ Please upload a PDF first.")
        st.stop()

    with st.spinner("🤖 Thinking..."):
        try:
            response = st.session_state.rag_chain.invoke({"question": user_question})
            bot_answer = response["answer"] if isinstance(response, dict) else response

            st.chat_message("🧑").write(user_question)
            st.chat_message("🤖").write(bot_answer)

            st.session_state.chat_memory.append({"user": user_question, "bot": bot_answer})

            # 💾 Save chat
            chat_file = st.session_state.current_file or f"chat_{uuid.uuid4().hex[:8]}.json"
            with open(os.path.join(user_folder, chat_file), "w") as f:
                json.dump(st.session_state.chat_memory, f, indent=2)
            st.session_state.current_file = chat_file

        except Exception as e:
            st.error(f"⚠️ Failed to respond: {e}")
