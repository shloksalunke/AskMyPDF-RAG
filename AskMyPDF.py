import streamlit as st
import os, json, uuid
from dotenv import load_dotenv

# LangChain tools
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

# Load API key from .env file
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# UI setup
st.set_page_config(page_title="📄 AskMyPDF", layout="wide")
st.markdown("""
<h1 style='text-align: center; font-size: 3em;'>🧠 AskMyPDF</h1>
<h4 style='text-align: center; color: gray;'>Chat with your documents. <b>Locally</b>. <b>Privately</b>.</h4>
""", unsafe_allow_html=True)

with st.sidebar.expander("📘 What AskMyPDF Can Do", expanded=False):  
    st.markdown("""
    - 📄 Understand any type of PDF: academic, legal, medical, etc.
    - 💊 Read and answer medical reports like a smart assistant.
    - 🤖 Ask any kind of question – technical, logical, or general.
    - 🔐 Works 100% locally — your data stays private.
    - 🧠 Remembers your chat – supports follow-up questions.
    """)

st.sidebar.title("🗂️ Chat History")  



# Reset chat session if new
if "new_chat" in st.query_params:
    st.session_state.clear()

# Create folder to store chat history
os.makedirs("chats", exist_ok=True)

# Get saved chats from folder
def get_saved_chats():
    return sorted([f for f in os.listdir("chats") if f.endswith(".json")], reverse=True)

# Sidebar options
selected_chat_file = st.sidebar.selectbox("Previous Chats", ["New Chat"] + get_saved_chats())

# Start new chat
if st.sidebar.button("➕ New Chat"):
    st.session_state.clear()
    st.query_params["new_chat"] = str(uuid.uuid4())

# 🔰 Upload PDF on the main screen (only)
uploaded_pdf = st.file_uploader("📄 Upload a PDF to get started", type="pdf")

# Show welcome instructions if no file is uploaded
if not uploaded_pdf:
    st.markdown("""
    ## 👋 Welcome to AskMyPDF
    To begin, please **upload a PDF file** using the box above.  
    - 💡 No internet needed  
    - 🔐 100% Private & Local  
    - 🤖 Ask questions from your documents after upload!
    """)


# Initialize empty memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# Clean text from PDF chunks
@st.cache_data
def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

# Load previous chat if selected
if selected_chat_file != "New Chat" and os.path.exists(f"chats/{selected_chat_file}"):
    with open(f"chats/{selected_chat_file}", "r") as f:
        st.session_state.chat_memory = json.load(f)
    st.subheader(f"📁 Chat: {selected_chat_file}")
    for entry in st.session_state.chat_memory:
        st.chat_message("🧑").write(entry["user"])
        st.chat_message("🤖").write(entry["bot"])
    st.stop()

# Process uploaded PDF
if uploaded_pdf:
    temp_pdf_path = f"temp_{uuid.uuid4().hex}.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    try:
        # Load and split PDF
        pdf_loader = PyPDFLoader(temp_pdf_path)
        pdf_pages = pdf_loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        pdf_chunks = splitter.split_documents(pdf_pages)

        # Clean and format chunks
        cleaned_chunks = [Document(page_content=clean_text(chunk.page_content), metadata=chunk.metadata) for chunk in pdf_chunks]

        # Generate embeddings & store in FAISS
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        vector_db = FAISS.from_documents(cleaned_chunks, embeddings)

        # Memory to keep previous messages
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load LLM (Mistral)
        llm = ChatMistralAI(api_key=api_key, model="mistral-small-latest", temperature=0)

        # Create Retrieval QA chain
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(),
            memory=memory
        )

        # Remove uploaded file after processing
        os.remove(temp_pdf_path)

        st.success("✅ PDF Loaded! You can now ask questions.")

        # Chat input
        user_question = st.chat_input("Ask something about the PDF...")
        if user_question:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"question": user_question})
                bot_answer = response["answer"] if isinstance(response, dict) else response

                # Save to session and file
                st.session_state.chat_memory.append({"user": user_question, "bot": bot_answer})

                chat_file_name = st.session_state.get("current_file", f"chat_{uuid.uuid4().hex[:8]}.json")
                with open(f"chats/{chat_file_name}", "w") as f:
                    json.dump(st.session_state.chat_memory, f, indent=2)
                st.session_state.current_file = chat_file_name

                # Show on screen
                st.chat_message("🧑").write(user_question)
                st.chat_message("🤖").write(bot_answer)

    except Exception as e:
        st.error(f"❌ Error loading PDF: {str(e)}")

else:
    st.warning("📂 Please upload a PDF from the sidebar.")
