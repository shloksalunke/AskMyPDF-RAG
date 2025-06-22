# 🧠 AskMyPDF

**Chat with your documents. Locally. Privately.**

AskMyPDF is a fully local, private RAG (Retrieval-Augmented Generation) chatbot that lets you upload any PDF file and interact with it in natural language. Whether it's a textbook, medical paper, or legal document — AskMyPDF can answer your queries intelligently, without ever sending your data to the cloud.

---

## 🚀 Features

✅ Upload any PDF and ask questions directly from it  
✅ Smart answers using MistralAI + LangChain's RAG pipeline  
✅ Fully **local + private** (no document goes online)  
✅ Remembers previous questions using LangChain Memory  
✅ Saves & displays **past chat history**  
✅ Built-in file handling and temp cleanup  
✅ Simple, clean UI powered by Streamlit

---

## 🛠️ Tech Stack

| Component       | Library / Tool             |
|----------------|----------------------------|
| 💬 LLM          | MistralAI (`mistral-small-latest`)  
| 🔎 Embedding    | `mistral-embed` via MistralAI  
| 🧠 RAG Chain    | LangChain `ConversationalRetrievalChain`  
| 📄 PDF Loader   | LangChain `PyPDFLoader`  
| 📚 Vector Store | FAISS (Local, In-Memory)  
| 💻 UI           | Streamlit  
| 🧠 Memory       | LangChain `ConversationBufferMemory`  

---

## 🔄 How It Works (Behind the Scenes)

1. 📤 **Upload PDF** using sidebar  
2. 📖 PDF is loaded and split into chunks  
3. 🧠 Each chunk is embedded using `mistral-embed`  
4. 🧲 FAISS index is built on the embedded chunks  
5. ❓ User asks a question  
6. 🔍 FAISS retrieves the most relevant chunks  
7. 🤖 Mistral LLM answers the question using context  
8. 💾 Chat is saved and memory is updated

---

## 👨‍💻 What I Built Myself

I developed the **entire backend RAG pipeline manually**, including:

- PDF chunking, embedding, and FAISS setup  
- LangChain-based ConversationalRetrievalChain  
- MistralAI integration and token handling  
- Session memory, file cleanup, and chat history logic

👉 I used ChatGPT **only** for frontend help to structure the Streamlit interface.  
The core logic, backend structure, testing, and debugging was done completely by me — from scratch.

---

## 📦 Installation

1. **Clone the repo**
```bash
git clone https://github.com/shloksalunke/AskMyPDF-RAG.git
cd AskMyPDF-RAG

pip install -r requirements.txt

MISTRAL_API_KEY=your_mistral_api_key

streamlit run AskMyPDF.py

📁 File Structure
bash
Copy code
AskMyPDF-RAG/
├── AskMyPDF.py               # Streamlit app (final UI)
├── pipeline_testing.py       # Backend-only pipeline test
├── requirements.txt          # All Python dependencies
├── .env                      
├── chats/                    
├── README.md                 
🔐 Privacy First
AskMyPDF runs fully offline. Your PDF is never uploaded online.
All processing is done locally, and all your chat history stays in your machine.

📄 License
This project is licensed under the MIT License.

📬 Feedback
Have ideas or suggestions?
Reach me on LinkedIn(www.linkedin.com/in/shlok-salunke-4947b429b) or drop a star ⭐ on GitHub!


