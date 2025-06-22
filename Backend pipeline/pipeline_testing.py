from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
import os, re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

#loaded the api key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
#choosing llm
llm = ChatMistralAI(
    api_key=api_key,
    model="mistral-small-latest",
    temperature=0,
)

#give file path to load it in doc variable
file_path = "path of pdf file "
loader = PyPDFLoader(file_path)
doc = loader.load()
#spit the documents in 1000 chunks and overlaping 150 chunks recursively
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks_of_doc = splitter.split_documents(doc)
print(f"Total chunks created: {len(chunks_of_doc)}")
#ignoring the utf-8 toget clean text.
def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

cleaned_chunks = [
    Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
    for doc in chunks_of_doc
]
#convert text in numerical format by embedding it 
embeddings = MistralAIEmbeddings(model="mistral-embed")
# storing the embedded data in vector database
vectorstore = FAISS.from_documents(cleaned_chunks, embeddings)
#Applying similarity search
response = vectorstore.similarity_search(question, k=3)
#This is for getting clean output
def clean_output(text):
    if isinstance(text, list):
        text = " ".join([doc.page_content for doc in text])
    text = re.sub(r'[\n`*•#\-\d]+\.*\s*', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

#memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#Here I have made the chain of whole RAG
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

#geting the output
final_answer = chain.run(question)


while True:
    question = input("🗣️ You: ")
    if question.lower() in ["exit", "quit", "bye"]:
        print("👋 Bot: Goodbye!")
        break
    final_answer = chain.run(question)
    print("\n🧠 Bot:", clean_output(final_answer), "\n")
