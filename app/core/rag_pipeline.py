# ============================================================
# 🧠 app/core/rag_pipeline.py — RAG Pipeline (LangChain 0.3+ / LCEL)
# Embedding → Vector Store → Retrieval Chain → Answer
# ============================================================

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

from app.config import (
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    MISTRAL_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    LLM_TEMPERATURE,
    MAX_CHUNKS_PER_PDF,
)
from app.core.text_utils import TextProcessor

logger = logging.getLogger(__name__)


# ── Document Processor ────────────────────────────────────────
class DocumentProcessor:
    """Chunks and cleans LangChain Document objects."""

    @staticmethod
    def clean(documents: List[Document]) -> List[Document]:
        """Clean text content of each document."""
        for doc in documents:
            doc.page_content = TextProcessor.clean_text(doc.page_content)
        return documents

    @staticmethod
    def chunk(
        documents: List[Document],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> List[Document]:
        """Split documents into overlapping chunks."""
        logger.info(
            "Chunking %d doc(s) — size=%d, overlap=%d",
            len(documents), chunk_size, chunk_overlap,
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        if len(chunks) > MAX_CHUNKS_PER_PDF:
            logger.warning(
                "Capping chunks: %d -> %d", len(chunks), MAX_CHUNKS_PER_PDF
            )
            chunks = chunks[:MAX_CHUNKS_PER_PDF]

        logger.info("Created %d chunk(s)", len(chunks))
        return chunks


# ── Vector Store Manager ──────────────────────────────────────
class VectorStoreManager:
    """Creates and manages a FAISS vector store."""

    @staticmethod
    def build(documents: List[Document]) -> FAISS:
        """Embed documents and build an in-memory FAISS index."""
        logger.info("Building vector store for %d chunk(s)...", len(documents))
        embeddings = MistralAIEmbeddings(
            model=MISTRAL_EMBEDDING_MODEL,
            api_key=MISTRAL_API_KEY,
        )
        store = FAISS.from_documents(documents, embeddings)
        logger.info("Vector store ready")
        return store


# ── RAG Chain (LCEL — LangChain 0.3+) ────────────────────────
class RAGChain:
    """
    Modern LCEL-based conversational RAG chain.
    Replaces the deprecated ConversationalRetrievalChain.
    """

    def __init__(self, vector_store: FAISS):
        self._retriever = vector_store.as_retriever(
            search_kwargs={"k": RETRIEVAL_K}
        )
        self._llm = ChatMistralAI(
            api_key=MISTRAL_API_KEY,
            model=MISTRAL_MODEL,
            temperature=LLM_TEMPERATURE,
        )
        self._chat_history: list = []
        logger.info("RAG chain ready (model=%s)", MISTRAL_MODEL)

    def query(self, question: str, tone: str = "Friendly", style: str = "Concise") -> str:
        """
        Run a question through the RAG chain with conversation history.

        Args:
            question: User's natural-language question.
            tone:     Desired response tone.
            style:    Desired response style.

        Returns:
            LLM answer as a plain string.
        """
        # ── Step 1: Contextualise question using chat history ──────────
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given the chat history and the latest user question, "
             "rewrite the question to be fully self-contained. "
             "Do NOT answer the question. Return only the rewritten question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # If there is history, rephrase; otherwise use the question as-is
        if self._chat_history:
            contextualize_chain = contextualize_prompt | self._llm | StrOutputParser()
            standalone_question = contextualize_chain.invoke({
                "input": question,
                "chat_history": self._chat_history,
            })
        else:
            standalone_question = question

        # ── Step 2: Retrieve relevant chunks ──────────────────────────
        docs = self._retriever.invoke(standalone_question)
        context = "\n\n".join(d.page_content for d in docs)

        # ── Step 3: Generate answer ────────────────────────────────────
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"You are AskMyPDF, a helpful document assistant.\n"
             f"Respond in a {tone} tone with a {style} style.\n"
             f"Answer ONLY from the context below. "
             f"If the answer is not in the context, say so honestly.\n\n"
             f"Context:\n{{context}}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        answer_chain = qa_prompt | self._llm | StrOutputParser()
        raw_answer = answer_chain.invoke({
            "input": question,
            "chat_history": self._chat_history,
            "context": context,
        })

        # ── Step 4: Update history ─────────────────────────────────────
        self._chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=raw_answer),
        ])
        # Keep last 10 exchanges (20 messages) to avoid token bloat
        if len(self._chat_history) > 20:
            self._chat_history = self._chat_history[-20:]

        return TextProcessor.clean_output(raw_answer)


# ── End-to-End RAG Pipeline ───────────────────────────────────
class RAGPipeline:
    """
    High-level pipeline: PDF Documents -> chunks -> FAISS -> LLM -> answers.

    Usage::

        pipeline = RAGPipeline()
        if pipeline.setup(documents):
            answer = pipeline.query("What is this document about?")
    """

    def __init__(self):
        self._chain: RAGChain | None = None
        self.num_chunks: int = 0

    def setup(self, documents: List[Document]) -> bool:
        """
        Initialise the pipeline with extracted PDF documents.

        Args:
            documents: List of LangChain Document objects (one per page).

        Returns:
            True on success, False on failure.
        """
        try:
            logger.info("Setting up RAG pipeline...")
            cleaned = DocumentProcessor.clean(documents)
            chunks  = DocumentProcessor.chunk(cleaned)
            store   = VectorStoreManager.build(chunks)
            self._chain     = RAGChain(store)
            self.num_chunks = len(chunks)
            logger.info("RAG pipeline ready (%d chunks)", self.num_chunks)
            return True
        except Exception as exc:
            logger.error("RAG pipeline setup failed: %s", exc)
            return False

    def query(self, question: str, tone: str = "Friendly", style: str = "Concise") -> str:
        """Ask a question. Pipeline must be set up first."""
        if not self._chain:
            raise RuntimeError("RAG pipeline not initialised — call setup() first.")
        return self._chain.query(question, tone=tone, style=style)

    @property
    def is_ready(self) -> bool:
        return self._chain is not None
