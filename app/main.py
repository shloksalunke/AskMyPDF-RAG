# ============================================================
# 📄 app/main.py — AskMyPDF Streamlit Application
# Run with: streamlit run app/main.py
# ============================================================

# Bootstrap: ensure project root is on sys.path so `app.*` imports work
# regardless of how Streamlit invokes this file.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import logging
import uuid
from datetime import datetime

import streamlit as st

from app.config import (
    APP_LAYOUT,
    APP_TITLE,
    CHAT_EXPIRATION_DAYS,
    ENABLE_BACKGROUND_CLEANUP,
    LOG_FILE,
    LOG_LEVEL,
)
from app.core.ocr_processor import get_ocr_processor
from app.core.rag_pipeline import RAGPipeline
from app.core.text_utils import TextProcessor
from app.utils.chat_manager import ChatDataManager, ChatHistoryManager
from app.utils.helpers import (
    check_system_health,
    cleanup_tmp_pdfs,
    get_tmp_pdf_path,
    validate_environment,
)
from app.utils.security import UserSecurityManager

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Environment guard ─────────────────────────────────────────
_env_ok, _env_msg = validate_environment()
if not _env_ok:
    # Must call set_page_config before any other st call
    st.set_page_config(page_title="AskMyPDF — Config Error", layout="centered")
    st.error(f"❌ Configuration error: {_env_msg}")
    st.info("Add your `MISTRAL_API_KEY` to the `.env` file and restart.")
    st.stop()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AskMyPDF",
    page_icon="🧠",
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/shloksalunke/AskMyPDF-RAG",
        "Report a bug": "https://github.com/shloksalunke/AskMyPDF-RAG/issues",
        "About": "AskMyPDF — OCR-based RAG Chatbot with MistralAI",
    },
)

# ── Global styles ─────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── Typography ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── Header ── */
        .main-header { text-align: center; padding: 1.5rem 0 0.5rem; }
        .main-header h1 { font-size: 2.4rem; font-weight: 700; margin-bottom: 0.25rem; }
        .main-header p  { color: #6b7280; font-size: 0.95rem; }

        /* ── Status badges ── */
        .badge-success {
            background: #d1fae5; border: 1px solid #6ee7b7;
            color: #065f46; padding: 0.6rem 1rem;
            border-radius: 0.5rem; margin-bottom: 0.75rem;
        }
        .badge-info {
            background: #dbeafe; border: 1px solid #93c5fd;
            color: #1e40af; padding: 0.6rem 1rem;
            border-radius: 0.5rem; margin-bottom: 0.75rem;
        }
        .badge-warning {
            background: #fef3c7; border: 1px solid #fcd34d;
            color: #92400e; padding: 0.6rem 1rem;
            border-radius: 0.5rem; margin-bottom: 0.75rem;
        }

        /* ── Upload zone ── */
        [data-testid="stFileUploader"] {
            border: 2px dashed #6366f1 !important;
            border-radius: 0.75rem !important;
            padding: 1rem !important;
        }

        /* ── Chat bubbles ── */
        [data-testid="stChatMessage"] { border-radius: 0.75rem; margin-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state bootstrap ───────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    logger.info("New session: %s", st.session_state.user_id)

_defaults = {
    "security_manager": lambda: UserSecurityManager(),
    "chat_manager":     lambda: ChatDataManager(expiration_days=CHAT_EXPIRATION_DAYS),
    "history_manager":  lambda: ChatHistoryManager(),
    "chat_memory":      list,
    "vector_db":        lambda: None,
    "pdf_loaded":       lambda: False,
    "rag_pipeline":     lambda: None,
    "pdf_filename":     lambda: None,
    "current_chat":     lambda: None,
    "cleanup_run":      lambda: False,
}
for key, factory in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = factory()

# ── Header ───────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🧠 AskMyPDF</h1>
        <p>OCR-powered RAG Chatbot &nbsp;•&nbsp; 100% Private &nbsp;•&nbsp;
           Chat history auto-deletes after 3 days</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    if st.button("➕ New Chat", use_container_width=True):
        for k in ("chat_memory", "vector_db", "rag_pipeline",
                  "pdf_loaded", "pdf_filename", "current_chat"):
            st.session_state[k] = [] if k == "chat_memory" else None
        st.session_state.pdf_loaded = False
        st.rerun()

    st.divider()

    # Chat history
    if st.session_state.pdf_loaded:
        st.markdown("### 📂 Chat History")
        user_chats = st.session_state.history_manager.list_chats(st.session_state.user_id)

        if user_chats:
            selected = st.selectbox(
                "Select chat", user_chats, label_visibility="collapsed"
            )
            if selected and selected != st.session_state.current_chat:
                loaded = st.session_state.history_manager.load(
                    st.session_state.user_id, selected
                )
                if loaded:
                    st.session_state.chat_memory = loaded
                    st.session_state.current_chat = selected
                    st.rerun()

            if st.button("🗑️ Delete This Chat", use_container_width=True):
                if st.session_state.current_chat:
                    st.session_state.history_manager.delete(
                        st.session_state.user_id, st.session_state.current_chat
                    )
                    st.session_state.chat_memory = []
                    st.session_state.current_chat = None
                    st.success("Chat deleted")
                    st.rerun()
        else:
            st.info("No saved chats yet")

    st.divider()

    # Download chat
    if st.session_state.chat_memory:
        st.download_button(
            "💾 Download Chat (JSON)",
            data=json.dumps(st.session_state.chat_memory, indent=2),
            file_name=f"askmypdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
        st.divider()

    # How-to expander
    with st.expander("ℹ️ How to Use"):
        st.markdown(
            """
**Supported PDFs:**
- ✅ Text-based PDFs (reports, ebooks, docs)
- ✅ Scanned / image PDFs (auto-OCR)
- ❌ Password-protected PDFs

**Steps:**
1. Upload a PDF below
2. Wait for processing (OCR if needed)
3. Ask questions in the chat box
4. Download your chat anytime
            """
        )

    with st.expander("🔒 Privacy"):
        st.markdown(
            """
- 🔐 Each session has an isolated UUID
- 💾 Data stored locally only
- 🗑️ Auto-deleted after 3 days
- 🚫 Nothing sent to external servers (except Mistral API calls)
            """
        )

    if st.checkbox("🏥 System Status"):
        st.json(check_system_health())

# ── PDF Upload & Processing ───────────────────────────────────
st.markdown("### 📄 Upload Your PDF")

if not st.session_state.pdf_loaded:
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type="pdf",
        key="pdf_uploader",
        help="Supports text-based and scanned PDFs",
    )

    if uploaded:
        tmp_path = get_tmp_pdf_path(st.session_state.user_id)
        tmp_path.write_bytes(uploaded.read())

        ocr = get_ocr_processor(enable_ocr=True)
        is_valid, msg = ocr.validate_pdf(str(tmp_path))

        st.markdown(
            f'<div class="badge-info">📋 {msg}</div>', unsafe_allow_html=True
        )

        if is_valid:
            with st.spinner("⏳ Extracting text (OCR if needed)…"):
                try:
                    documents = ocr.extract_from_pdf(str(tmp_path))

                    if documents:
                        pipeline = RAGPipeline()
                        if pipeline.setup(documents):
                            st.session_state.rag_pipeline = pipeline
                            st.session_state.pdf_loaded   = True
                            st.session_state.pdf_filename = uploaded.name
                            st.markdown(
                                f'<div class="badge-success">'
                                f'✅ Ready! Extracted {len(documents)} page(s) '
                                f'({pipeline.num_chunks} chunks indexed)'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            logger.info("PDF loaded: %s", uploaded.name)
                        else:
                            st.error("❌ Failed to build RAG index")
                    else:
                        st.error("❌ No text could be extracted from the PDF")

                except Exception as exc:
                    st.error(f"❌ Processing error: {exc}")
                    logger.exception("PDF processing error")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
        else:
            st.error(f"❌ {msg}")
            if tmp_path.exists():
                tmp_path.unlink()
else:
    st.success(f"✅ Loaded: **{st.session_state.pdf_filename}**")
    if st.button("🔄 Load a Different PDF", use_container_width=True):
        st.session_state.pdf_loaded   = False
        st.session_state.rag_pipeline = None
        st.session_state.pdf_filename = None
        st.rerun()

# ── Chat Interface ────────────────────────────────────────────
if st.session_state.pdf_loaded and st.session_state.rag_pipeline:
    st.markdown("### 💬 Chat with Your PDF")

    # Render history
    for msg in st.session_state.chat_memory:
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            st.write(msg["bot"])

    # Input
    user_question = st.chat_input("Ask anything about your PDF…")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("🤖 Thinking…"):
            try:
                tone, style = TextProcessor.detect_tone_and_style(user_question)
                answer = st.session_state.rag_pipeline.query(
                    user_question, tone=tone, style=style
                )

                with st.chat_message("assistant"):
                    st.write(answer)

                # Persist
                st.session_state.chat_memory.append(
                    {
                        "user":      user_question,
                        "bot":       answer,
                        "tone":      tone,
                        "style":     style,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                chat_file = (
                    st.session_state.current_chat
                    or f"chat_{uuid.uuid4().hex[:8]}.json"
                )
                st.session_state.history_manager.save(
                    st.session_state.user_id,
                    chat_file,
                    st.session_state.chat_memory,
                )
                st.session_state.current_chat = chat_file

            except Exception as exc:
                st.error(f"❌ Error: {exc}")
                logger.exception("Query error")

# ── Background Cleanup ────────────────────────────────────────
if ENABLE_BACKGROUND_CLEANUP and not st.session_state.cleanup_run:
    try:
        stats = st.session_state.chat_manager.cleanup_expired()
        st.session_state.chat_manager.cleanup_empty_dirs()
        tmp_removed = cleanup_tmp_pdfs()

        if stats["deleted"] or tmp_removed:
            logger.info(
                "🧹 Cleanup: %d expired chat(s), %d tmp PDF(s) removed",
                stats["deleted"], tmp_removed,
            )
        st.session_state.cleanup_run = True
    except Exception as exc:
        logger.error("Cleanup error: %s", exc)

# ── Footer ────────────────────────────────────────────────────
st.divider()
c1, c2, c3 = st.columns(3)
c1.caption(f"👤 Session: `{st.session_state.user_id[:8]}…`")
c2.caption(f"💬 Messages: {len(st.session_state.chat_memory)}")
c3.caption("🔒 100% Private")
