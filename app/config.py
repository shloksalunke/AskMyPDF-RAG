# ============================================================
# ⚙️ app/config.py — Centralized Configuration
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level up from app/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# ── Paths ────────────────────────────────────────────────────
DATA_DIR       = _ROOT / "data"
CHATS_DIR      = DATA_DIR / "chats"
TMP_DIR        = DATA_DIR / "tmp"
LOGS_DIR       = _ROOT / "logs"
LOG_FILE       = str(LOGS_DIR / "app.log")

# Ensure runtime directories exist
for _d in (CHATS_DIR, TMP_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── API Configuration ────────────────────────────────────────
MISTRAL_API_KEY        = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL          = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_EMBEDDING_MODEL = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")

# ── Security ─────────────────────────────────────────────────
SECRET_KEY               = os.getenv("SECRET_KEY", "change-me-in-production")
SESSION_TIMEOUT_MINUTES  = int(os.getenv("SESSION_TIMEOUT_MINUTES", "120"))

# ── Storage ──────────────────────────────────────────────────
CHAT_EXPIRATION_DAYS               = int(os.getenv("CHAT_EXPIRATION_DAYS", "3"))
TEMP_PDF_CLEANUP_INTERVAL_MINUTES  = int(os.getenv("TEMP_PDF_CLEANUP_INTERVAL_MINUTES", "60"))

# ── RAG Pipeline ─────────────────────────────────────────────
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K         = int(os.getenv("RETRIEVAL_K", "5"))
LLM_TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_CHUNKS_PER_PDF  = int(os.getenv("MAX_CHUNKS_PER_PDF", "500"))

# ── OCR ──────────────────────────────────────────────────────
ENABLE_OCR                  = os.getenv("ENABLE_OCR", "true").lower() == "true"
OCR_DPI                     = int(os.getenv("OCR_DPI", "300"))
OCR_CONFIDENCE_THRESHOLD    = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.6"))
IMAGE_PDF_CHAR_THRESHOLD    = int(os.getenv("IMAGE_PDF_CHAR_THRESHOLD", "15"))

# ── UI ───────────────────────────────────────────────────────
APP_TITLE  = os.getenv("APP_TITLE", "📄 AskMyPDF — OCR RAG Chatbot")
APP_LAYOUT = os.getenv("APP_LAYOUT", "wide")

# ── Background Tasks ─────────────────────────────────────────
ENABLE_BACKGROUND_CLEANUP  = os.getenv("ENABLE_BACKGROUND_CLEANUP", "true").lower() == "true"
CLEANUP_INTERVAL_HOURS     = int(os.getenv("CLEANUP_INTERVAL_HOURS", "6"))

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Validation ───────────────────────────────────────────────
def validate_config() -> tuple[bool, str]:
    """Validate required configuration values."""
    if not MISTRAL_API_KEY:
        return False, "MISTRAL_API_KEY is not set in .env"
    if len(MISTRAL_API_KEY) < 10:
        return False, "MISTRAL_API_KEY appears invalid (too short)"
    return True, "Configuration is valid"
