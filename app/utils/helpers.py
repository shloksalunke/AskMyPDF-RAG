# ============================================================
# 🛠️ app/utils/helpers.py — Environment, Health & Cleanup Helpers
# ============================================================

import os
import logging
from pathlib import Path

from app.config import MISTRAL_API_KEY, CHATS_DIR, TMP_DIR

logger = logging.getLogger(__name__)


def validate_environment() -> tuple[bool, str]:
    """Check that required environment variables are present and plausible."""
    if not MISTRAL_API_KEY:
        return False, "MISTRAL_API_KEY is not set — add it to your .env file"
    if len(MISTRAL_API_KEY) < 10:
        return False, "MISTRAL_API_KEY appears invalid (too short)"
    return True, "Environment validated ✅"


def check_system_health() -> dict:
    """Return a dict of basic runtime health indicators."""
    return {
        "api_key_configured":   bool(MISTRAL_API_KEY),
        "chats_dir_exists":     CHATS_DIR.exists(),
        "tmp_dir_exists":       TMP_DIR.exists(),
        "logs_writable":        _is_writable(Path("logs")),
        "stale_tmp_files":      _count_tmp_pdfs(),
    }


def cleanup_tmp_pdfs() -> int:
    """Delete all temporary PDF files from data/tmp/."""
    removed = 0
    try:
        for f in TMP_DIR.glob("*.pdf"):
            try:
                f.unlink()
                removed += 1
                logger.debug("Removed tmp PDF: %s", f)
            except Exception as exc:
                logger.error("Could not remove %s: %s", f, exc)
    except Exception as exc:
        logger.error("Error during tmp cleanup: %s", exc)
    if removed:
        logger.info("🧹 Removed %d stale tmp PDF(s)", removed)
    return removed


def get_tmp_pdf_path(user_id: str) -> Path:
    """Return the tmp path for a user's uploaded PDF."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    return TMP_DIR / f"upload_{user_id}.pdf"


# ── Private ───────────────────────────────────────────────────

def _is_writable(directory: Path) -> bool:
    probe = directory / ".write_test"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_text("x")
        probe.unlink()
        return True
    except Exception:
        return False


def _count_tmp_pdfs() -> int:
    try:
        return sum(1 for _ in TMP_DIR.glob("*.pdf"))
    except Exception:
        return 0
