# ============================================================
# 🔐 app/utils/security.py — User Isolation & Session Security
# ============================================================

import os
import hashlib
import hmac
import logging
from datetime import datetime
from pathlib import Path

from app.config import SECRET_KEY, CHATS_DIR

logger = logging.getLogger(__name__)


class UserSecurityManager:
    """
    Provides per-user isolation and basic session validation.

    Each user is assigned a UUID at session start. Their chat data
    lives in an isolated sub-directory under ``data/chats/<user_id>/``.
    """

    def __init__(self, secret_key: str | None = None):
        self._key = (secret_key or SECRET_KEY).encode()

    # ── Session ID ───────────────────────────────────────────

    def generate_session_id(self, user_id: str) -> str:
        """Create a signed session identifier for the given user."""
        timestamp = datetime.utcnow().isoformat()
        payload   = f"{user_id}:{timestamp}"
        sig = hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()
        return f"{user_id}:{sig[:16]}"

    def validate_session_id(self, session_id: str, user_id: str) -> bool:
        """Verify that *session_id* was issued for *user_id*."""
        try:
            return session_id.split(":")[0] == user_id
        except Exception:
            return False

    # ── Chat Folder ──────────────────────────────────────────

    def get_user_chat_dir(self, user_id: str) -> Path:
        """Return (and create if missing) the isolated chat directory for a user."""
        folder = CHATS_DIR / user_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def is_owner(self, user_id: str, chat_filename: str) -> bool:
        """Return True if the chat file exists under the user's directory."""
        path = CHATS_DIR / user_id / chat_filename
        return path.is_file()
