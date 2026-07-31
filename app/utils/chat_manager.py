# ============================================================
# 💾 app/utils/chat_manager.py — Chat Persistence & Lifecycle
# ============================================================

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

from app.config import CHATS_DIR, CHAT_EXPIRATION_DAYS

logger = logging.getLogger(__name__)


# ── Chat History Manager ─────────────────────────────────────
class ChatHistoryManager:
    """Persists and retrieves per-user chat sessions as JSON files."""

    def save(self, user_id: str, filename: str, messages: List[dict]) -> bool:
        """Write a chat session to ``data/chats/<user_id>/<filename>``."""
        try:
            folder = CHATS_DIR / user_id
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / filename

            payload = {
                "user_id":    user_id,
                "created_at": datetime.utcnow().isoformat(),
                "messages":   messages,
            }
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.debug("Chat saved → %s", path)
            return True
        except Exception as exc:
            logger.error("Error saving chat: %s", exc)
            return False

    def load(self, user_id: str, filename: str) -> List[dict]:
        """Load messages from a saved chat file."""
        try:
            path = CHATS_DIR / user_id / filename
            if not path.is_file():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("messages", [])
        except Exception as exc:
            logger.error("Error loading chat: %s", exc)
            return []

    def list_chats(self, user_id: str) -> List[str]:
        """Return chat filenames for a user, newest first."""
        try:
            folder = CHATS_DIR / user_id
            if not folder.exists():
                return []
            files = sorted(
                (f for f in folder.iterdir() if f.suffix == ".json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            return [f.name for f in files]
        except Exception as exc:
            logger.error("Error listing chats: %s", exc)
            return []

    def delete(self, user_id: str, filename: str) -> bool:
        """Delete a single chat file."""
        try:
            path = CHATS_DIR / user_id / filename
            if path.is_file():
                path.unlink()
                logger.info("Chat deleted: %s", path)
                return True
            return False
        except Exception as exc:
            logger.error("Error deleting chat: %s", exc)
            return False


# ── Chat Data Manager (expiration & cleanup) ──────────────────
class ChatDataManager:
    """Handles expiration and storage-level cleanup of chat data."""

    def __init__(self, expiration_days: int = CHAT_EXPIRATION_DAYS):
        self._expiry_seconds = expiration_days * 86_400

    def is_expired(self, path: Path) -> bool:
        """Return True if the file is older than the expiry window."""
        try:
            age = datetime.utcnow().timestamp() - path.stat().st_mtime
            return age > self._expiry_seconds
        except Exception:
            return False

    def cleanup_expired(self, verbose: bool = False) -> dict:
        """Delete expired chat files across all users."""
        stats = {"deleted": 0, "errors": 0}

        if not CHATS_DIR.exists():
            return stats

        for user_dir in CHATS_DIR.iterdir():
            if not user_dir.is_dir():
                continue
            for chat_file in list(user_dir.iterdir()):
                if self.is_expired(chat_file):
                    try:
                        chat_file.unlink()
                        stats["deleted"] += 1
                        if verbose:
                            logger.info("Deleted expired chat: %s", chat_file)
                    except Exception as exc:
                        stats["errors"] += 1
                        logger.error("Error deleting %s: %s", chat_file, exc)

        return stats

    def cleanup_empty_dirs(self) -> int:
        """Remove empty user directories under data/chats/."""
        removed = 0
        if not CHATS_DIR.exists():
            return 0
        for user_dir in CHATS_DIR.iterdir():
            if user_dir.is_dir() and not any(user_dir.iterdir()):
                try:
                    user_dir.rmdir()
                    removed += 1
                except Exception as exc:
                    logger.error("Error removing dir %s: %s", user_dir, exc)
        return removed

    def storage_stats(self) -> dict:
        """Return high-level statistics about chat storage."""
        stats = {"total_users": 0, "total_chats": 0, "expired_chats": 0, "size_mb": 0.0}
        if not CHATS_DIR.exists():
            return stats

        total_bytes = 0
        for user_dir in CHATS_DIR.iterdir():
            if not user_dir.is_dir():
                continue
            stats["total_users"] += 1
            for chat_file in user_dir.iterdir():
                stats["total_chats"] += 1
                total_bytes += chat_file.stat().st_size
                if self.is_expired(chat_file):
                    stats["expired_chats"] += 1

        stats["size_mb"] = round(total_bytes / 1_048_576, 2)
        return stats
