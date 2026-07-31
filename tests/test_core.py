# ============================================================
# 🧪 tests/test_core.py — Unit & Integration Tests
# Run with: python -m pytest tests/ -v
# ============================================================

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

# Bootstrap environment before importing app modules
os.environ.setdefault("MISTRAL_API_KEY", "test-key-1234567890")

from app.config import validate_config
from app.core.text_utils import TextProcessor
from app.utils.chat_manager import ChatDataManager, ChatHistoryManager
from app.utils.helpers import validate_environment
from app.utils.security import UserSecurityManager


# ── Configuration ─────────────────────────────────────────────
class TestConfig(unittest.TestCase):

    def test_validate_config_passes_with_key(self):
        ok, msg = validate_config()
        self.assertTrue(ok, msg)

    def test_validate_environment_passes_with_key(self):
        ok, msg = validate_environment()
        self.assertTrue(ok, msg)


# ── Text Processor ────────────────────────────────────────────
class TestTextProcessor(unittest.TestCase):

    def test_clean_text_strips_null_bytes(self):
        result = TextProcessor.clean_text("Hello\x00World")
        self.assertNotIn("\x00", result)

    def test_clean_text_handles_non_string(self):
        self.assertEqual(TextProcessor.clean_text(None), "")
        self.assertEqual(TextProcessor.clean_text(123), "")

    def test_clean_output_collapses_whitespace(self):
        result = TextProcessor.clean_output("Hello   \n\n\n   World")
        self.assertNotIn("\n\n\n", result)

    def test_detect_tone_academic(self):
        tone, _ = TextProcessor.detect_tone_and_style("Explain this academically")
        self.assertEqual(tone, "Academic")

    def test_detect_tone_casual(self):
        tone, _ = TextProcessor.detect_tone_and_style("quick bro")
        self.assertEqual(tone, "Casual")

    def test_detect_style_bulleted(self):
        _, style = TextProcessor.detect_tone_and_style("Give me a list of steps")
        self.assertEqual(style, "Bulleted")

    def test_detect_style_detailed(self):
        _, style = TextProcessor.detect_tone_and_style("Explain in depth")
        self.assertEqual(style, "Detailed")

    def test_detect_style_concise(self):
        _, style = TextProcessor.detect_tone_and_style("Give a brief summary")
        self.assertEqual(style, "Concise")


# ── Security Manager ─────────────────────────────────────────
class TestSecurity(unittest.TestCase):

    def setUp(self):
        self.mgr = UserSecurityManager(secret_key="test-secret")
        self.uid = "user_abc123"

    def test_session_id_starts_with_user_id(self):
        sid = self.mgr.generate_session_id(self.uid)
        self.assertTrue(sid.startswith(self.uid))

    def test_validate_valid_session(self):
        sid = self.mgr.generate_session_id(self.uid)
        self.assertTrue(self.mgr.validate_session_id(sid, self.uid))

    def test_validate_wrong_user(self):
        sid = self.mgr.generate_session_id(self.uid)
        self.assertFalse(self.mgr.validate_session_id(sid, "other_user"))

    def test_get_user_chat_dir_creates_folder(self):
        folder = self.mgr.get_user_chat_dir(self.uid)
        self.assertTrue(folder.exists())
        # Cleanup
        shutil.rmtree(folder, ignore_errors=True)


# ── Chat History Manager ──────────────────────────────────────
class TestChatHistoryManager(unittest.TestCase):

    def setUp(self):
        self.mgr = ChatHistoryManager()
        self.uid = "test_hist_user"
        self.fname = "session_001.json"

    def tearDown(self):
        from app.config import CHATS_DIR
        shutil.rmtree(CHATS_DIR / self.uid, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        messages = [{"user": "Hi", "bot": "Hello!"}]
        self.mgr.save(self.uid, self.fname, messages)
        loaded = self.mgr.load(self.uid, self.fname)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["user"], "Hi")

    def test_list_chats_returns_filename(self):
        self.mgr.save(self.uid, self.fname, [{"user": "x", "bot": "y"}])
        chats = self.mgr.list_chats(self.uid)
        self.assertIn(self.fname, chats)

    def test_delete_chat(self):
        self.mgr.save(self.uid, self.fname, [{"user": "x", "bot": "y"}])
        result = self.mgr.delete(self.uid, self.fname)
        self.assertTrue(result)
        self.assertEqual(self.mgr.list_chats(self.uid), [])

    def test_load_missing_returns_empty(self):
        result = self.mgr.load(self.uid, "nonexistent.json")
        self.assertEqual(result, [])


# ── Chat Data Manager ─────────────────────────────────────────
class TestChatDataManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_is_expired_for_old_file(self):
        mgr = ChatDataManager(expiration_days=1)
        f = Path(self.tmp) / "old.json"
        f.write_text("{}")
        old_time = (datetime.utcnow() - timedelta(days=2)).timestamp()
        os.utime(f, (old_time, old_time))
        self.assertTrue(mgr.is_expired(f))

    def test_is_not_expired_for_new_file(self):
        mgr = ChatDataManager(expiration_days=3)
        f = Path(self.tmp) / "new.json"
        f.write_text("{}")
        self.assertFalse(mgr.is_expired(f))


# ── Integration ───────────────────────────────────────────────
class TestIntegration(unittest.TestCase):

    def test_full_chat_lifecycle(self):
        from app.config import CHATS_DIR
        uid   = "integration_user"
        fname = "integration_chat.json"
        mgr   = ChatHistoryManager()
        sec   = UserSecurityManager()

        try:
            folder = sec.get_user_chat_dir(uid)
            self.assertTrue(folder.exists())

            data = [{"user": "What is this?", "bot": "It is a test."}]
            mgr.save(uid, fname, data)

            loaded = mgr.load(uid, fname)
            self.assertEqual(len(loaded), 1)

            chats = mgr.list_chats(uid)
            self.assertIn(fname, chats)

            mgr.delete(uid, fname)
            self.assertEqual(mgr.list_chats(uid), [])
        finally:
            shutil.rmtree(CHATS_DIR / uid, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
