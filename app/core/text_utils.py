# ============================================================
# 📝 app/core/text_utils.py — Text Processing Utilities
# ============================================================

import re
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text cleaning, normalisation, and tone detection."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Strip null bytes, control characters, and normalise whitespace."""
        if not isinstance(text, str):
            return ""
        # Remove null bytes and other ASCII control characters (except \t, \n, \r)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Drop remaining non-UTF-8 bytes
        text = text.encode("utf-8", "ignore").decode("utf-8")
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def clean_output(text: str) -> str:
        """Clean LLM output for readable display."""
        if isinstance(text, list):
            text = " ".join(
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in text
            )
        # Keep markdown structure intact — only collapse excessive whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def detect_tone_and_style(question: str) -> tuple[str, str]:
        """Infer the user's preferred tone and response style from the question."""
        q = question.lower()
        tone = "Friendly"
        style = "Concise"

        # Tone
        if any(k in q for k in ("academic", "research", "paper", "theory", "scholarly")):
            tone = "Academic"
        elif any(k in q for k in ("explain", "discuss", "describe", "define", "elaborate")):
            tone = "Formal"
        elif any(k in q for k in ("pls", "yaar", "bro", "quick", "asap")):
            tone = "Casual"

        # Style
        if any(k in q for k in ("detailed", "complete", "in depth", "in-depth", "thoroughly")):
            style = "Detailed"
        elif any(k in q for k in ("points", "bullets", "list", "steps", "enumerate")):
            style = "Bulleted"
        elif any(k in q for k in ("short", "brief", "summary", "summarise", "summarize", "quick")):
            style = "Concise"

        return tone, style
