# ============================================================
# 🧾 app/core/ocr_processor.py — OCR & PDF Text Extraction
# OCR Engine: Tesseract (via pytesseract)
# ============================================================

import logging
from pathlib import Path
from typing import List

from PIL import Image
from pdf2image import convert_from_path

# ── Poppler auto-detection (Windows) ─────────────────────────
def _find_poppler_path() -> str | None:
    """Find the Poppler bin directory on Windows at common install locations."""
    import platform, os
    if platform.system() != "Windows":
        return None  # On Linux/Mac, poppler is on PATH via package manager

    candidates = [
        r"C:\Program Files\poppler\Library\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\poppler\Library\bin",
        r"C:\poppler\bin",
        # Also check versioned extracts like C:\poppler-24.08.0\...
        *[
            os.path.join(r"C:\\", d, "Library", "bin")
            for d in os.listdir("C:\\")
            if d.lower().startswith("poppler")
        ],
    ]
    for path in candidates:
        if os.path.isdir(path) and any(
            f.startswith("pdftoppm") for f in os.listdir(path)
        ):
            return path
    return None

_POPPLER_PATH = _find_poppler_path()
if _POPPLER_PATH:
    logging.getLogger(__name__).info("Poppler found at: %s", _POPPLER_PATH)
else:
    logging.getLogger(__name__).warning(
        "Poppler not found in common locations. "
        "Download from https://github.com/oschwartz10612/poppler-windows/releases "
        "and extract to C:\\poppler\\"
    )

try:
    import pytesseract
    TESSERACT_AVAILABLE = True

    # Auto-detect Tesseract binary on Windows if not on PATH
    import os, platform
    if platform.system() == "Windows":
        _default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(_default_path):
            pytesseract.pytesseract.tesseract_cmd = _default_path

except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning(
        "pytesseract not installed — scanned PDFs will not be processed. "
        "Install with: pip install pytesseract"
    )

from langchain_core.documents import Document

from app.config import OCR_DPI, IMAGE_PDF_CHAR_THRESHOLD
from app.core.text_utils import TextProcessor

logger = logging.getLogger(__name__)


# ── OCR Processor ─────────────────────────────────────────────
class OCRProcessor:
    """
    Intelligently extracts text from PDFs.

    - Text-based PDFs  → PyPDF (fast, lossless)
    - Scanned/image PDFs → Tesseract OCR
    """

    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        self._ocr_available = enable_ocr and TESSERACT_AVAILABLE

        if enable_ocr and not TESSERACT_AVAILABLE:
            logger.warning(
                "Tesseract not available. Install: pip install pytesseract "
                "AND the Tesseract binary from https://github.com/UB-Mannheim/tesseract/wiki"
            )
        elif self._ocr_available:
            # Verify tesseract binary is accessible
            try:
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR engine ready: v%s", pytesseract.get_tesseract_version())
            except Exception as exc:
                logger.warning(
                    "Tesseract binary not found: %s\n"
                    "Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "Then add it to PATH or set pytesseract.pytesseract.tesseract_cmd",
                    exc,
                )
                self._ocr_available = False

    # ── Public API ───────────────────────────────────────────

    @property
    def ocr_ready(self) -> bool:
        return self._ocr_available

    def extract_from_pdf(
        self, pdf_path: str, force_ocr: bool = False
    ) -> List[Document]:
        """
        Auto-detect PDF type and extract text.

        Args:
            pdf_path:  Absolute path to PDF file.
            force_ocr: If True, always use OCR regardless of PDF type.

        Returns:
            List of LangChain Document objects (one per page).
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if force_ocr:
            logger.info("Force-OCR enabled")
            return self._extract_with_ocr(pdf_path)

        if self._is_text_pdf(pdf_path):
            logger.info("Text-based PDF detected — using PyPDF")
            return self._extract_text_pdf(pdf_path)

        logger.info("Scanned PDF detected — applying Tesseract OCR")
        if not self._ocr_available:
            logger.warning("OCR engine unavailable; falling back to PyPDF")
            return self._extract_text_pdf(pdf_path)

        return self._extract_with_ocr(pdf_path)

    def validate_pdf(self, pdf_path: str) -> tuple[bool, str]:
        """Check whether the PDF is readable. Returns (is_valid, message)."""
        path = Path(pdf_path)
        if not path.exists():
            return False, "PDF file not found"
        if path.suffix.lower() != ".pdf":
            return False, "File is not a PDF"

        try:
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(pdf_path).load()
            if not docs:
                return False, "PDF contains no pages"

            total_chars = sum(len(d.page_content.strip()) for d in docs)
            if total_chars < 10:
                if self.enable_ocr and not self._ocr_available:
                    return (
                        False,
                        "PDF appears to be image-only but Tesseract OCR is not available. "
                        "See README for installation instructions.",
                    )
                return True, f"Scanned PDF detected — {len(docs)} page(s) will be processed with Tesseract OCR"

            return True, f"Valid PDF — {len(docs)} page(s), ~{total_chars:,} characters"
        except Exception as exc:
            return False, f"Error validating PDF: {exc}"

    # ── Private Helpers ──────────────────────────────────────

    def _is_text_pdf(self, pdf_path: str) -> bool:
        """Return True if the PDF has extractable text above the threshold."""
        try:
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(pdf_path).load()
            if not docs:
                return False
            total = sum(len(d.page_content.strip()) for d in docs)
            return total >= IMAGE_PDF_CHAR_THRESHOLD
        except Exception as exc:
            logger.warning("Could not check PDF type: %s — assuming text-based", exc)
            return True

    def _extract_text_pdf(self, pdf_path: str) -> List[Document]:
        """Extract text from a text-based PDF using PyPDF."""
        try:
            from langchain_community.document_loaders import PyPDFLoader

            logger.info("Loading PDF with PyPDF: %s", pdf_path)
            docs = PyPDFLoader(pdf_path).load()

            for i, doc in enumerate(docs):
                doc.page_content = TextProcessor.clean_text(doc.page_content)
                doc.metadata.setdefault("page", i + 1)
                doc.metadata["extracted_by"] = "PyPDF"

            logger.info("PyPDF extracted %d page(s)", len(docs))
            return docs
        except Exception as exc:
            logger.error("PyPDF extraction failed: %s", exc)
            raise RuntimeError(f"PDF extraction failed: {exc}") from exc

    def _extract_with_ocr(self, pdf_path: str) -> List[Document]:
        """Extract text from a scanned PDF using Tesseract OCR."""
        if not self._ocr_available:
            raise RuntimeError(
                "Tesseract OCR not available. "
                "Install: pip install pytesseract "
                "AND the Tesseract binary from https://github.com/UB-Mannheim/tesseract/wiki"
            )

        try:
            logger.info("Tesseract OCR processing %s at %d DPI...", pdf_path, OCR_DPI)
            pages = convert_from_path(
                pdf_path,
                dpi=OCR_DPI,
                poppler_path=_POPPLER_PATH,  # None on Linux/Mac (uses PATH)
            )
            docs: List[Document] = []

            for i, page_img in enumerate(pages):
                logger.info("  OCR page %d / %d", i + 1, len(pages))

                # Run Tesseract on the PIL Image directly
                page_text = pytesseract.image_to_string(
                    page_img,
                    lang="eng",
                    config="--psm 3",  # Fully automatic page segmentation
                )

                docs.append(
                    Document(
                        page_content=TextProcessor.clean_text(
                            page_text or "[No text detected on this page]"
                        ),
                        metadata={
                            "page": i + 1,
                            "source": pdf_path,
                            "extracted_by": "Tesseract",
                        },
                    )
                )

            logger.info("Tesseract OCR complete — %d page(s) extracted", len(docs))
            return docs
        except Exception as exc:
            logger.error("OCR processing failed: %s", exc)
            raise RuntimeError(f"OCR processing failed: {exc}") from exc


# ── Singleton Factory ─────────────────────────────────────────
_ocr_processor_instance: OCRProcessor | None = None


def get_ocr_processor(enable_ocr: bool = True) -> OCRProcessor:
    """Return a (lazily created) singleton OCRProcessor."""
    global _ocr_processor_instance
    if _ocr_processor_instance is None:
        _ocr_processor_instance = OCRProcessor(enable_ocr=enable_ocr)
    return _ocr_processor_instance
