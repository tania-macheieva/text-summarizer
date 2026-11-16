"""Витягування тексту з TXT, PDF та DOCX."""
from __future__ import annotations

import io
from pathlib import Path

from ..config import ALLOWED_EXTENSIONS, MAX_TEXT_CHARS, MAX_UPLOAD_BYTES


class FileParseError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def _decode_txt(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1251", "cp1252"):
        try:
            text = data.decode(enc)
            if enc == "utf-16" and not data[:2] in (b"\xff\xfe", b"\xfe\xff"):
                continue
            return text
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _parse_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise FileParseError("The PDF is encrypted. Remove its password and try again.")
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
    except FileParseError:
        raise
    except (PdfReadError, Exception) as exc:
        raise FileParseError(f"Could not read the PDF: {exc}") from exc
    text = "\n\n".join(p for p in pages if p)
    if not text.strip():
        raise FileParseError("The PDF has no text layer (it may be a scanned document).")
    return text


def _parse_docx(data: bytes) -> str:
    from docx import Document

    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        raise FileParseError(f"Could not read the DOCX file: {exc}") from exc
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                paras.append(" ".join(cells))
    return "\n\n".join(paras)


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileParseError(f"Unsupported file type '{ext or '?'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", 415)
    if not data:
        raise FileParseError("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise FileParseError(f"The file is too large (maximum {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).", 413)
    text = {".txt": _decode_txt, ".pdf": _parse_pdf, ".docx": _parse_docx}[ext](data)
    if not text.strip():
        raise FileParseError("No text was found in the file.")
    if len(text) > MAX_TEXT_CHARS:
        raise FileParseError(f"The extracted text is too large (maximum {MAX_TEXT_CHARS} characters).", 413)
    return text
