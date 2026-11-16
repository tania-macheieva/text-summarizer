"""Очищення тексту, розбиття на речення (NLTK) та токенізація."""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache

from .language import detect_language, english_stemmer, stopwords_for

logger = logging.getLogger(__name__)

_CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F\u200B\u200C\u200D\uFEFF]")
_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)
_TERMINAL = re.compile(r"[.!?…][\"'”’»)\]]*\s*$")
_NUMBER = re.compile(r"\d")
_QUOTE = re.compile(r"[\"“”«»„]")

UK_ABBREVIATIONS = {
    "т.д", "т.п", "т.ін", "ін", "грн", "тис", "млн", "млрд", "р", "рр", "ст", "вул", "ім", "див",
    "напр", "проф", "доц", "акад", "див", "п", "пп", "с", "м", "обл", "див", "ред", "тел", "кв", "буд",
}

_nltk_ready: bool | None = None


def ensure_nltk() -> bool:
    """Перевіряє/завантажує потрібні ресурси NLTK. False — якщо їх немає (використовуємо regex)."""
    global _nltk_ready
    if _nltk_ready is not None:
        return _nltk_ready
    import nltk

    ok = True
    for pkg, path in (("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                ok = bool(nltk.download(pkg, quiet=True)) and ok
            except Exception:  # pragma: no cover - немає мережі
                ok = False
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        ok = False
    _nltk_ready = ok
    if not ok:
        logger.warning("Ресурси NLTK недоступні — використовується regex-сегментатор речень")
    return ok


def clean_text(text: str) -> str:
    """Нормалізує Unicode, прибирає керівні символи, зберігає межі абзаців."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = _CONTROL.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_title(text: str) -> tuple[str | None, str]:
    """Відокремлює заголовок: перший рядок без кінцевої пунктуації, за яким іде порожній рядок."""
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        head = parts[0].strip()
        if head and "\n" not in head and len(head.split()) <= 25 and not _TERMINAL.search(head):
            return head, parts[1].strip()
    return None, text


def fix_missing_spaces(text: str) -> str:
    """У BBC News Summary речення в резюме склеєні без пробілу ('time.She said') — відновлюємо."""
    return re.sub(r'([a-z0-9][.!?]["”’]?)(?=[A-Z])', r"\1 ", text)


def _regex_split(text: str, language: str) -> list[str]:
    """Резервний сегментатор (і основний для української)."""
    pieces: list[str] = []
    start = 0
    for m in re.finditer(r"[.!?…]+[\"'”’»)\]]*\s+(?=[\"'“«(\[]?[A-ZА-ЯІЇЄҐ0-9])", text):
        end = m.end()
        candidate = text[start:end].strip()
        last = re.findall(r"(\S+)\s*$", text[start:m.start() + 1])
        token = last[0].rstrip(".!?…").lower() if last else ""
        if language == "uk" and (token in UK_ABBREVIATIONS or (len(token) == 1 and token.isalpha())):
            continue  # скорочення або ініціал, а не кінець речення
        if candidate:
            pieces.append(candidate)
        start = end
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def split_sentences(text: str, language: str = "en") -> list[str]:
    """Розбиває документ на речення: абзаци -> NLTK Punkt (англ.) або regex (укр.)."""
    use_nltk = language == "en" and ensure_nltk()
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip()
        if not para:
            continue
        if use_nltk:
            from nltk.tokenize import sent_tokenize

            parts = sent_tokenize(para, language="english")
        else:
            parts = _regex_split(para, language)
        for p in parts:
            p = p.strip()
            if len(_WORD.findall(p)) >= 1:
                sentences.append(p)
    return sentences


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


@lru_cache(maxsize=200_000)
def _stem_en(token: str) -> str:
    return english_stemmer().stem(token)


def tokenize(sentence: str, language: str = "en", remove_stopwords: bool = True) -> list[str]:
    """Токени для TF-IDF: нижній регістр, без стоп-слів, стемінг (лише англійська)."""
    stop = stopwords_for(language) if remove_stopwords else frozenset()
    out: list[str] = []
    for tok in _TOKEN.findall(sentence.lower()):
        if tok in stop:
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        out.append(_stem_en(tok) if language == "en" and tok.isalpha() else tok)
    return out


def has_number(sentence: str) -> bool:
    return bool(_NUMBER.search(sentence))


def has_quote(sentence: str) -> bool:
    return bool(_QUOTE.search(sentence))


class Preprocessor:
    """Об'єднує кроки: очищення -> мова -> заголовок -> речення -> токени."""

    def __init__(self, drop_title: bool = True):
        self.drop_title = drop_title

    def process(self, text: str, language: str = "auto") -> dict:
        cleaned = clean_text(text)
        lang = detect_language(cleaned) if language in (None, "", "auto") else language
        title, body = split_title(cleaned) if self.drop_title else (None, cleaned)
        sentences = split_sentences(body, lang)
        if not sentences and title:  # документ складається лише із заголовка
            sentences = split_sentences(title, lang)
            title = None
        return {
            "language": lang,
            "title": title,
            "sentences": sentences,
            "tokens": [tokenize(s, lang) for s in sentences],
            "cleaned": cleaned,
        }
