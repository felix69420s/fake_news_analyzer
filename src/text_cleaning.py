import hashlib
import re

from bs4 import BeautifulSoup
from razdel import tokenize


def clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    cleaned = clean_html(text)
    normalized = cleaned.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("‘", "'").replace("’", "'")
    normalized = normalized.replace("—", "-").replace("–", "-")
    return normalize_spaces(normalized)


def build_full_text(title: str, lead: str, text: str) -> str:
    parts = [part.strip() for part in [title, lead, text] if part and part.strip()]
    return "\n\n".join(parts)


def make_text_hash(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def count_chars(text: str) -> int:
    return len(text)


def count_tokens_approx(text: str) -> int:
    try:
        return sum(1 for _ in tokenize(text))
    except Exception:
        return len(text.split())
