from typing import Any
import re

import pandas as pd

from src.config import KAGGLE_DATASET_URL
from src.schemas import InputNewsRecord


TOKEN_LIST_RE = re.compile(r"^\s*\[(?:\s*['\"][^'\"]+['\"]\s*,?)+\]\s*$")


class KaggleDatasetAdapterError(ValueError):
    """Raised when the Kaggle dataset cannot be adapted safely."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def _safe_str(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()


def _looks_like_token_list(value: str) -> bool:
    text = _safe_str(value)
    if not text:
        return False
    if TOKEN_LIST_RE.match(text):
        return True
    return text.startswith("[") and text.endswith("]") and ("'" in text or '"' in text)


def _clean_dataset_text(value: Any) -> str:
    text = _safe_str(value)
    if not text or _looks_like_token_list(text):
        return ""
    return text


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        real_column = lowered.get(candidate.lower())
        if real_column is not None:
            return real_column
    return None


def _require_column(df: pd.DataFrame, candidates: list[str], logical_name: str) -> str:
    column = _first_existing_column(df, candidates)
    if column is None:
        raise KaggleDatasetAdapterError(
            f"Kaggle adapter cannot find required column '{logical_name}'. "
            f"Tried: {', '.join(candidates)}. Available columns: {', '.join(map(str, df.columns))}"
        )
    return column


def normalize_kaggle_stance(value: Any) -> str:
    """Normalizes the Kaggle dataset truth label.

    The dataset description/code uses: agree -> 1, disagree -> 0.
    In this project: agree/1 = real, disagree/0 = fake.
    """
    normalized = _safe_str(value).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")

    mapping = {
        "agree": "real",
        "1": "real",
        "true": "real",
        "real": "real",
        "правда": "real",
        "правдивая": "real",
        "достоверная": "real",
        "disagree": "fake",
        "0": "fake",
        "false": "fake",
        "fake": "fake",
        "фейк": "fake",
        "ложная": "fake",
        "недостоверная": "fake",
    }
    if normalized in mapping:
        return mapping[normalized]
    raise KaggleDatasetAdapterError(
        f"Unsupported Stance value '{value}'. Expected agree/disagree or already encoded 1/0."
    )


def _select_title(row: pd.Series, title_col: str | None, title_fallback_col: str | None) -> str:
    title = _clean_dataset_text(row.get(title_col)) if title_col else ""
    if title:
        return title

    fallback = _clean_dataset_text(row.get(title_fallback_col)) if title_fallback_col else ""
    return fallback


def _select_article_body(row: pd.Series, text_col: str, text_fallback_col: str | None) -> str:
    text = _clean_dataset_text(row.get(text_col))
    if text:
        return text

    fallback = _clean_dataset_text(row.get(text_fallback_col)) if text_fallback_col else ""
    return fallback


def adapt_kaggle_fake_real_news_ru(df: pd.DataFrame) -> list[InputNewsRecord]:

    id_col = _first_existing_column(df, ["Body ID", "body_id", "id", "Unnamed: 0"])
    label_col = _require_column(df, ["Stance", "label", "target", "class"], "Stance")

    # Prefer original columns only. Fallbacks are restricted to cleaned text fields.
    title_col = _first_existing_column(df, ["Headline", "headline", "title"])
    title_fallback_col = _first_existing_column(df, ["Headline1", "headline1", "clean_headline"])

    text_col = _require_column(
        df,
        ["articleBody", "article_body", "articleBody_original", "Body", "body", "text", "content"],
        "articleBody",
    )
    text_fallback_col = _first_existing_column(df, ["articleBody1", "article_body1", "clean_body"])

    records: list[InputNewsRecord] = []
    skipped_empty_text = 0
    skipped_bad_label = 0
    skipped_tokenized_title = 0
    skipped_tokenized_text = 0

    for idx, row in df.iterrows():
        raw_title_value = row.get(title_col) if title_col else ""
        raw_text_value = row.get(text_col) if text_col else ""

        if _looks_like_token_list(_safe_str(raw_title_value)):
            skipped_tokenized_title += 1
        if _looks_like_token_list(_safe_str(raw_text_value)):
            skipped_tokenized_text += 1

        title = _select_title(row, title_col, title_fallback_col)
        text = _select_article_body(row, text_col, text_fallback_col)

        if not text:
            skipped_empty_text += 1
            continue

        try:
            label = normalize_kaggle_stance(row.get(label_col))
        except KaggleDatasetAdapterError:
            skipped_bad_label += 1
            continue

        record_id = _safe_str(row.get(id_col)) if id_col else str(idx)
        if not record_id:
            record_id = str(idx)

        records.append(
            InputNewsRecord(
                id=record_id,
                title=title,
                lead="",
                text=text,
                date="",
                source="Kaggle: morfifinka/fake-real-news-ru",
                source_type="dataset",
                url=KAGGLE_DATASET_URL,
                author="",
                language="ru",
                label=label,
            )
        )

    print(
        "Kaggle fake-real-news-ru adapter: "
        f"input_rows={len(df)}, "
        f"created_records={len(records)}, "
        f"skipped_empty_text={skipped_empty_text}, "
        f"skipped_bad_label={skipped_bad_label}, "
        f"tokenized_title_artifacts_seen={skipped_tokenized_title}, "
        f"tokenized_text_artifacts_seen={skipped_tokenized_text}"
    )
    return records


def adapt_dataset(df: pd.DataFrame) -> list[InputNewsRecord]:
    return adapt_kaggle_fake_real_news_ru(df)
