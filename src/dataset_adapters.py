import json
import re
from typing import Any

import pandas as pd

from src.schemas import InputNewsRecord


KAZAKH_SPECIFIC_RE = re.compile(r"[әғқңөұүһіӘҒҚҢӨҰҮҺІ]")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]")
RUSSIAN_RE = re.compile(r"[А-Яа-яЁё]")


def _is_missing(value: Any) -> bool:
    """Safely checks scalar missing values without breaking on dict/list."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def _safe_str(value: Any) -> str:
    """Converts scalar values to string. Dict/list are not flattened here."""
    if _is_missing(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return ""
    return str(value)


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        real_column = lowered.get(candidate.lower())
        if real_column is not None:
            return real_column
    return None


def _normalize_label(value: Any) -> str:
    normalized = _safe_str(value).strip().lower()
    if not normalized:
        return ""

    normalized = normalized.replace("-", "_").replace(" ", "_")

    fake_values = {
        "fake",
        "f",
        "false",
        "1",
        "fake_news",
        "fakenews",
        "фейк",
        "фейковая",
        "ложная",
        "недостоверная",
    }
    real_values = {
        "real",
        "r",
        "true",
        "0",
        "real_news",
        "realnews",
        "reliable",
        "достоверная",
        "настоящая",
        "правдивая",
    }

    # NOTE: Mapping for 0/1 can vary by dataset and should be verified
    # against the dataset description or representative examples.
    if normalized in fake_values:
        return "fake"
    if normalized in real_values:
        return "real"
    return normalized


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            return []
    return []


def _extract_from_nested_dict(value: Any, candidate_keys: list[str]) -> str:
    data = _as_dict(value)
    if not data:
        return ""

    lowered_map = {str(k).lower(): v for k, v in data.items()}
    for key in candidate_keys:
        current = lowered_map.get(key.lower())
        text = _safe_str(current).strip()
        if text:
            return text
    return ""


def _extract_list_text(value: Any) -> str:
    """Returns readable text if Label Studio value contains text as list/string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_safe_str(item).strip() for item in value]
        return " ".join(part for part in parts if part)
    return ""


def _extract_label_from_label_studio(row: pd.Series) -> str:
    # Some exports can already contain this field as a column.
    for column in ["main_classification", "classification", "label", "class", "target", "is_fake"]:
        if column in row.index:
            normalized = _normalize_label(row.get(column))
            if normalized in {"fake", "real"}:
                return normalized

    annotations = _as_list(row.get("annotations"))

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue

        # Sometimes classification is stored directly in annotation.
        for key in ["main_classification", "classification", "label"]:
            normalized = _normalize_label(annotation.get(key))
            if normalized in {"fake", "real"}:
                return normalized

        result_items = annotation.get("result", [])
        if not isinstance(result_items, list):
            continue

        for item in result_items:
            if not isinstance(item, dict):
                continue

            value = item.get("value", {})
            if not isinstance(value, dict):
                continue

            for field in ["choices", "textlabels", "labels"]:
                values = value.get(field, [])
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, list):
                    continue
                for item_value in values:
                    normalized = _normalize_label(item_value)
                    if normalized in {"fake", "real"}:
                        return normalized

            text_value = _extract_list_text(value.get("text"))
            normalized = _normalize_label(text_value)
            if normalized in {"fake", "real"}:
                return normalized

    return ""


def _extract_text_from_label_studio(row: pd.Series) -> str:
    # Plain tabular format.
    for column in ["text", "content", "news", "article", "body", "full_text"]:
        if column in row.index:
            direct_text = _safe_str(row.get(column)).strip()
            if direct_text:
                return direct_text

    nested_data = _as_dict(row.get("data"))
    if not nested_data:
        return ""

    claim = _extract_from_nested_dict(nested_data, ["claim", "CLAIM", "statement", "заявление"])
    evidence = _extract_from_nested_dict(
        nested_data,
        ["evidence", "EVIDENCE", "text", "news", "article", "content", "body", "текст"],
    )

    if claim and evidence and claim != evidence:
        return f"{claim}\n\n{evidence}"
    if evidence:
        return evidence
    if claim:
        return claim

    return _extract_from_nested_dict(
        nested_data,
        ["text", "news", "article", "content", "body", "full_text", "claim", "CLAIM"],
    )


def _extract_title(row: pd.Series, title_col: str | None) -> str:
    if title_col:
        title = _safe_str(row.get(title_col)).strip()
        if title:
            return title

    nested_data = _as_dict(row.get("data"))
    return _extract_from_nested_dict(nested_data, ["title", "headline", "name", "заголовок"])


def _extract_language(row: pd.Series) -> str:
    for column in ["language", "lang", "locale"]:
        if column in row.index:
            value = _safe_str(row.get(column)).strip().lower()
            if value:
                return value

    nested_data = _as_dict(row.get("data"))
    value = _extract_from_nested_dict(nested_data, ["language", "lang", "locale"]).strip().lower()
    return value or ""


def is_probably_russian(text: str, language_hint: str = "") -> bool:
    """
    Heuristic language filter for this course project.

    Goal: keep Russian texts and remove clearly Kazakh texts.
    It is intentionally softer than "any Kazakh-specific letter => remove",
    because Russian-language news can contain Kazakh names/toponyms.
    """
    if not text:
        return False

    language_hint = (language_hint or "").strip().lower()
    if language_hint in {"kk", "kaz", "kazakh", "kz", "қазақ", "kazakhstan_kazakh"}:
        return False

    lowered = str(text).lower()

    russian_count = len(RUSSIAN_RE.findall(lowered))
    kazakh_specific_count = len(KAZAKH_SPECIFIC_RE.findall(lowered))
    cyrillic_count = len(CYRILLIC_RE.findall(lowered))

    if cyrillic_count < 30:
        return False

    # If metadata explicitly says Russian, keep the record unless it is very clearly Kazakh.
    if language_hint in {"ru", "rus", "russian", "рус", "русский"}:
        if kazakh_specific_count >= 12 and kazakh_specific_count / max(cyrillic_count, 1) > 0.03:
            return False
        return True

    # Without metadata: remove texts with many Kazakh-specific symbols.
    if kazakh_specific_count >= 8:
        return False
    if kazakh_specific_count >= 4 and kazakh_specific_count / max(cyrillic_count, 1) > 0.015:
        return False

    # Keep texts that contain enough Russian Cyrillic.
    return russian_count >= 30


def adapt_baiangali_fake_news(df: pd.DataFrame) -> list[InputNewsRecord]:
    title_col = _first_existing_column(df, ["title", "headline", "name"])
    lead_col = _first_existing_column(df, ["lead", "summary", "description"])
    date_col = _first_existing_column(df, ["date", "published_at", "created_at", "time", "TIME"])
    source_col = _first_existing_column(df, ["source", "site", "publisher"])
    url_col = _first_existing_column(df, ["url", "link"])
    author_col = _first_existing_column(df, ["author"])
    label_col = _first_existing_column(df, ["label", "class", "target", "is_fake", "main_classification"])

    records: list[InputNewsRecord] = []
    has_id_column = "id" in df.columns

    skipped_empty_text = 0
    skipped_non_russian = 0
    skipped_empty_label = 0

    for idx, row in df.iterrows():
        nested_data = _as_dict(row.get("data"))

        text_value = _extract_text_from_label_studio(row).strip()
        language_hint = _extract_language(row)

        if not text_value:
            skipped_empty_text += 1
            continue

        if not is_probably_russian(text_value, language_hint=language_hint):
            skipped_non_russian += 1
            continue

        source_value = _safe_str(row.get(source_col)).strip() if source_col else ""
        if not source_value:
            source_value = _extract_from_nested_dict(nested_data, ["source", "SOURCE", "site", "publisher"])

        date_value = _safe_str(row.get(date_col)).strip() if date_col else ""
        if not date_value:
            date_value = _extract_from_nested_dict(
                nested_data,
                ["date", "time", "TIME", "published_at", "created_at", "created"],
            )

        language_value = language_hint or "ru"

        label_value = _normalize_label(row.get(label_col)) if label_col else ""
        if label_value not in {"fake", "real"}:
            label_value = _extract_label_from_label_studio(row) or label_value
        if not label_value:
            skipped_empty_label += 1

        record_id = _safe_str(row.get("id")).strip() if has_id_column else _safe_str(idx)

        record = InputNewsRecord(
            id=record_id,
            title=_extract_title(row, title_col),
            lead=_safe_str(row.get(lead_col)).strip() if lead_col else "",
            text=text_value,
            date=date_value,
            source=source_value,
            source_type="dataset",
            url=_safe_str(row.get(url_col)).strip() if url_col else "",
            author=_safe_str(row.get(author_col)).strip() if author_col else "",
            language="ru",
            label=label_value,
        )
        records.append(record)

    print(
        "Baiangali adapter: "
        f"input_rows={len(df)}, "
        f"created_records={len(records)}, "
        f"skipped_empty_text={skipped_empty_text}, "
        f"skipped_non_russian={skipped_non_russian}, "
        f"records_without_label={skipped_empty_label}"
    )

    return records


def adapt_lenta_extended(df: pd.DataFrame) -> list[InputNewsRecord]:
    title_col = _first_existing_column(df, ["title"])
    text_col = _first_existing_column(df, ["news", "text", "content"])
    date_col = _first_existing_column(df, ["date"])
    url_col = _first_existing_column(df, ["url"])

    records: list[InputNewsRecord] = []
    for idx, row in df.iterrows():
        records.append(
            InputNewsRecord(
                id=_safe_str(idx),
                title=_safe_str(row.get(title_col)).strip() if title_col else "",
                lead="",
                text=_safe_str(row.get(text_col)).strip() if text_col else "",
                date=_safe_str(row.get(date_col)).strip() if date_col else "",
                source="lenta.ru",
                source_type="news_site",
                url=_safe_str(row.get(url_col)).strip() if url_col else "",
                author="",
                language="ru",
                label="",
            )
        )
    return records


def adapt_dataset(df: pd.DataFrame, dataset_key: str) -> list[InputNewsRecord]:
    key = dataset_key.strip().lower()
    if key == "baiangali":
        return adapt_baiangali_fake_news(df)
    if key == "lenta":
        return adapt_lenta_extended(df)
    raise ValueError(f"Unsupported dataset_key: {dataset_key}")
