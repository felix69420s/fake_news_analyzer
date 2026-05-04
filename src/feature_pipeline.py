from src.config import DEFAULT_MANIPULATION_MAX_CHARS, DEFAULT_MANIPULATION_THRESHOLD
from src.manipulation_features import extract_manipulation_features
from src.sentiment_features import extract_sentiment_profile
from src.ner_features import extract_ner_features
from src.schemas import InputNewsRecord, OutputNewsRecord
from src.text_cleaning import (
    build_full_text,
    count_chars,
    count_tokens_approx,
    make_text_hash,
    normalize_text,
)


def process_record(
    record: InputNewsRecord,
    model_manager,
    manipulation_threshold: float = DEFAULT_MANIPULATION_THRESHOLD,
    manipulation_max_chars: int = DEFAULT_MANIPULATION_MAX_CHARS,
    include_evidence: bool = False,
) -> OutputNewsRecord:
    title = record.title or ""
    lead = record.lead or ""
    text = record.text or ""
    full_text = build_full_text(title=title, lead=lead, text=text)
    normalized_text = normalize_text(full_text) if full_text else ""
    text_hash = make_text_hash(normalized_text) if normalized_text else ""
    char_count = count_chars(normalized_text)
    token_count_approx = count_tokens_approx(normalized_text)

    if normalized_text:
        ner_features = extract_ner_features(normalized_text, model_manager)
        sentiment_profile = extract_sentiment_profile(normalized_text, model_manager)
        manipulation_features = extract_manipulation_features(
            normalized_text,
            model_manager=model_manager,
            threshold=manipulation_threshold,
            max_chars=manipulation_max_chars,
            include_evidence=include_evidence,
        )
    else:
        ner_features = {
            "named_entities": [],
            "persons": [],
            "organizations": [],
            "locations": [],
            "geopolitical_entities": [],
            "media_entities": [],
            "persons_count": 0,
            "organizations_count": 0,
            "locations_count": 0,
            "geopolitical_count": 0,
            "media_count": 0,
        }
        sentiment_profile = {
            "sentiment_label": "",
            "sentiment_score": 0.0,
        }
        manipulation_features = {
            "manipulation_flags": {},
            "manipulation_scores": {},
            "manipulation_score": 0,
            "manipulation_method": "zero_shot_nli",
            "manipulation_model": "",
            "manipulation_threshold": manipulation_threshold,
            "manipulation_evidence_sentences": {},
        }

    flags_value = manipulation_features.get("manipulation_flags", {})
    manipulation_flags = {str(k): bool(v) for k, v in flags_value.items()} if isinstance(flags_value, dict) else {}

    scores_value = manipulation_features.get("manipulation_scores", {})
    manipulation_scores = (
        {str(k): float(v) for k, v in scores_value.items()} if isinstance(scores_value, dict) else {}
    )

    evidence_value = manipulation_features.get("manipulation_evidence_sentences", {})
    manipulation_evidence = evidence_value if isinstance(evidence_value, dict) else {}

    return OutputNewsRecord(
        id=record.id,
        date=record.date or "",
        source=record.source or "",
        source_type=record.source_type or "",
        url=record.url or "",
        author=record.author or "",
        language=record.language or "ru",
        label=record.label or "",
        title=title,
        lead=lead,
        text=text,
        full_text=full_text,
        normalized_text=normalized_text,
        text_hash=text_hash,
        char_count=char_count,
        token_count_approx=token_count_approx,
        named_entities=ner_features.get("named_entities", []),
        persons=ner_features.get("persons", []),
        organizations=ner_features.get("organizations", []),
        locations=ner_features.get("locations", []),
        geopolitical_entities=ner_features.get("geopolitical_entities", []),
        media_entities=ner_features.get("media_entities", []),
        persons_count=ner_features.get("persons_count", 0),
        organizations_count=ner_features.get("organizations_count", 0),
        locations_count=ner_features.get("locations_count", 0),
        geopolitical_count=ner_features.get("geopolitical_count", 0),
        media_count=ner_features.get("media_count", 0),
        sentiment_label=sentiment_profile.get("sentiment_label", ""),
        sentiment_score=sentiment_profile.get("sentiment_score", 0.0),
        manipulation_flags=manipulation_flags,
        manipulation_scores=manipulation_scores,
        manipulation_score=manipulation_features.get("manipulation_score", 0),
        manipulation_method=manipulation_features.get("manipulation_method", ""),
        manipulation_model=manipulation_features.get("manipulation_model", ""),
        manipulation_threshold=manipulation_features.get(
            "manipulation_threshold", manipulation_threshold
        ),
        manipulation_evidence_sentences=manipulation_evidence,
    )
