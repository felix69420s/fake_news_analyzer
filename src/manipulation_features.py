from razdel import sentenize


MANIPULATION_LABELS: dict[str, str] = {
    "fear_appeal": "апелляция к страху",
    "emotionally_charged": "эмоционально нагруженная лексика",
    "sensationalism": "сенсационная подача",
    "vague_source": "неопределённый источник информации",
    "categorical_generalization": "категоричное обобщение",
    "urgency_pressure": "давление к немедленному действию",
}

HYPOTHESIS_TEMPLATE = "В данном новостном тексте присутствует {}."


def _score_zero_shot(text: str, labels: list[str], classifier) -> dict[str, float]:
    if not text.strip() or not labels:
        return {}
    result = classifier(
        text,
        candidate_labels=labels,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=True,
    )
    raw_labels = result.get("labels", []) if isinstance(result, dict) else []
    raw_scores = result.get("scores", []) if isinstance(result, dict) else []
    scores: dict[str, float] = {}
    for label, score in zip(raw_labels, raw_scores):
        try:
            scores[str(label)] = float(score)
        except Exception:
            scores[str(label)] = 0.0
    return scores


def _split_sentences(text: str, max_sentences: int) -> list[str]:
    sentences: list[str] = []
    try:
        for sentence in sentenize(text):
            value = sentence.text.strip()
            if len(value) >= 25:
                sentences.append(value)
            if len(sentences) >= max_sentences:
                break
    except Exception:
        sentences = [part.strip() for part in text.split(".") if len(part.strip()) >= 25]
        sentences = sentences[:max_sentences]
    return sentences


def _extract_evidence_sentences(
    text: str,
    active_categories: list[str],
    classifier,
    max_sentences: int,
    max_categories: int = 3,
) -> dict[str, list[dict]]:
    sentences = _split_sentences(text, max_sentences=max_sentences)
    if not sentences or not active_categories:
        return {}

    evidence: dict[str, list[dict]] = {}
    selected_categories = active_categories[:max_categories]
    for category_key in selected_categories:
        label = MANIPULATION_LABELS[category_key]
        best: dict | None = None
        for sentence in sentences:
            scores = _score_zero_shot(sentence, [label], classifier)
            score = scores.get(label, 0.0)
            if best is None or score > best["score"]:
                best = {"sentence": sentence, "score": round(float(score), 4)}
        if best is not None:
            evidence[category_key] = [best]
    return evidence


def extract_manipulation_features(
    text: str,
    model_manager,
    threshold: float = 0.55,
    max_chars: int = 1500,
    include_evidence: bool = False,
    max_evidence_sentences: int = 8,
) -> dict:
    """Extracts manipulation-related linguistic features with a multilingual NLI model.

    The function returns probabilistic indicators. They are analytical features, not proof
    that the news item is fake.
    """
    base_text = (text or "")[:max_chars]
    labels = list(MANIPULATION_LABELS.values())
    label_to_key = {label: key for key, label in MANIPULATION_LABELS.items()}

    empty_result = {
        "manipulation_flags": {key: False for key in MANIPULATION_LABELS},
        "manipulation_scores": {key: 0.0 for key in MANIPULATION_LABELS},
        "manipulation_score": 0,
        "manipulation_threshold": threshold,
        "manipulation_evidence_sentences": {},
    }

    if not base_text.strip():
        return empty_result

    try:
        classifier = model_manager.get_manipulation_pipeline()
        raw_scores = _score_zero_shot(base_text, labels, classifier)
    except Exception as exc:
        result = dict(empty_result)
        result["manipulation_error"] = str(exc)
        return result

    scores = {key: 0.0 for key in MANIPULATION_LABELS}
    for label, score in raw_scores.items():
        key = label_to_key.get(label)
        if key:
            scores[key] = round(float(score), 4)

    flags = {key: value >= threshold for key, value in scores.items()}
    active_categories = [key for key, is_active in flags.items() if is_active]

    evidence = {}
    if include_evidence and active_categories:
        evidence = _extract_evidence_sentences(
            base_text,
            active_categories=active_categories,
            classifier=classifier,
            max_sentences=max_evidence_sentences,
        )

    return {
        "manipulation_flags": flags,
        "manipulation_scores": scores,
        "manipulation_score": sum(1 for value in flags.values() if value),
        "manipulation_threshold": threshold,
        "manipulation_evidence_sentences": evidence,
    }
