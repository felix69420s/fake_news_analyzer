def _scores_to_dict(model_output) -> dict[str, float]:
    if not isinstance(model_output, list):
        return {}
    if len(model_output) == 0:
        return {}

    items = model_output
    if len(model_output) == 1 and isinstance(model_output[0], list):
        items = model_output[0]

    scores: dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        score_raw = item.get("score", 0.0)
        try:
            score = float(score_raw)
        except Exception:
            score = 0.0
        scores[label] = score
    return scores


def get_top_label(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "", 0.0
    label = max(scores, key=scores.get)
    return label, float(scores[label])


def extract_sentiment_features(text: str, model_manager) -> dict:
    empty_result = {"sentiment_label": "", "sentiment_score": 0.0}
    try:
        sentiment_pipeline = model_manager.get_sentiment_pipeline()
        input_text = model_manager.truncate_text_for_model(text)
        output = sentiment_pipeline(input_text)
        scores = _scores_to_dict(output)
        label, score = get_top_label(scores)
        return {"sentiment_label": label, "sentiment_score": score}
    except Exception as exc:
        result = dict(empty_result)
        result["sentiment_error"] = str(exc)
        return result


def extract_emotional_profile(text: str, model_manager) -> dict:
    sentiment_features = extract_sentiment_features(text, model_manager)
    return {**sentiment_features}
