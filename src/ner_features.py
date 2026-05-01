def normalize_entity_group(label: str) -> str:
    normalized = (label or "").strip().upper()

    mapping = {
        "PER": "PER",
        "PERSON": "PER",
        "B-PER": "PER",
        "I-PER": "PER",
        "ORG": "ORG",
        "ORGANIZATION": "ORG",
        "B-ORG": "ORG",
        "I-ORG": "ORG",
        "LOC": "LOC",
        "LOCATION": "LOC",
        "B-LOC": "LOC",
        "I-LOC": "LOC",
        "GPE": "GPE",
        "GEOPOLITICAL": "GPE",
        "B-GPE": "GPE",
        "I-GPE": "GPE",
        "MEDIA": "MEDIA",
        "MASS_MEDIA": "MEDIA",
    }
    return mapping.get(normalized, "OTHER")


def deduplicate_entities(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_ner_features(text: str, model_manager) -> dict:
    empty_result = {
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

    try:
        ner_pipeline = model_manager.get_ner_pipeline()
        input_text = model_manager.truncate_text_for_model(text)
        raw_entities = ner_pipeline(input_text)
    except Exception as exc:
        result = dict(empty_result)
        result["ner_error"] = str(exc)
        return result

    persons: list[str] = []
    organizations: list[str] = []
    locations: list[str] = []
    geopolitical_entities: list[str] = []
    media_entities: list[str] = []
    named_entities: list[dict] = []

    if not isinstance(raw_entities, list):
        raw_entities = []

    for entity in raw_entities:
        if not isinstance(entity, dict):
            continue

        entity_text = str(entity.get("word", entity.get("text", ""))).strip()
        if not entity_text:
            continue

        raw_label = str(entity.get("entity_group", entity.get("entity", "")))
        label = normalize_entity_group(raw_label)

        score_raw = entity.get("score", 0.0)
        try:
            score = float(score_raw)
        except Exception:
            score = 0.0

        named_entities.append({"text": entity_text, "label": label, "score": score})

        if label == "PER":
            persons.append(entity_text)
        elif label == "ORG":
            organizations.append(entity_text)
        elif label == "LOC":
            locations.append(entity_text)
        elif label == "GPE":
            geopolitical_entities.append(entity_text)
        elif label == "MEDIA":
            media_entities.append(entity_text)

    persons = deduplicate_entities(persons)
    organizations = deduplicate_entities(organizations)
    locations = deduplicate_entities(locations)
    geopolitical_entities = deduplicate_entities(geopolitical_entities)
    media_entities = deduplicate_entities(media_entities)

    return {
        "named_entities": named_entities,
        "persons": persons,
        "organizations": organizations,
        "locations": locations,
        "geopolitical_entities": geopolitical_entities,
        "media_entities": media_entities,
        "persons_count": len(persons),
        "organizations_count": len(organizations),
        "locations_count": len(locations),
        "geopolitical_count": len(geopolitical_entities),
        "media_count": len(media_entities),
    }
