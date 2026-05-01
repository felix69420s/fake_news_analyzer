import re

# NOTE: These rules are heuristics and do not prove that a text is fake.

RULES_KEYWORDS: dict[str, list[str]] = {
    "emotionally_charged": [
        "ужас",
        "кошмар",
        "катастрофа",
        "скандал",
        "позор",
        "предательство",
        "жестокий",
        "чудовищный",
        "шокирующий",
        "разгром",
        "истерика",
    ],
    "fear_appeal": [
        "опасность",
        "угроза",
        "страх",
        "паника",
        "катастрофические последствия",
        "может погибнуть",
        "грозит",
        "рискует",
        "под угрозой",
        "налог",
        "сбор",
        "штраф",
        "запрет",
        "мобилизация",
        "уголовная ответственность",
        "лишение",
        "опасно",
        "угрожает",
        "пострадают",
    ],
    "generalization": [
        "все",
        "никто",
        "всегда",
        "никогда",
        "каждый",
        "любой",
        "везде",
        "все знают",
        "ни один",
    ],
    "sensationalism": [
        "шок",
        "срочно",
        "сенсация",
        "эксклюзив",
        "невероятно",
        "скандал",
        "раскрыта правда",
        "вся правда",
        "то, что скрывали",
    ],
    "pressure": [
        "немедленно",
        "обязательно",
        "срочно",
        "нельзя медлить",
        "прямо сейчас",
        "вы должны",
        "нужно срочно",
        "надо срочно",
        "требуется немедленно",
        "до конца дня",
        "иначе",
    ],
    "unverified_claims": [
        "якобы",
        "утверждается",
        "сообщается",
        "распространяется",
        "в соцсетях",
        "в социальных сетях",
        "в мессенджерах",
        "в telegram",
        "в telegram-каналах",
        "в телеграм",
        "в телеграм-каналах",
        "пользователи сообщают",
        "очевидцы сообщают",
        "без подтверждения",
        "не подтверждена",
        "неподтвержденная информация",
        "неподтверждённая информация",
        "по неподтвержденным данным",
        "по неподтверждённым данным",
    ],
    "black_white": [
        "только так",
        "другого выбора нет",
        "враг народа",
        "предатели",
        "патриоты против",
    ],
}

RULES_REGEX: dict[str, list[str]] = {
    "vague_sources": [
        r"\bэксперты считают\b",
        r"\bисточники сообщают\b",
        r"\bпо данным источников\b",
        r"\bкак стало известно\b",
        r"\bученые доказали\b",
        r"\bврачи предупреждают\b",
        r"\bаналитики заявили\b",
        r"\bинсайдеры сообщили\b",
        r"\bпо некоторым данным\b",
        r"\bв сети сообщают\b",
        r"\bв интернете пишут\b",
        r"\bпоявилась информация\b",
        r"\bпо сообщениям пользователей\b",
        r"\bанонимные источники\b",
        r"\bнеизвестные сообщили\b",
    ],
    "black_white": [
        r"\bлибо\b.+\bлибо\b",
        r"\bили\b.+\bили\b",
    ],
}


def normalize_for_rules(text: str) -> str:
    lowered = (text or "").lower()
    return re.sub(r"\s+", " ", lowered).strip()


def find_keyword_matches(text: str, keywords: list[str]) -> list[str]:
    normalized = normalize_for_rules(text)
    matches: list[str] = []
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, normalized) and keyword not in matches:
            matches.append(keyword)
    return matches


def find_regex_matches(text: str, patterns: list[str]) -> list[str]:
    normalized = normalize_for_rules(text)
    matches: list[str] = []
    for pattern in patterns:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            if pattern not in matches:
                matches.append(pattern)
    return matches


def extract_manipulation_features(text: str) -> dict:
    manipulation_matches: dict[str, list[str]] = {}
    manipulation_flags: dict[str, bool] = {}

    for category, keywords in RULES_KEYWORDS.items():
        keyword_matches = find_keyword_matches(text, keywords)
        manipulation_matches[category] = keyword_matches
        manipulation_flags[category] = len(keyword_matches) > 0

    for category, patterns in RULES_REGEX.items():
        regex_matches = find_regex_matches(text, patterns)
        existing = manipulation_matches.get(category, [])
        combined = existing + [item for item in regex_matches if item not in existing]
        manipulation_matches[category] = combined
        manipulation_flags[category] = len(combined) > 0

    punctuation_matches: list[str] = []
    raw_text = text or ""
    for marker in ["!!!", "???", "?!", "!?"]:
        if marker in raw_text:
            punctuation_matches.append(marker)
    exclamation_count = raw_text.count("!")
    if exclamation_count > 3:
        punctuation_matches.append("!>3")

    manipulation_matches["excessive_punctuation"] = punctuation_matches
    manipulation_flags["excessive_punctuation"] = len(punctuation_matches) > 0

    manipulation_score = sum(1 for flag in manipulation_flags.values() if flag)

    return {
        "manipulation_flags": manipulation_flags,
        "manipulation_matches": manipulation_matches,
        "manipulation_score": manipulation_score,
    }
