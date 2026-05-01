import pandas as pd
import json
import ast
from pathlib import Path


CSV_PATH = Path("data/processed/baiangali.csv")
TOP_N = 5


def parse_cell(value):
    if pd.isna(value):
        return None
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except Exception:
        pass

    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _safe_len(value) -> int:
    parsed = parse_cell(value)
    if isinstance(parsed, (list, dict, tuple, set)):
        return len(parsed)
    return 0


def _matches_count(value) -> int:
    parsed = parse_cell(value)
    if isinstance(parsed, dict):
        total = 0
        for item in parsed.values():
            if isinstance(item, list):
                total += len(item)
        return total
    if isinstance(parsed, list):
        return len(parsed)
    return 0


def _build_interest_score(row: pd.Series) -> float:
    manipulation_score = float(row.get("manipulation_score", 0) or 0)
    entities_count = _safe_len(row.get("named_entities"))
    matches_count = _matches_count(row.get("manipulation_matches"))
    token_count = float(pd.to_numeric(row.get("token_count_approx", 0), errors="coerce") or 0)

    return manipulation_score * 3 + matches_count * 2 + entities_count + min(token_count / 200, 3)


def main():
    df = pd.read_csv(CSV_PATH)

    fake = df[df["label"].astype(str).str.lower() == "fake"].copy()
    if fake.empty:
        raise RuntimeError("Нет fake-новостей.")

    fake["manipulation_score"] = pd.to_numeric(
        fake["manipulation_score"],
        errors="coerce"
    ).fillna(0)

    fake["interest_score"] = fake.apply(_build_interest_score, axis=1)
    fake = fake.sort_values(
        ["interest_score", "manipulation_score"],
        ascending=False,
    )
    top_fake = fake.head(TOP_N)

    print("=" * 100)
    print(f"ТОП-{len(top_fake)} НАИБОЛЕЕ ИНТЕРЕСНЫХ ФЕЙКОВЫХ НОВОСТЕЙ")
    print("=" * 100)

    for rank, (_, row) in enumerate(top_fake.iterrows(), start=1):
        print("\n" + "-" * 100)
        print(f"#{rank} | ID: {row['id']} | LABEL: {row['label']}")
        print(
            f"interest_score={row['interest_score']:.2f} | "
            f"manipulation_score={row['manipulation_score']}"
        )

        print("\nТЕКСТ:")
        print(str(row["text"])[:1000])

        print("\nСУЩНОСТИ:")
        print("Персоны:", parse_cell(row["persons"]))
        print("Организации:", parse_cell(row["organizations"]))
        print("Локации:", parse_cell(row["locations"]))
        print("Геополитические объекты:", parse_cell(row["geopolitical_entities"]))

        print("\nЭМОЦИОНАЛЬНЫЙ ПРОФИЛЬ:")
        print("Тональность:", row["sentiment_label"])
        print("Оценка тональности:", row["sentiment_score"])
        print("Доминирующая эмоция:", row["dominant_emotion"])
        print("Оценки эмоций:", parse_cell(row["emotion_scores"]))

        print("\nЛИНГВИСТИЧЕСКИЕ ПРИЗНАКИ МАНИПУЛЯЦИИ:")
        print("Флаги:", parse_cell(row["manipulation_flags"]))
        print("Совпадения:", parse_cell(row["manipulation_matches"]))
        print("Manipulation score:", row["manipulation_score"])

    print("\nВЫВОД:")
    print(
        "Из фейковых новостей извлечены именованные сущности, "
        "эмоциональный профиль и rule-based признаки манипулятивности. "
        "Метка fake взята из исходного датасета, модуль не выполняет фактчекинг."
    )


if __name__ == "__main__":
    main()