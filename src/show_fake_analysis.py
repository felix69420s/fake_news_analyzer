import ast
import json
from pathlib import Path
import pandas as pd

CSV_PATH = Path("data/processed/kaggle_ru_analytical_dataset.csv")


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


def short_text(text, limit=1200):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Файл {CSV_PATH} не найден. Сначала запусти: "
        "python -m src.main --data-path data/raw/fake_real_news_ru --limit 50"
    )

df = pd.read_csv(CSV_PATH)
fake_df = df[df["label"].astype(str).str.lower() == "fake"].copy()
if fake_df.empty:
    raise RuntimeError("В файле нет строк с label=fake. Проверь поле Stance или увеличь --limit.")

fake_df["manipulation_score_num"] = pd.to_numeric(
    fake_df["manipulation_score"], errors="coerce"
).fillna(0)
fake_df = fake_df.sort_values(by="manipulation_score_num", ascending=False)
row = fake_df.iloc[0]

persons = parse_cell(row.get("persons"))
organizations = parse_cell(row.get("organizations"))
locations = parse_cell(row.get("locations"))
geopolitical_entities = parse_cell(row.get("geopolitical_entities"))
named_entities = parse_cell(row.get("named_entities"))
manipulation_flags = parse_cell(row.get("manipulation_flags"))
manipulation_scores = parse_cell(row.get("manipulation_scores"))
evidence = parse_cell(row.get("manipulation_evidence_sentences"))

print("АНАЛИЗ ФЕЙКОВОЙ НОВОСТИ")
print("\n1. ИСХОДНЫЕ ДАННЫЕ")
print("ID:", row.get("id"))
print("LABEL:", row.get("label"))
print("TITLE:", row.get("title"))
print("\nTEXT:")
print(short_text(row.get("text"), 1500))

print("\n2. ИЗВЛЕЧЁННЫЕ СУЩНОСТИ")
print("Персоны:", persons)
print("Организации:", organizations)
print("Локации:", locations)
print("Геополитические объекты:", geopolitical_entities)
print("\nВсе найденные сущности:")
print(named_entities)

print("\n3. ЭМОЦИОНАЛЬНЫЙ ПРОФИЛЬ")
print("Тональность:", row.get("sentiment_label"))
print("Оценка тональности:", row.get("sentiment_score"))

print("\n4. ЛИНГВИСТИЧЕСКИЕ ПРИЗНАКИ МАНИПУЛЯЦИИ")
print("Метод:", row.get("manipulation_method"))
print("Модель:", row.get("manipulation_model"))
print("Порог:", row.get("manipulation_threshold"))
print("Флаги:")
print(manipulation_flags)
print("\nОценки модели:")
print(manipulation_scores)
print("\nФрагменты-основания, если были рассчитаны:")
print(evidence)
print("\nManipulation score:", row.get("manipulation_score"))

print("\n5. КРАТКИЙ ВЫВОД")
print("-" * 100)
active_flags = []
if isinstance(manipulation_flags, dict):
    active_flags = [k for k, v in manipulation_flags.items() if v]
print(
    "В выбранной фейковой новости модуль извлёк сведения о сущностях, "
    "эмоциональном профиле и модельных признаках манипулятивной подачи."
)
print("Активные признаки манипулятивной подачи:", active_flags)
if persons or organizations or locations or geopolitical_entities:
    print("В тексте обнаружены упоминания субъектов и объектов новости.")
else:
    print("Именованные сущности в выбранной записи не обнаружены или не распознаны моделью.")
