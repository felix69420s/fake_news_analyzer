# Fake News Analyzer

Проект разработан для курсовой работы по теме анализа фейковых новостей и автоматизированной обработки медиатекстов.

## Назначение

`Fake News Analyzer` формирует аналитический датасет признаков по размеченным русскоязычным новостным текстам. Проект не выполняет самостоятельный фактчекинг: метка `fake/real` берётся из исходного датасета.

Один датасет: `morfifinka/fake-real-news-ru` с Kaggle.

## Что делает проект

1. Загружает датасет `fake-real-news-ru` из локального CSV-файла или из папки с `train_bodies.csv` и `train_stances.csv`.
2. Приводит записи к единой структуре проекта.
3. Кодирует метку достоверности:
   - `agree` или `1` -> `real`;
   - `disagree` или `0` -> `fake`.
4. Очищает и нормализует текст.
5. Извлекает именованные сущности:
   - персоны;
   - организации;
   - локации;
   - геополитические объекты;
   - медиа-сущности.
6. Определяет эмоциональный профиль текста:
   - тональность;
   - оценку тональности;
7. Выявляет модельные лингвистические признаки манипулятивной подачи через zero-shot классификацию.
8. Сохраняет результат в CSV и JSONL.

## Используемый датасет

Kaggle: `morfifinka/fake-real-news-ru`

Страница датасета: `https://www.kaggle.com/datasets/morfifinka/fake-real-news-ru`

Ожидаемые поля подготовленного CSV:

- `Body ID` — идентификатор новости;
- `Stance` — правдивость новости;
- `Headline` — исходный заголовок;
- `articleBody` — исходный текст новости.

Также поддерживается вариант с двумя файлами:

- `train_bodies.csv`;
- `train_stances.csv`.

Оба файла должны лежать в одной папке, например:

```text
data/raw/fake_real_news_ru/train_bodies.csv
data/raw/fake_real_news_ru/train_stances.csv
```

## Используемые модели Hugging Face

### NER

```text
r1char9/ner-rubert-tiny-news
```

### Sentiment

```text
blanchefort/rubert-base-cased-sentiment-rusentiment
```

### Манипулятивные лингвистические признаки

```text
MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
```

Модель используется в режиме zero-shot classification. Она возвращает вероятностные признаки, а не доказывает факт фейковости новости.

Категории анализа:

- апелляция к страху;
- эмоционально нагруженная лексика;
- сенсационная подача;
- неопределённый источник информации;
- категоричное обобщение;
- давление к немедленному действию;

## Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Получение датасета

Скачать датасет с Kaggle вручную и положить файлы в папку:

```text
data/raw/fake_real_news_ru/
```

Рекомендуемый вариант структуры:

```text
data/raw/fake_real_news_ru/train_bodies.csv
data/raw/fake_real_news_ru/train_stances.csv
```

## Запуск

### Рабочий запуск на 50 записях

```powershell
python -m src.main --data-path data/raw/fake_real_news_ru --limit 50 --output data/processed/kaggle_ru_analytical_dataset.csv
```

### Запуск с извлечением фрагментов-оснований

Этот режим медленнее, потому что дополнительно анализирует отдельные предложения.

```powershell
python -m src.main --data-path data/raw/fake_real_news_ru --limit 50 --include-evidence --output data/processed/kaggle_ru_analytical_dataset.csv
```


## Демонстрация разбора одной фейковой новости

```powershell
python src/show_fake_analysis.py
```

## Основные выходные поля

- `id` — идентификатор новости;
- `label` — метка из датасета: `fake` или `real`;
- `title` — заголовок;
- `text` — исходный текст;
- `normalized_text` — нормализованный текст;
- `named_entities` — найденные именованные сущности;
- `persons`, `organizations`, `locations`, `geopolitical_entities`, `media_entities` — сгруппированные сущности;
- `sentiment_label`, `sentiment_score` — тональность;
- `manipulation_flags` — бинарные признаки манипулятивной подачи по порогу;
- `manipulation_scores` — оценки zero-shot модели;
- `manipulation_score` — количество активных категорий;
- `manipulation_model` — модель, применённая для анализа;
- `manipulation_evidence_sentences` — фрагменты-основания, если включён `--include-evidence`.
