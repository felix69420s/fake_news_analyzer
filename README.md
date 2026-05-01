# Fake News Analyzer

Проект разработан для курсовой работы по теме анализа фейковых новостей и автоматизированной обработки медиатекстов.

## Назначение проекта

`Fake News Analyzer` — это модуль для извлечения интерпретируемых признаков из новостных текстов.

Проект **не является системой фактчекинга** и **не определяет самостоятельно**, является ли новость фейковой.  
Метка `fake/real` берётся из исходного размеченного датасета.

Основная задача проекта — обработать уже готовые датасеты медиатекстов и сформировать аналитический датасет с признаками, которые можно использовать в прикладной части курсовой работы.

## Что делает проект

Проект выполняет следующие этапы:

1. Загружает готовые датасеты.
2. Приводит разные форматы данных к единой структуре.
3. Фильтрует нерусскоязычные записи для основного датасета.
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
   - эмоции;
   - доминирующую эмоцию.
7. Выявляет rule-based признаки манипулятивности:
   - эмоционально нагруженную лексику;
   - апелляцию к страху;
   - неопределённые источники;
   - обобщения;
   - сенсационность;
   - давление;
   - чёрно-белое противопоставление;
   - чрезмерную пунктуацию.
8. Сохраняет итоговый аналитический датасет в CSV и JSONL.

## Используемые датасеты

### 1. Основной датасет

Используется GitHub-репозиторий:

```text
https://github.com/baiangali/fake_news
```

Файл датасета:

```text
fake_real_news_annotated.json
```

Особенности:

- формат: JSON-экспорт Label Studio;
- содержит разметку `fake/real`;
- содержит тексты на русском и казахском языках;
- в проекте используются только русскоязычные записи;
- казахские записи отфильтровываются на этапе адаптации датасета.

### 2. Дополнительный датасет

Используется Hugging Face датасет:

```text
data-silence/lenta.ru_2-extended
```

Особенности:

- большой корпус новостей Lenta.ru;
- не содержит разметки `fake/real`;
- используется для проверки работы предобработки и извлечения признаков на большом корпусе новостных текстов;
- не используется как основной датасет для анализа фейковых новостей.

## Используемые модели Hugging Face

### NER

```text
r1char9/ner-rubert-tiny-news
```

Используется для извлечения именованных сущностей.

### Emotion

```text
cointegrated/rubert-tiny2-cedr-emotion-detection
```

Используется для определения эмоционального профиля текста.

### Sentiment

```text
blanchefort/rubert-base-cased-sentiment-rusentiment
```

Используется для определения тональности текста.

### Манипулятивные признаки

Признаки манипулятивности реализованы rule-based способом, без отдельной ML-модели.  
Это сделано для интерпретируемости: каждое срабатывание связано с конкретным словом, фразой или регулярным выражением.

## Структура проекта

```text
fake_news_analyzer/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── dataset_loader.py
│   ├── dataset_adapters.py
│   ├── text_cleaning.py
│   ├── hf_models.py
│   ├── ner_features.py
│   ├── emotion_features.py
│   ├── manipulation_features.py
│   ├── feature_pipeline.py
│   ├── dataset_builder.py
│   ├── main.py
│   └── show_fake_analysis.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Назначение основных файлов

### `src/config.py`

Хранит настройки проекта:

- пути к папкам данных;
- названия моделей Hugging Face;
- названия датасетов;
- путь к выходному файлу по умолчанию.

### `src/schemas.py`

Описывает входную и выходную структуру данных через Pydantic-модели.

### `src/dataset_loader.py`

Отвечает за загрузку данных:

- из Hugging Face;
- из локальных файлов `.csv`, `.json`, `.jsonl`, `.parquet`.

### `src/dataset_adapters.py`

Приводит разные датасеты к единой структуре проекта.

Также содержит фильтрацию русскоязычных записей для датасета `baiangali/fake_news`.

### `src/text_cleaning.py`

Выполняет очистку и нормализацию текста:

- удаление HTML;
- нормализация пробелов;
- сборка полного текста новости;
- подсчёт символов;
- примерный подсчёт токенов;
- вычисление хэша текста.

### `src/hf_models.py`

Централизованно загружает модели Hugging Face.

Модели загружаются лениво, то есть только в момент первого использования.

### `src/ner_features.py`

Извлекает именованные сущности:

- персоны;
- организации;
- локации;
- геополитические объекты;
- медиа-сущности.

### `src/emotion_features.py`

Извлекает эмоциональный профиль:

- тональность;
- score тональности;
- список эмоций;
- score эмоций;
- доминирующую эмоцию.

### `src/manipulation_features.py`

Ищет признаки манипулятивности по словарям и регулярным выражениям.

### `src/feature_pipeline.py`

Объединяет все этапы обработки одной новости.

### `src/dataset_builder.py`

Обрабатывает список новостей и сохраняет итоговый аналитический датасет.

### `src/main.py`

Главная точка запуска проекта из терминала.

### `src/show_fake_analysis.py`

Демонстрационный скрипт.  
Показывает разбор одной фейковой новости:

- исходный текст;
- найденные сущности;
- эмоциональный профиль;
- признаки манипуляции.

## Установка проекта

### 1. Клонировать репозиторий

```powershell
git clone https://github.com/felix69420s/fake_news_analyzer.git
cd fake_news_analyzer
```

Если проект уже скачан вручную, достаточно перейти в папку проекта:

```powershell
cd C:\Users\katya\fake_news_analyzer
```

### 2. Создать виртуальное окружение

```powershell
python -m venv .venv
```

### 3. Активировать виртуальное окружение

```powershell
.venv\Scripts\Activate.ps1
```

После активации в терминале должно появиться:

```text
(.venv)
```

### 4. Установить зависимости

```powershell
pip install -r requirements.txt
```

Если `pip` не найден, можно использовать:

```powershell
python -m pip install -r requirements.txt
```

### 5. Проверить установку

```powershell
python -c "import pandas, transformers, datasets; print('OK')"
```

Если вывод:

```text
OK
```

значит зависимости установлены корректно.

## Получение датасетов

Датасеты не включены в репозиторий, потому что они могут занимать много места.  
Их нужно скачать отдельно в папку `data/raw/`.

## Получение основного датасета Baiangali

### Вариант 1. Скачать через PowerShell

Выполнить из корня проекта:

```powershell
New-Item -ItemType Directory -Force data\raw\baiangali_fake_news

Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/baiangali/fake_news/main/fake_real_news_annotated.json" `
  -OutFile "data\raw\baiangali_fake_news\fake_real_news_annotated.json"
```

После скачивания проверить файл:

```powershell
Get-Item data\raw\baiangali_fake_news\fake_real_news_annotated.json
```

Также можно проверить структуру:

```powershell
python -c "import pandas as pd; df=pd.read_json('data/raw/baiangali_fake_news/fake_real_news_annotated.json'); print(df.shape); print(df.columns.tolist())"
```

Ожидаемый результат:

```text
(204, 19)
```

### Вариант 2. Скачать через Git

```powershell
git clone https://github.com/baiangali/fake_news.git data\raw\baiangali_fake_news
```

После этого файл должен находиться по пути:

```text
data/raw/baiangali_fake_news/fake_real_news_annotated.json
```

## Получение дополнительного датасета Lenta

Датасет Lenta можно не скачивать вручную.  
При первом запуске он будет загружен через библиотеку `datasets` и сохранён в локальный кэш Hugging Face.

Запуск:

```powershell
python -m src.main --dataset lenta --limit 50 --output data/processed/lenta_50.csv
```

Если нужно скачать датасет Lenta явно в папку `data/raw/`, можно использовать Hugging Face CLI:

```powershell
pip install -U huggingface_hub
hf download data-silence/lenta.ru_2-extended --repo-type dataset --local-dir data/raw/lenta
```

## Авторизация Hugging Face

Для публичных датасетов и моделей авторизация обычно не обязательна.  
Но без токена Hugging Face может показывать предупреждение о неаутентифицированных запросах.

Для авторизации:

```powershell
pip install -U huggingface_hub
hf auth login
```

После этого нужно вставить токен Hugging Face с правами `Read`.

## Запуск проекта

## 1. Запуск основного датасета Baiangali

Перед запуском убедиться, что файл существует:

```text
data/raw/baiangali_fake_news/fake_real_news_annotated.json
```

Команда запуска:

```powershell
python -m src.main --dataset local --local-path data/raw/baiangali_fake_news/fake_real_news_annotated.json --local-adapter baiangali --limit 50 --output data/processed/baiangali_ru_50.csv
```

Ожидаемый вывод:

```text
Directories are ready.
Loading dataset source: local
Loaded DataFrame shape: (204, 19)
Source columns: id, annotations, file_upload, drafts, predictions, data, ...
Baiangali adapter: input_rows=204, created_records=104, skipped_empty_text=0, skipped_non_russian=100, records_without_label=0
Adapted records: 104
Model manager initialized on device=-1
Processing records: 100%
Analytical dataset saved to: data\processed\baiangali_ru_50.csv
Final DataFrame shape: (50, ...)
```

Важно:

```text
created_records=104
Adapted records=104
Final DataFrame shape=(50, ...)
```

Это означает, что:

- датасет прочитан;
- казахские тексты отфильтрованы;
- русские новости обработаны;
- итоговый CSV создан.

## 2. Запуск дополнительного датасета Lenta

```powershell
python -m src.main --dataset lenta --limit 50 --output data/processed/lenta_50.csv
```

Ожидаемый результат:

```text
Analytical dataset saved to: data\processed\lenta_50.csv
Final DataFrame shape: (50, ...)
```

## 3. Демонстрация анализа одной фейковой новости

После запуска Baiangali выполнить:

```powershell
python src/show_fake_analysis.py
```

Скрипт выводит:

1. исходный текст фейковой новости;
2. найденные персоны, организации и локации;
3. эмоциональный профиль;
4. признаки манипуляции;
5. краткий вывод.

Ожидаемая структура вывода:

```text
АНАЛИЗ ФЕЙКОВОЙ НОВОСТИ

1. ИСХОДНЫЕ ДАННЫЕ
2. ИЗВЛЕЧЁННЫЕ СУЩНОСТИ
3. ЭМОЦИОНАЛЬНЫЙ ПРОФИЛЬ
4. ЛИНГВИСТИЧЕСКИЕ ПРИЗНАКИ МАНИПУЛЯЦИИ
5. КРАТКИЙ ВЫВОД
```

Именно этот скрипт удобно использовать для демонстрации работы проекта преподавателю.

## Проверка результата

### Проверить, что CSV создан

```powershell
Get-Item data\processed\baiangali_ru_50.csv
```

### Посмотреть размер итогового датасета

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/baiangali_ru_50.csv'); print(df.shape)"
```

Ожидаемый результат:

```text
(50, ...)
```

### Проверить распределение меток

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/baiangali_ru_50.csv'); print(df['label'].value_counts(dropna=False))"
```

### Проверить эмоциональные признаки

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/baiangali_ru_50.csv'); print(df[['label','sentiment_label','sentiment_score','dominant_emotion','manipulation_score']].head().to_string())"
```

### Проверить, что казахские тексты отфильтрованы

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/processed/baiangali_ru_50.csv'); print('kazakh_rows=', df['text'].astype(str).str.contains('[әғқңөұүһіӘҒҚҢӨҰҮҺІ]', regex=True).sum())"
```

Ожидаемый результат:

```text
kazakh_rows= 0
```

Если значение больше нуля, это не всегда ошибка: в русском тексте могут встретиться казахские имена, топонимы или названия организаций.  
Основная цель фильтра — убрать целиком казахские тексты.

## Выходная структура данных

Итоговый датасет содержит следующие поля:

```text
id
date
source
source_type
url
author
language
label
title
lead
text
full_text
normalized_text
text_hash
char_count
token_count_approx
named_entities
persons
organizations
locations
geopolitical_entities
media_entities
persons_count
organizations_count
locations_count
geopolitical_count
media_count
sentiment_label
sentiment_score
emotion_labels
emotion_scores
dominant_emotion
manipulation_flags
manipulation_matches
manipulation_score
```

## Что означают основные поля

### `label`

Метка из исходного датасета:

```text
fake
real
```

Проект не создаёт эту метку самостоятельно.

### `named_entities`

Полный список найденных именованных сущностей.

### `persons`

Список найденных персон.

### `organizations`

Список найденных организаций.

### `locations`

Список найденных локаций.

### `geopolitical_entities`

Список геополитических объектов.

### `sentiment_label`

Метка тональности текста.

### `sentiment_score`

Оценка уверенности модели в тональности.

### `emotion_labels`

Список эмоций, возвращённых моделью.

### `emotion_scores`

Оценки эмоций.

### `dominant_emotion`

Эмоция с наибольшей оценкой.

### `manipulation_flags`

Словарь с признаками манипулятивности.

Пример:

```text
{
  "fear_appeal": true,
  "sensationalism": false,
  "vague_sources": true
}
```

### `manipulation_matches`

Конкретные слова и фразы, из-за которых сработали признаки.

### `manipulation_score`

Количество найденных категорий манипулятивности.

Например:

```text
0
```

означает, что признаки не найдены.

```text
3
```

означает, что найдено три категории признаков.

## Как показывать работу проекта

Для демонстрации достаточно выполнить три команды.

### 1. Запуск обработки основного датасета

```powershell
python -m src.main --dataset local --local-path data/raw/baiangali_fake_news/fake_real_news_annotated.json --local-adapter baiangali --limit 50 --output data/processed/baiangali_ru_50.csv
```

### 2. Демонстрация анализа одной фейковой новости

```powershell
python src/show_fake_analysis.py
```

### 3. Просмотр результата в CSV

Открыть файл:

```text
data/processed/baiangali_ru_50.csv
```

Показать колонки:

```text
label
persons
organizations
locations
geopolitical_entities
sentiment_label
sentiment_score
dominant_emotion
manipulation_flags
manipulation_matches
manipulation_score
```

## Как описывать результат в курсовой

Корректная формулировка:

```text
В результате был разработан программный модуль, который принимает на вход готовые датасеты медиатекстов, приводит их к единой структуре, выполняет очистку текста и извлекает интерпретируемые признаки: именованные сущности, эмоциональный профиль и rule-based маркеры манипулятивности.
```

Также важно указать:

```text
Проект не выполняет фактчекинг и не определяет достоверность новости. Метка fake/real берётся из исходного размеченного датасета и используется только для последующего анализа.
```

## Ограничения проекта

1. Проект не является системой фактчекинга.
2. Проект не предсказывает `fake/real`.
3. Метка `fake/real` берётся из исходного датасета.
4. Rule-based признаки манипуляции являются эвристическими.
5. Наличие манипулятивных признаков не доказывает фейковость текста.
6. Модели Hugging Face могут ошибаться при распознавании сущностей, эмоций и тональности.
7. Для основного датасета применяется фильтрация русскоязычных текстов, так как используемые модели ориентированы на русский язык.

## Работа с GitHub

В репозиторий загружаются:

```text
src/
README.md
requirements.txt
.gitignore
```

В репозиторий не загружаются:

```text
.venv/
.cursor/
.idea/
data/raw/
data/interim/
data/processed/
__pycache__/
```

Эти файлы исключаются через `.gitignore`.

## Полная последовательность запуска с нуля

```powershell
git clone https://github.com/felix69420s/fake_news_analyzer.git
cd fake_news_analyzer

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

New-Item -ItemType Directory -Force data\raw\baiangali_fake_news

Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/baiangali/fake_news/main/fake_real_news_annotated.json" `
  -OutFile "data\raw\baiangali_fake_news\fake_real_news_annotated.json"

python -m src.main --dataset local --local-path data/raw/baiangali_fake_news/fake_real_news_annotated.json --local-adapter baiangali --limit 50 --output data/processed/baiangali_ru_50.csv

python src/show_fake_analysis.py
```
