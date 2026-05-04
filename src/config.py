from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

KAGGLE_DATASET_DIR = RAW_DATA_DIR / "fake_real_news_ru"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/morfifinka/fake-real-news-ru"

NER_MODEL_NAME = "r1char9/ner-rubert-tiny-news"
SENTIMENT_MODEL_NAME = "blanchefort/rubert-base-cased-sentiment-rusentiment"
MANIPULATION_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

DEFAULT_LANGUAGE = "ru"
DEFAULT_OUTPUT_FILE = PROCESSED_DATA_DIR / "kaggle_ru_analytical_dataset.csv"
DEFAULT_LIMIT = 50
DEFAULT_MANIPULATION_THRESHOLD = 0.70
DEFAULT_MANIPULATION_MAX_CHARS = 1500


def ensure_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
