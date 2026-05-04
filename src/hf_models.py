from transformers import pipeline

from src.config import (
    MANIPULATION_MODEL_NAME,
    NER_MODEL_NAME,
    SENTIMENT_MODEL_NAME,
)


class HFModelManager:
    def __init__(self, device: int = -1) -> None:
        self.device = device
        self._ner_pipeline = None
        self._sentiment_pipeline = None
        self._manipulation_pipeline = None

    def get_ner_pipeline(self):
        if self._ner_pipeline is None:
            try:
                self._ner_pipeline = pipeline(
                    "ner",
                    model=NER_MODEL_NAME,
                    tokenizer=NER_MODEL_NAME,
                    aggregation_strategy="simple",
                    device=self.device,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to load NER model '{NER_MODEL_NAME}'.") from exc
        return self._ner_pipeline

    def get_sentiment_pipeline(self):
        if self._sentiment_pipeline is None:
            try:
                self._sentiment_pipeline = pipeline(
                    "text-classification",
                    model=SENTIMENT_MODEL_NAME,
                    tokenizer=SENTIMENT_MODEL_NAME,
                    top_k=None,
                    device=self.device,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to load sentiment model '{SENTIMENT_MODEL_NAME}'.") from exc
        return self._sentiment_pipeline

    def get_manipulation_pipeline(self):
        if self._manipulation_pipeline is None:
            try:
                self._manipulation_pipeline = pipeline(
                    "zero-shot-classification",
                    model=MANIPULATION_MODEL_NAME,
                    tokenizer=MANIPULATION_MODEL_NAME,
                    device=self.device,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load manipulation model '{MANIPULATION_MODEL_NAME}'."
                ) from exc
        return self._manipulation_pipeline

    @staticmethod
    def truncate_text_for_model(text: str, max_chars: int = 3000) -> str:
        return (text or "")[:max_chars]
