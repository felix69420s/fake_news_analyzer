from transformers import pipeline

from src.config import EMOTION_MODEL_NAME, NER_MODEL_NAME, SENTIMENT_MODEL_NAME


class HFModelManager:
    def __init__(self, device: int = -1) -> None:
        self.device = device
        self._ner_pipeline = None
        self._emotion_pipeline = None
        self._sentiment_pipeline = None

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
                raise RuntimeError(
                    f"Failed to load NER model '{NER_MODEL_NAME}'."
                ) from exc
        return self._ner_pipeline

    def get_emotion_pipeline(self):
        if self._emotion_pipeline is None:
            try:
                self._emotion_pipeline = pipeline(
                    "text-classification",
                    model=EMOTION_MODEL_NAME,
                    tokenizer=EMOTION_MODEL_NAME,
                    top_k=None,
                    device=self.device,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load emotion model '{EMOTION_MODEL_NAME}'."
                ) from exc
        return self._emotion_pipeline

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
                raise RuntimeError(
                    f"Failed to load sentiment model '{SENTIMENT_MODEL_NAME}'."
                ) from exc
        return self._sentiment_pipeline

    @staticmethod
    def truncate_text_for_model(text: str, max_chars: int = 3000) -> str:
        return text[:max_chars]
