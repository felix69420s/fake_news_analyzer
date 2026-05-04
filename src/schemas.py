from typing import Any
from pydantic import BaseModel, Field
from typing import Any

class InputNewsRecord(BaseModel):
    id: str
    title: str | None = ""
    lead: str | None = ""
    text: str | None = ""
    date: str | None = ""
    source: str | None = ""
    source_type: str | None = ""
    url: str | None = ""
    author: str | None = ""
    language: str | None = "ru"
    label: str | None = ""


class OutputNewsRecord(BaseModel):
    id: str
    date: str | None = ""
    source: str | None = ""
    source_type: str | None = ""
    url: str | None = ""
    author: str | None = ""
    language: str | None = "ru"
    label: str | None = ""

    title: str | None = ""
    lead: str | None = ""
    text: str | None = ""
    full_text: str | None = ""
    normalized_text: str | None = ""
    text_hash: str | None = ""
    char_count: int = 0
    token_count_approx: int = 0

    named_entities: list[dict[str, Any]] = Field(default_factory=list)
    persons: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    geopolitical_entities: list[str] = Field(default_factory=list)
    media_entities: list[str] = Field(default_factory=list)
    persons_count: int = 0
    organizations_count: int = 0
    locations_count: int = 0
    geopolitical_count: int = 0
    media_count: int = 0

    sentiment_label: str | None = ""
    sentiment_score: float = 0.0

    # Модельные признаки манипулятивной подачи.
    manipulation_flags: dict[str, bool] = Field(default_factory=dict)
    manipulation_scores: dict[str, float] = Field(default_factory=dict)
    manipulation_score: float = 0.0
    manipulation_method: str | None = ""
    manipulation_model: str | None = ""
    manipulation_threshold: float = 0.0
    manipulation_evidence_sentences: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

