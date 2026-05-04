import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.feature_pipeline import process_record
from src.schemas import InputNewsRecord, OutputNewsRecord


def _model_dump(record: OutputNewsRecord) -> dict:
    if hasattr(record, "model_dump"):
        return record.model_dump()
    return record.dict()


def records_to_dataframe(records: list[OutputNewsRecord]) -> pd.DataFrame:
    rows: list[dict] = []
    for record in records:
        data = _model_dump(record)
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                data[key] = json.dumps(value, ensure_ascii=False)
        rows.append(data)
    return pd.DataFrame(rows)


def build_analytical_dataset(
    input_records: list[InputNewsRecord],
    model_manager,
    limit: int | None = None,
    manipulation_threshold: float = 0.55,
    manipulation_max_chars: int = 1500,
    include_evidence: bool = False,
) -> pd.DataFrame:
    records = input_records[:limit] if limit is not None else input_records
    processed: list[OutputNewsRecord] = []
    for idx, record in enumerate(tqdm(records, desc="Processing records"), start=1):
        try:
            processed_record = process_record(
                record,
                model_manager=model_manager,
                manipulation_threshold=manipulation_threshold,
                manipulation_max_chars=manipulation_max_chars,
                include_evidence=include_evidence,
            )
            processed.append(processed_record)
        except Exception as exc:
            record_id = getattr(record, "id", f"index_{idx - 1}")
            print(f"Warning: failed to process record '{record_id}': {exc}")
    return records_to_dataframe(processed)


def save_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    jsonl_path = output.with_suffix(".jsonl")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
