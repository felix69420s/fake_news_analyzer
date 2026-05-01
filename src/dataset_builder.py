import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.feature_pipeline import process_record
from src.schemas import InputNewsRecord, OutputNewsRecord


def records_to_dataframe(records: list[OutputNewsRecord]) -> pd.DataFrame:
    rows: list[dict] = []
    for record in records:
        data = record.model_dump()
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                data[key] = json.dumps(value, ensure_ascii=False)
        rows.append(data)
    return pd.DataFrame(rows)


def build_analytical_dataset(
    input_records: list[InputNewsRecord], model_manager, limit: int | None = None
) -> pd.DataFrame:
    records = input_records[:limit] if limit is not None else input_records
    processed: list[OutputNewsRecord] = []

    for idx, record in enumerate(tqdm(records, desc="Processing records"), start=1):
        try:
            processed_record = process_record(record, model_manager)
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
