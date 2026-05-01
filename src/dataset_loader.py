from pathlib import Path

import json
import pandas as pd
from datasets import load_dataset


def load_hf_dataset(
    dataset_name: str, split: str = "train", limit: int | None = None
) -> pd.DataFrame:
    try:
        dataset = load_dataset(dataset_name, split=split)
        df = dataset.to_pandas()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Hugging Face dataset '{dataset_name}'."
        ) from exc

    if limit is not None:
        return df.head(limit).copy()
    return df


def load_local_file(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            records_key = None
            for key in ["data", "records", "items", "examples"]:
                value = data.get(key)
                if isinstance(value, list):
                    records_key = key
                    break

            if records_key is not None:
                df = pd.DataFrame(data[records_key])
            else:
                df = pd.json_normalize(data)
        else:
            df = pd.json_normalize(data)
    elif suffix == ".jsonl":
        df = pd.read_json(file_path, lines=True)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Supported: .csv, .json, .jsonl, .parquet."
        )

    if limit is not None:
        return df.head(limit).copy()
    return df


def preview_columns(df: pd.DataFrame) -> list[str]:
    return list(df.columns)
