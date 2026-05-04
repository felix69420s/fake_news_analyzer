from pathlib import Path
import json
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet", ".xlsx"}


def load_local_file(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            for key in ["data", "records", "items", "examples"]:
                if isinstance(data.get(key), list):
                    df = pd.DataFrame(data[key])
                    break
            else:
                df = pd.json_normalize(data)
        else:
            df = pd.json_normalize(data)
    elif suffix == ".jsonl":
        df = pd.read_json(file_path, lines=True)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif suffix == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{suffix}'. Supported: {supported}.")

    if limit is not None:
        return df.head(limit).copy()
    return df


def load_kaggle_ru_dataset(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Loads the only supported source dataset.

    Supported layouts:
    1. A single prepared CSV file with columns Stance, Headline, articleBody.
    2. A directory containing train_bodies.csv and train_stances.csv.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {input_path}. "
            "Download the Kaggle dataset and pass --data-path to its file or folder."
        )

    if input_path.is_file():
        return load_local_file(input_path, limit=limit)

    bodies_path = input_path / "train_bodies.csv"
    stances_path = input_path / "train_stances.csv"
    if bodies_path.exists() and stances_path.exists():
        bodies = pd.read_csv(bodies_path)
        stances = pd.read_csv(stances_path)
        if "Body ID" not in bodies.columns or "Body ID" not in stances.columns:
            raise ValueError("Both train_bodies.csv and train_stances.csv must contain 'Body ID'.")
        df = stances.merge(bodies, on="Body ID", how="left", suffixes=("", "_body"))
        if limit is not None:
            return df.head(limit).copy()
        return df

    csv_files = sorted(input_path.glob("*.csv"))
    if len(csv_files) == 1:
        return load_local_file(csv_files[0], limit=limit)

    raise FileNotFoundError(
        "Could not find Kaggle dataset files. Expected either one CSV file or "
        "train_bodies.csv + train_stances.csv in the dataset directory."
    )


def preview_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns]
