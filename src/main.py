import argparse
from pathlib import Path

from src.config import (
    DEFAULT_LIMIT,
    DEFAULT_OUTPUT_FILE,
    LENTA_DATASET_NAME,
    MAIN_DATASET_NAME,
    ensure_dirs,
)
from src.dataset_adapters import adapt_dataset
from src.dataset_builder import build_analytical_dataset, save_dataset
from src.dataset_loader import load_kaggle_ru_dataset, preview_columns
from src.hf_models import HFModelManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build analytical dataset for fake_news_analyzer using Kaggle fake-real-news-ru."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(KAGGLE_DATASET_DIR),
        help=(
            "Path to the Kaggle dataset CSV file or to a folder with "
            "train_bodies.csv and train_stances.csv."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max number of records to process (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device for HF pipelines: -1 CPU, 0 GPU.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ensure_dirs()
    print("Directories are ready.")
    print(f"Loading Kaggle fake-real-news-ru dataset from: {args.data_path}")
    # Limit is applied after adaptation in order not to hide label or text problems.
    df = load_kaggle_ru_dataset(args.data_path, limit=None)

    columns = preview_columns(df)
    print(f"Loaded DataFrame shape: {df.shape}")
    print("Source columns:", ", ".join(columns) if columns else "(no columns)")

    input_records = adapt_dataset(df)
    print(f"Adapted records: {len(input_records)}")
    if not input_records:
        raise RuntimeError("No records were adapted. Check Stance and article body columns.")

    model_manager = HFModelManager(device=args.device)
    print(f"Model manager initialized on device={args.device}")

    analytical_df = build_analytical_dataset(
        input_records=input_records,
        model_manager=model_manager,
        limit=args.limit,
    )

    output_path = Path(args.output)
    save_dataset(analytical_df, output_path)
    print(f"Analytical dataset saved to: {output_path}")
    print(f"Final DataFrame shape: {analytical_df.shape}")


if __name__ == "__main__":
    main()
