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
from src.dataset_loader import load_hf_dataset, load_local_file, preview_columns
from src.hf_models import HFModelManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build analytical dataset for fake_news_analyzer."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["baiangali", "lenta", "local"],
        help="Dataset source: baiangali, lenta, or local.",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to local file (.csv/.json/.jsonl/.parquet). Required for --dataset local.",
    )
    parser.add_argument(
        "--local-adapter",
        type=str,
        choices=["baiangali", "lenta"],
        default=None,
        help="Adapter for local dataset. Required for --dataset local.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Hugging Face split to load (default: train).",
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

    if args.dataset == "local":
        if not args.local_path:
            parser.error("--local-path is required when --dataset local.")
        if not args.local_adapter:
            parser.error("--local-adapter is required when --dataset local.")

    ensure_dirs()
    print("Directories are ready.")

    print(f"Loading dataset source: {args.dataset}")
    if args.dataset == "baiangali":
        df = load_hf_dataset(MAIN_DATASET_NAME, split=args.split)
        adapter_key = "baiangali"
    elif args.dataset == "lenta":
        df = load_hf_dataset(LENTA_DATASET_NAME, split=args.split)
        adapter_key = "lenta"
    else:
        df = load_local_file(args.local_path)
        adapter_key = args.local_adapter

    columns = preview_columns(df)
    print(f"Loaded DataFrame shape: {df.shape}")
    print("Source columns:", ", ".join(columns) if columns else "(no columns)")
    # ВАЖНО:
    # До адаптации можно ограничивать только полностью русскоязычные большие датасеты.
    # Для Baiangali так делать нельзя, потому что первые строки могут быть казахскими,
    # и языковой фильтр удалит их все.
    pre_limit_allowed = (
        args.dataset == "lenta"
        or (args.dataset == "local" and args.local_adapter == "lenta")
    )

    if args.limit and pre_limit_allowed:
        df = df.head(args.limit)
        print(f"Applied pre-adaptation limit: {args.limit}. New shape: {df.shape}")

    input_records = adapt_dataset(df, adapter_key)
    print(f"Adapted records: {len(input_records)}")

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
