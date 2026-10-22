import logging
import sys
from pathlib import Path

from src.labels import build_dataset

PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "data" / "samples.parquet"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples = build_dataset(PROCESSED_DIR)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())