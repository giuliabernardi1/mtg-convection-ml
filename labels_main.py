import logging
import sys

from src.config import CONFIG, project_path
from src.labels import build_dataset

PROCESSED_DIR = project_path(CONFIG["paths"]["processed_dir"])
OUTPUT_PATH = project_path(CONFIG["paths"]["samples"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples = build_dataset(PROCESSED_DIR)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())