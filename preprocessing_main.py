import logging
import sys
from pathlib import Path

from src.preprocessing import preprocess


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    failed_count = preprocess(
    fci_dir=Path("data/raw_chunks_test/EO_EUM_DAT_0662"),
    li_dir=Path("data/raw_chunks_test/EO_EUM_DAT_0686"),
    out_dir=Path("data/processed_test"),
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    sys.exit(main())