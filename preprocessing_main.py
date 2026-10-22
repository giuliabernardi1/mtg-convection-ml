import logging
import sys
from pathlib import Path

from src.preprocessing import preprocess

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    failed_count = preprocess(
        fci_dir=PROJECT_ROOT / "data" / "raw_chunks" / "EO_EUM_DAT_0662",
        li_dir=PROJECT_ROOT / "data" / "raw_chunks" / "EO_EUM_DAT_0686",
        out_dir=PROJECT_ROOT / "data" / "processed",
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    sys.exit(main())