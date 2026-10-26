import logging
import sys

from src.config import CONFIG, project_path
from src.preprocessing import preprocess


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_dir = project_path(CONFIG["paths"]["raw_dir"])
    failed_count = preprocess(
        fci_dir=raw_dir / CONFIG["download"]["fci_collection_id"].replace(":", "_"),
        li_dir=raw_dir / CONFIG["download"]["li_collection_id"].replace(":", "_"),
        out_dir=project_path(CONFIG["paths"]["processed_dir"]),
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    sys.exit(main())