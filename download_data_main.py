import logging
import sys

from datetime import datetime
from dotenv import load_dotenv
from itertools import islice
from pathlib import Path

from src.download_data import download, get_datastore, search

log = logging.getLogger(__name__)

load_dotenv()

START = datetime(2026, 7, 20, 12, 0)
END = datetime(2026, 7, 20, 13, 0)
AREA = '6.0, 36.5, 19.0, 47.5'        # lon_min,lat_min,lon_max,lat_max
OUT_ROOT = Path("data/raw")

COLLECTIONS = ["EO:EUM:DAT:0662", "EO:EUM:DAT:0686"]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    datastore = get_datastore()
    failed = 0

    for collection_id in COLLECTIONS:
        out_dir = OUT_ROOT / collection_id.replace(":", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        products = search(datastore, collection_id, START, END, AREA)
        for product in islice(products, 0, None):
            if download(product, out_dir) is None:
                failed += 1

    if failed:
        log.error("%d download(s) failed", failed)
        return 1
    log.info("All downloads completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())