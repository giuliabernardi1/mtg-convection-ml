"""Entry point for the data download step: FCI (only the chunks covering the region) and LI."""
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import xarray as xr

from src.chunks import find_relevant_chunks
from src.download_data import download_entry, get_datastore, search, select_entries

log = logging.getLogger(__name__)

load_dotenv()

# --- Configuration ---
START = datetime(2026, 7, 18, 0, 0)
END = datetime(2026, 7, 23, 0, 0)    # 18-22 July
OUT_ROOT = Path("data/raw_chunks")
REGION = (6.6, 36.5, 18.8, 47.1)   # lon_min, lat_min, lon_max, lat_max; keep in sync with preprocessing
CHUNKS_WKT_PATH = Path("resources/FCI_chunks.wkt")

FCI_COLLECTION_ID = "EO:EUM:DAT:0662"
LI_COLLECTION_ID = "EO:EUM:DAT:0686"
FCI_STEP = 3  # keep one FCI image every 30 minutes
LI_STEP = 1   # keep every LI product: needed for the labels


def is_readable_netcdf(file_path: Path) -> bool:
    """A truncated download usually cannot be opened."""
    try:
        xr.open_dataset(file_path).close()
    except Exception:
        return False
    return True


def download_collection(datastore, collection_id, step, chunk_ids, area, out_dir) -> int:
    """Download the selected files of every `step`-th product; return the number of failures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    products = search(datastore, collection_id, START, END, area)
    selected_products = islice(products, 0, None, step)
    total_selected = len(range(0, products.total_results, step))

    failed_count = 0
    for product in tqdm(selected_products, total=total_selected, desc=collection_id, unit="product"):
        selected_names = select_entries(list(product.entries), chunk_ids)
        if not selected_names:
            log.warning("No matching files in %s", product)
            failed_count += 1
            continue
        for entry_name in selected_names:
            file_path = download_entry(product, entry_name, out_dir)
            if file_path is None:
                failed_count += 1
            elif not is_readable_netcdf(file_path):
                log.error("Corrupt or truncated file, deleted: %s", file_path.name)
                file_path.unlink()
                failed_count += 1
    return failed_count


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with logging_redirect_tqdm():
        datastore = get_datastore()
        area = ",".join(str(value) for value in REGION)

        fci_chunk_ids = find_relevant_chunks(CHUNKS_WKT_PATH, *REGION)
        log.info("FCI chunks to download: %s", fci_chunk_ids)

        failed_count = download_collection(
            datastore, FCI_COLLECTION_ID, FCI_STEP, fci_chunk_ids, area,
            OUT_ROOT / FCI_COLLECTION_ID.replace(":", "_"),
        )
        failed_count += download_collection(
            datastore, LI_COLLECTION_ID, LI_STEP, None, area,
            OUT_ROOT / LI_COLLECTION_ID.replace(":", "_"),
        )

    if failed_count:
        log.error("%d download(s) failed", failed_count)
        return 1
    log.info("All downloads completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())