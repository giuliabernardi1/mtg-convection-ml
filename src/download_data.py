import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import eumdac
import requests

log = logging.getLogger(__name__)


def get_datastore() -> eumdac.DataStore:
    """Get a DataStore using EUMETSAT credentials from environment variables.
    Requires CONSUMER_KEY_EUMETSAT and CONSUMER_SECRET_EUMETSAT to be set in the environment.
    :returns: configured eumdac.DataStore"""
    consumer_key = os.getenv("CONSUMER_KEY_EUMETSAT")
    consumer_secret = os.getenv("CONSUMER_SECRET_EUMETSAT")
    if not consumer_key or not consumer_secret:
        raise ValueError(
            "Set CONSUMER_KEY_EUMETSAT and CONSUMER_SECRET_EUMETSAT."
        )

    token = eumdac.AccessToken((consumer_key, consumer_secret))
    log.info("Token expires %s", token.expiration)  # fails fast on bad credentials
    return eumdac.DataStore(token)


def search(
    datastore: eumdac.DataStore,
    collection_id: str,
    start: datetime,
    end: datetime,
    area: str,
) -> eumdac.collection.SearchResults:
    """
    Search for products in a collection within a time range and area.

    :param datastore: The configured EUMETSAT DataStore instance to use for the search.
    :param collection_id: The ID of the collection to search within.
    :param start: The start datetime for the search range.
    :param end: The end datetime for the search range.
    :param area: The bounding box area for the search in the format "min_lon,min_lat,max_lon,max_lat".
    :return: The search results containing the found products.
    """
    collection = datastore.get_collection(collection_id)
    products = collection.search(
        dtstart=start, dtend=end, sort="start,time,1", bbox=area
    )
    log.info("Found %d products in %s", products.total_results, collection_id)
    return products


def download(product, out_dir: Path) -> Path | None:
    """
    Download a product to the specified output directory.
    :param product: The product to download, an instance of eumdac.product.Product.
    :param out_dir: The output directory where the product will be saved, as a Path object.
    :return: The path to the downloaded product file, or None if the download failed.
    """
    try:
        with product.open() as src:
            target = out_dir / src.name
            if target.exists():
                log.info("Skipped %s: already exists", target.name)
                return target
            tmp = target.with_name(target.name + ".part")
            with open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
        tmp.rename(target)
        return target
    except eumdac.product.ProductError as error:
        log.error("Product %s: %s", product, error)
    except requests.exceptions.RequestException as error:
        log.error("Download of %s failed: %s", product, error)
    return None

