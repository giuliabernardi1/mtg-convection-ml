import io
from datetime import datetime
from unittest.mock import MagicMock, patch

import eumdac
import pytest
import requests

from src import download_data as dd


class FakeSrc(io.BytesIO):
    """File-like object that mimics what product.open() yields."""

    def __init__(self, name, content=b"data"):
        super().__init__(content)
        self.name = name


def make_product(name="file.nc", content=b"data"):
    product = MagicMock()
    product.open.return_value = FakeSrc(name, content)
    return product


# --- get_datastore ---

def test_get_datastore_missing_credentials(monkeypatch):
    monkeypatch.delenv("CONSUMER_KEY_EUMETSAT", raising=False)
    monkeypatch.delenv("CONSUMER_SECRET_EUMETSAT", raising=False)
    with pytest.raises(ValueError):
        dd.get_datastore()


def test_get_datastore_builds_token_and_store(monkeypatch):
    monkeypatch.setenv("CONSUMER_KEY_EUMETSAT", "key")
    monkeypatch.setenv("CONSUMER_SECRET_EUMETSAT", "secret")
    with patch.object(dd.eumdac, "AccessToken") as token_cls, \
         patch.object(dd.eumdac, "DataStore") as store_cls:
        result = dd.get_datastore()
    token_cls.assert_called_once_with(("key", "secret"))
    store_cls.assert_called_once_with(token_cls.return_value)
    assert result is store_cls.return_value


# --- search ---

def test_search_passes_parameters_to_collection():
    datastore = MagicMock()
    collection = datastore.get_collection.return_value
    collection.search.return_value.total_results = 5
    start, end = datetime(2026, 7, 20, 0), datetime(2026, 7, 20, 1)

    result = dd.search(datastore, "COLL", start, end, "6.6,36.5,18.8,47.1")

    datastore.get_collection.assert_called_once_with("COLL")
    collection.search.assert_called_once_with(
        dtstart=start, dtend=end, sort="start,time,1", bbox="6.6,36.5,18.8,47.1"
    )
    assert result is collection.search.return_value


# --- download ---

def test_download_writes_file_and_leaves_no_part(tmp_path):
    path = dd.download(make_product("a.nc", b"hello"), tmp_path)
    assert path == tmp_path / "a.nc"
    assert path.read_bytes() == b"hello"
    assert not list(tmp_path.glob("*.part"))


def test_download_skips_existing_file(tmp_path):
    (tmp_path / "a.nc").write_bytes(b"old")
    path = dd.download(make_product("a.nc", b"new"), tmp_path)
    assert path == tmp_path / "a.nc"
    assert path.read_bytes() == b"old"


def test_download_returns_none_on_product_error(tmp_path):
    product = MagicMock()
    product.open.side_effect = eumdac.product.ProductError("boom")
    assert dd.download(product, tmp_path) is None


def test_download_returns_none_on_request_error(tmp_path):
    product = MagicMock()
    product.open.side_effect = requests.exceptions.ConnectionError("down")
    assert dd.download(product, tmp_path) is None