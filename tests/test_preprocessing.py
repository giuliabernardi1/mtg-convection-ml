from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src import preprocessing as pp

FCI_NAME = (
    "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY---NC4E_C_EUMT_"
    "20260720004021_IDPFI_OPE_20260720003735_20260720003834_N__O_0004_0033.nc"
)
LI_NAME = (
    "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-AF--FD--CHK-BODY--ARC-NC4E_C_EUMT_"
    "20260720131032_L2PF_OPE_20260720130000_20260720131000_N__O_0079_0001.nc"
)


def make_fci_name(cycle_number: int, chunk_number: int = 33) -> str:
    return FCI_NAME.replace("_0004_0033.nc", f"_{cycle_number:04d}_{chunk_number:04d}.nc")


def make_li_name(cycle_number: int) -> str:
    return LI_NAME.replace("_0079_0001.nc", f"_{cycle_number:04d}_0001.nc")


class FakeScene:
    """Minimal stand-in for a satpy Scene: scene[name] returns a DataArray."""

    def __init__(self, data_arrays: dict):
        self._data_arrays = data_arrays

    def __getitem__(self, dataset_name):
        return self._data_arrays[dataset_name]


class FakeArea:
    """Minimal stand-in for a pyresample area definition."""

    def __init__(self, longitudes, latitudes):
        self._longitudes = longitudes
        self._latitudes = latitudes

    def get_lonlats(self):
        return self._longitudes, self._latitudes


# --- file names and cycles ---

@pytest.mark.parametrize(
    "file_name, expected_start",
    [
        (FCI_NAME, datetime(2026, 7, 20, 0, 30)),
        (LI_NAME, datetime(2026, 7, 20, 13, 0)),
        (make_li_name(1), datetime(2026, 7, 20, 0, 0)),
        (make_li_name(144), datetime(2026, 7, 20, 23, 50)),
    ],
)
def test_cycle_start_from_file_name(file_name, expected_start):
    assert pp.cycle_start(Path(file_name)) == expected_start


def test_cycle_start_returns_none_for_unrecognised_name():
    assert pp.cycle_start(Path("not_an_mtg_file.nc")) is None


def test_group_by_cycle_groups_chunks_of_the_same_cycle():
    chunk_paths = [Path(make_fci_name(4, chunk_number)) for chunk_number in (33, 34, 35, 36)]
    other_cycle_path = Path(make_fci_name(7, 33))

    groups = pp.group_by_cycle(chunk_paths + [other_cycle_path, Path("unknown.nc")])

    assert set(groups) == {datetime(2026, 7, 20, 0, 30), datetime(2026, 7, 20, 1, 0)}
    assert groups[datetime(2026, 7, 20, 0, 30)] == chunk_paths
    assert groups[datetime(2026, 7, 20, 1, 0)] == [other_cycle_path]


# --- window over the region of interest ---

def test_find_italy_window_selects_rows_and_columns_inside_the_box(monkeypatch):
    monkeypatch.setattr(pp, "AREA_EXTENT", (6.6, 36.5, 18.8, 47.1))
    grid_size = 10
    # longitude grows with the column (0, 3, ..., 27); latitude falls with the row (60, 57, ..., 33)
    longitudes = np.tile(np.arange(grid_size) * 3.0, (grid_size, 1))
    latitudes = np.tile((60.0 - np.arange(grid_size) * 3.0)[:, None], (1, grid_size))

    row_window, col_window, window_longitudes, window_latitudes = pp.find_italy_window(
        FakeArea(longitudes, latitudes)
    )

    assert (row_window.start, row_window.stop) == (5, 8)    # latitudes 45, 42, 39
    assert (col_window.start, col_window.stop) == (3, 7)    # longitudes 9, 12, 15, 18
    np.testing.assert_array_equal(window_longitudes[0], [9.0, 12.0, 15.0, 18.0])
    np.testing.assert_array_equal(window_latitudes[:, 0], [45.0, 42.0, 39.0])


# --- one cycle ---

def test_process_cycle_flips_fci_crops_both_products_and_fills_missing_flashes(monkeypatch):
    rows, cols = np.indices((6, 6))
    south_up_values = (rows * 10 + cols).astype("float64")  # FCI: row 0 is the southernmost
    flash_values = np.full((6, 6), np.nan)                   # LI: north-up, NaN = no flash
    flash_values[1, 2] = 0.5
    flash_values[2, 4] = 0.25
    fci_arrays = {
        "ir_105": xr.DataArray(south_up_values, dims=("y", "x")),
        "wv_63": xr.DataArray(south_up_values + 200.0, dims=("y", "x")),
    }
    li_arrays = {pp.LI_VAR: xr.DataArray(flash_values, dims=("y", "x"))}

    def fake_load_scene(input_files, reader_name, dataset_names):
        return FakeScene(fci_arrays if reader_name == "fci_l1c_nc" else li_arrays)

    monkeypatch.setattr(pp, "_load_scene", fake_load_scene)
    window_longitudes = np.arange(6.0).reshape(2, 3)
    window_latitudes = np.arange(6.0).reshape(2, 3) + 40.0
    window = (slice(1, 3), slice(2, 5), window_longitudes, window_latitudes)

    dataset = pp.process_cycle(["fci.nc"], "li.nc", window)

    assert dataset["ir_105"].dims == ("row", "col")
    # after the flip, window row 1 is original row 4 and window row 2 is original row 3
    np.testing.assert_array_equal(dataset["ir_105"].values, [[42, 43, 44], [32, 33, 34]])
    np.testing.assert_array_equal(dataset["wv_63"].values, [[242, 243, 244], [232, 233, 234]])
    # the LI is already north-up: no flip, NaN replaced with 0
    np.testing.assert_array_equal(dataset[pp.LI_VAR].values, [[0.5, 0.0, 0.0], [0.0, 0.0, 0.25]])
    for variable_name in ("ir_105", "wv_63", pp.LI_VAR):
        assert dataset[variable_name].dtype == np.float32
    assert dataset[pp.LI_VAR].attrs["units"] == "flashes/pixel"
    np.testing.assert_array_equal(dataset["longitude"].values, window_longitudes)
    np.testing.assert_array_equal(dataset["latitude"].values, window_latitudes)


# --- the whole loop ---

@pytest.fixture
def raw_dirs(tmp_path):
    fci_dir, li_dir = tmp_path / "fci", tmp_path / "li"
    for cycle_number in (1, 4, 7):
        cycle_dir = fci_dir / f"cycle_{cycle_number}"
        cycle_dir.mkdir(parents=True)
        for chunk_number in (33, 34):
            (cycle_dir / make_fci_name(cycle_number, chunk_number)).touch()
    trailer_name = make_fci_name(4, 41).replace("CHK-BODY---", "CHK-TRAIL---")
    (fci_dir / "cycle_4" / trailer_name).touch()  # must be ignored
    for cycle_number in (1, 4):                    # no LI for cycle 7
        cycle_dir = li_dir / f"cycle_{cycle_number}"
        cycle_dir.mkdir(parents=True)
        (cycle_dir / make_li_name(cycle_number)).touch()
    return fci_dir, li_dir


@pytest.fixture
def fake_pipeline(monkeypatch):
    state = {"processed_cycles": [], "window_computations": 0, "failing_cycles": set()}

    def fake_load_scene(input_files, reader_name, dataset_names):
        li_array = xr.DataArray(np.zeros((1, 1)), dims=("y", "x"), attrs={"area": "fake-area"})
        return FakeScene({pp.LI_VAR: li_array})

    def fake_find_italy_window(li_area):
        state["window_computations"] += 1
        return slice(0, 2), slice(0, 2), np.zeros((2, 2)), np.zeros((2, 2))

    def fake_process_cycle(fci_files, li_file, window):
        cycle_time = pp.cycle_start(Path(fci_files[0]))
        if cycle_time in state["failing_cycles"]:
            raise RuntimeError("simulated failure")
        state["processed_cycles"].append(
            (cycle_time, sorted(Path(path).name for path in fci_files), Path(li_file).name)
        )
        return xr.Dataset({"ir_105": (("row", "col"), np.zeros((2, 2), dtype="float32"))})

    monkeypatch.setattr(pp, "_load_scene", fake_load_scene)
    monkeypatch.setattr(pp, "find_italy_window", fake_find_italy_window)
    monkeypatch.setattr(pp, "process_cycle", fake_process_cycle)
    return state


def test_preprocess_skips_existing_outputs_and_cycles_without_li(raw_dirs, fake_pipeline, tmp_path):
    fci_dir, li_dir = raw_dirs
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    (output_dir / "20260720T0000.nc").write_bytes(b"")  # cycle 1 already done

    failed_count = pp.preprocess(fci_dir, li_dir, output_dir)

    assert failed_count == 0
    assert fake_pipeline["processed_cycles"] == [
        (
            datetime(2026, 7, 20, 0, 30),
            sorted([make_fci_name(4, 33), make_fci_name(4, 34)]),  # trailer excluded
            make_li_name(4),
        )
    ]
    assert (output_dir / "20260720T0030.nc").exists()
    assert not (output_dir / "20260720T0100.nc").exists()      # cycle 7 has no LI
    assert not list(output_dir.glob("*.part"))
    assert fake_pipeline["window_computations"] == 1


def test_preprocess_counts_failures_and_keeps_going(raw_dirs, fake_pipeline, tmp_path):
    fci_dir, li_dir = raw_dirs
    output_dir = tmp_path / "processed"
    fake_pipeline["failing_cycles"] = {datetime(2026, 7, 20, 0, 0)}

    failed_count = pp.preprocess(fci_dir, li_dir, output_dir)

    assert failed_count == 1
    assert not (output_dir / "20260720T0000.nc").exists()
    assert (output_dir / "20260720T0030.nc").exists()


def test_preprocess_returns_error_when_no_input_files(fake_pipeline, tmp_path):
    empty_fci_dir, empty_li_dir = tmp_path / "empty_fci", tmp_path / "empty_li"
    empty_fci_dir.mkdir()
    empty_li_dir.mkdir()

    assert pp.preprocess(empty_fci_dir, empty_li_dir, tmp_path / "processed") == 1
    assert fake_pipeline["processed_cycles"] == []