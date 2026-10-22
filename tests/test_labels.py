from datetime import datetime

import numpy as np
import xarray as xr

from src.labels import build_dataset

CELL_SIZE = 5
GRID_SIZE = 10  # 2 x 2 cells


def write_fake_cycle(directory, cycle_time, flash_cell=None, ir_value=250.0):
    longitudes, latitudes = np.meshgrid(
        np.linspace(10, 11, GRID_SIZE), np.linspace(41, 42, GRID_SIZE)
    )  # inside the region of interest
    flash_accumulation = np.zeros((GRID_SIZE, GRID_SIZE), dtype="float32")
    if flash_cell is not None:
        cell_row, cell_col = flash_cell
        flash_accumulation[
            cell_row * CELL_SIZE:(cell_row + 1) * CELL_SIZE,
            cell_col * CELL_SIZE:(cell_col + 1) * CELL_SIZE,
        ] = 0.5  # 25 pixels x 0.5 = 12.5 >= threshold
    xr.Dataset(
        {
            "ir_105": (("row", "col"), np.full((GRID_SIZE, GRID_SIZE), ir_value, dtype="float32")),
            "wv_63": (("row", "col"), np.full((GRID_SIZE, GRID_SIZE), 240.0, dtype="float32")),
            "flash_accumulation": (("row", "col"), flash_accumulation),
        },
        coords={
            "latitude": (("row", "col"), latitudes),
            "longitude": (("row", "col"), longitudes),
        },
    ).to_netcdf(directory / f"{cycle_time:%Y%m%dT%H%M}.nc")


def test_label_uses_future_cycle_and_persistence_uses_current_cycle(tmp_path):
    write_fake_cycle(tmp_path, datetime(2026, 7, 20, 12, 0), ir_value=250.0)
    write_fake_cycle(tmp_path, datetime(2026, 7, 20, 12, 30), flash_cell=(1, 0), ir_value=240.0)
    write_fake_cycle(tmp_path, datetime(2026, 7, 20, 13, 0), flash_cell=(0, 1))

    samples = build_dataset(tmp_path, lead_cycles=3, past_cycles=3, threshold=1.0, cell_size=CELL_SIZE)

    assert len(samples) == 4                                   # only 12:30 has past and target
    assert set(samples["time"]) == {datetime(2026, 7, 20, 12, 30)}
    positive_label_cells = samples.loc[samples["label"], ["cell_row", "cell_col"]].values.tolist()
    persistence_cells = samples.loc[samples["flash_now"], ["cell_row", "cell_col"]].values.tolist()
    assert positive_label_cells == [[0, 1]]                    # flash at 13:00
    assert persistence_cells == [[1, 0]]                       # flash at 12:30
    assert np.allclose(samples["ir_cooling"], -10.0)           # 240 K now vs 250 K thirty minutes earlier

def test_neighbourhood_feature_spreads_current_lightning_to_adjacent_cells(tmp_path):
    write_fake_cycle(tmp_path, datetime(2026, 7, 20, 12, 0))
    write_fake_cycle(tmp_path, datetime(2026, 7, 20, 12, 30), flash_cell=(1, 0))
    write_fake_cycle(tmp_path, datetime(2026, 7, 20, 13, 0))

    samples = build_dataset(tmp_path, lead_cycles=3, past_cycles=3, threshold=1.0, cell_size=CELL_SIZE)

    expected_accumulation = np.log1p(25 * 0.5)
    flash_cell_rows = samples[(samples["cell_row"] == 1) & (samples["cell_col"] == 0)]
    other_cell_rows = samples.drop(flash_cell_rows.index)
    assert np.isclose(flash_cell_rows["flash_sum_now"].iloc[0], expected_accumulation)
    assert (other_cell_rows["flash_sum_now"] == 0).all()
    assert np.allclose(samples["flash_sum_now_neighbourhood_max"], expected_accumulation)