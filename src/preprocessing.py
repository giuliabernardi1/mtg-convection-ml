"""Put FCI L1c and LI L2 AF products on the native 2 km grid, cropped over Italy, one NetCDF per cycle.

No resampling: the LI grid is the FCI grid flipped vertically, so the FCI is flipped to match.
All outputs are north-up.
"""
import logging
import re
import warnings

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from satpy import Scene

from src.config import CONFIG

log = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
)

AREA_EXTENT = tuple(CONFIG["region"]["extent"])  # lon_min, lat_min, lon_max, lat_max
CHANNELS = CONFIG["preprocessing"]["channels"]
LI_VAR = CONFIG["preprocessing"]["li_variable"]

# ...<start 14 digits>_<end 14 digits>_N__O_<cycle of day>_<chunk>.nc
_NAME_RE = re.compile(r"_(\d{8})\d{6}_\d{14}_N__\w_(\d{4})_\d{4}\.nc$")

def cycle_start(path: Path) -> datetime | None:
    """Start time of the 10-min repeat cycle, from the file name (cycle n starts at (n-1)*10 min)."""
    m = _NAME_RE.search(path.name)
    if not m:
        return None
    day = datetime.strptime(m.group(1), "%Y%m%d")
    return day + timedelta(minutes=(int(m.group(2)) - 1) * 10)


def group_by_cycle(files) -> dict[datetime, list[Path]]:
    groups = defaultdict(list)
    for f in files:
        t = cycle_start(f)
        if t is None:
            log.warning("Unrecognised file name, skipped: %s", f.name)
            continue
        groups[t].append(f)
    return groups

def _load_scene(input_files, reader_name, dataset_names) -> Scene:
    scene = Scene(filenames=[str(path) for path in input_files], reader=reader_name)
    scene.load(dataset_names)
    return scene


def find_italy_window(li_area):
    """Pixel window covering the bounding box, plus the longitude/latitude of its pixels."""
    longitudes, latitudes = li_area.get_lonlats()
    lon_min, lat_min, lon_max, lat_max = AREA_EXTENT
    inside_box = (
        (longitudes >= lon_min) & (longitudes <= lon_max)
        & (latitudes >= lat_min) & (latitudes <= lat_max)
    )
    row_indices = np.flatnonzero(inside_box.any(axis=1))
    col_indices = np.flatnonzero(inside_box.any(axis=0))
    row_window = slice(row_indices.min(), row_indices.max() + 1)
    col_window = slice(col_indices.min(), col_indices.max() + 1)
    return row_window, col_window, longitudes[row_window, col_window], latitudes[row_window, col_window]


def process_cycle(fci_files, li_file, window) -> xr.Dataset:
    row_window, col_window, window_longitudes, window_latitudes = window
    fci_scene = _load_scene(fci_files, "fci_l1c_nc", CHANNELS)
    li_scene = _load_scene([li_file], "li_l2_nc", [LI_VAR])

    data_vars = {}
    for channel in CHANNELS:
        north_up_channel = fci_scene[channel].isel(y=slice(None, None, -1))  # FCI rows are south-up
        window_values = north_up_channel.isel(y=row_window, x=col_window).values
        data_vars[channel] = (("row", "col"), window_values.astype("float32"))

    flash_window = li_scene[LI_VAR].isel(y=row_window, x=col_window).values
    data_vars["flash_accumulation"] = (
        ("row", "col"),
        np.nan_to_num(flash_window, nan=0.0).astype("float32"),  # sparse product: missing = no flash
        {"units": "flashes/pixel"},
    )
    return xr.Dataset(
        data_vars,
        coords={
            "latitude": (("row", "col"), window_latitudes),
            "longitude": (("row", "col"), window_longitudes),
        },
    )


def preprocess(fci_dir: Path, li_dir: Path, out_dir: Path) -> int:
    """Return the number of failed cycles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fci_cycles = group_by_cycle(sorted(fci_dir.rglob("*CHK-BODY*.nc")))
    li_cycles = group_by_cycle(sorted(li_dir.rglob("*CHK-BODY*.nc")))
    log.info("FCI cycles: %d | LI cycles: %d | in common: %d",
             len(fci_cycles), len(li_cycles), len(set(fci_cycles) & set(li_cycles)))
    if not fci_cycles or not li_cycles:
        log.error("No input files found in %s / %s (are the zips extracted?)", fci_dir, li_dir)
        return 1

    window = None  # computed once, from the first cycle
    failed_cycles = 0
    for cycle_time in sorted(fci_cycles):
        output_path = out_dir / f"{cycle_time:%Y%m%dT%H%M}.nc"
        if output_path.exists():
            log.info("Skipped %s: already exists", output_path.name)
            continue
        if cycle_time not in li_cycles:
            log.warning("No LI file for cycle %s, skipped", cycle_time)
            continue
        try:
            li_file = li_cycles[cycle_time][0]
            if window is None:
                li_area = _load_scene([li_file], "li_l2_nc", [LI_VAR])[LI_VAR].attrs["area"]
                window = find_italy_window(li_area)
                log.info("Window rows %s, cols %s", window[0], window[1])
            dataset = process_cycle(fci_cycles[cycle_time], li_file, window)
            temporary_path = output_path.with_suffix(".part")
            dataset.to_netcdf(temporary_path)
            temporary_path.rename(output_path)
            log.info("Written %s", output_path.name)
        except Exception:
            log.exception("Cycle %s failed", cycle_time)
            failed_cycles += 1
    return failed_cycles