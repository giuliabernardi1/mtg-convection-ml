"""All plots: reliability diagrams, a case-study forecast map, lightning frequency map, diurnal cycle.

Usage: python plot_main.py [--time 2026-07-15T15:00] [--only reliability maps]
Without --time, the cycle with the most lightning cells 30 min later is used as case study.
The case-study model is the gradient boosting model with all features, trained without the weather
episode (block of days) of the chosen time, so the forecast is out-of-sample. Coastlines and borders come
from Natural Earth and are downloaded by cartopy the first time (internet access needed).
Reliability diagrams are built from results/*_reliability.csv.
"""
import argparse
import logging
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, PowerNorm
from matplotlib.patches import Patch

from src.config import CONFIG, project_path
from src.preprocessing import AREA_EXTENT
from src.train import assign_blocks, fit_model
from train_main import FEATURE_SETS

log = logging.getLogger(__name__)

SAMPLES_PATH = project_path(CONFIG["paths"]["samples"])
RESULTS_DIR = project_path(CONFIG["paths"]["results_dir"])
FEATURE_SET = "all_features"
DEFAULT_THRESHOLD = 0.25  # used only if results/all_features.csv is missing
PROJECTION = ccrs.PlateCarree()

RELIABILITY_PATH = RESULTS_DIR / "reliability.png"
RELIABILITY_SUFFIX = "_reliability.csv"
MIN_COUNT = 30  # bins with fewer samples are too noisy: drawn faded


# ---------------------------------------------------------------- reliability

def plot_reliability(tables: dict[str, pd.DataFrame], output_path) -> None:
    figure, (diagram_axis, histogram_axis) = plt.subplots(
        2, 1, figsize=(7, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    diagram_axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect reliability")

    for name, table in tables.items():
        reliable = table["count"] >= MIN_COUNT
        line, = diagram_axis.plot(
            table.loc[reliable, "mean_forecast"], table.loc[reliable, "observed_frequency"],
            marker="o", label=name,
        )
        diagram_axis.plot(
            table.loc[~reliable, "mean_forecast"], table.loc[~reliable, "observed_frequency"],
            marker="o", linestyle="none", color=line.get_color(), alpha=0.3,
        )
        histogram_axis.plot(table["mean_forecast"], table["count"], marker="o", color=line.get_color())

    diagram_axis.set_ylabel("Observed frequency")
    diagram_axis.set_title(f"Reliability diagram (faded points: fewer than {MIN_COUNT} samples)")
    diagram_axis.set_xlim(0, 1)
    diagram_axis.set_ylim(0, 1)
    diagram_axis.grid(alpha=0.3)
    diagram_axis.legend(fontsize=8)

    histogram_axis.set_yscale("log")
    histogram_axis.set_xlabel("Forecast probability")
    histogram_axis.set_ylabel("Samples per bin")
    histogram_axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_reliability() -> bool:
    paths = sorted(RESULTS_DIR.glob(f"*{RELIABILITY_SUFFIX}"))
    if not paths:
        log.error("No *%s files in %s: run train_main.py / cnn_main.py first", RELIABILITY_SUFFIX, RESULTS_DIR)
        return False
    tables = {path.name.removesuffix(RELIABILITY_SUFFIX): pd.read_csv(path) for path in paths}
    plot_reliability(tables, RELIABILITY_PATH)
    log.info("Written %s", RELIABILITY_PATH)
    return True


# ----------------------------------------------------------------------- maps

def cell_grid_shape(samples: pd.DataFrame) -> tuple[int, int]:
    return int(samples["cell_row"].max()) + 1, int(samples["cell_col"].max()) + 1


def to_grid(frame: pd.DataFrame, column: str, shape: tuple[int, int]) -> np.ndarray:
    """Put one value per cell on the (cell_row, cell_col) grid; cells without a sample are NaN."""
    grid = np.full(shape, np.nan)
    grid[frame["cell_row"].to_numpy(), frame["cell_col"].to_numpy()] = frame[column].to_numpy(dtype="float64")
    return grid


def coordinate_grids(samples: pd.DataFrame, shape: tuple[int, int]):
    """Cell-centre longitude and latitude on the full grid; cells never sampled (region edges) are
    filled with a plane fitted on the known cells, so that the map has no holes."""
    per_cell = samples.groupby(["cell_row", "cell_col"], as_index=False)[["latitude", "longitude"]].mean()
    rows, cols = np.indices(shape)
    design = np.column_stack([np.ones(rows.size), rows.ravel(), cols.ravel()])
    grids = []
    for column in ("longitude", "latitude"):
        grid = to_grid(per_cell, column, shape)
        known = ~np.isnan(grid.ravel())
        coefficients, *_ = np.linalg.lstsq(design[known], grid.ravel()[known], rcond=None)
        grids.append(np.where(np.isnan(grid), (design @ coefficients).reshape(shape), grid))
    return grids


def new_map_axis(figure, position):
    axis = figure.add_subplot(*position, projection=PROJECTION)
    lon_min, lat_min, lon_max, lat_max = AREA_EXTENT
    axis.set_extent([lon_min, lon_max, lat_min, lat_max], crs=PROJECTION)
    axis.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, linestyle=":")
    axis.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5)
    return axis


def threshold_for_block(block: int) -> float:
    results_path = RESULTS_DIR / f"{FEATURE_SET}.csv"
    if not results_path.exists():
        log.warning("%s not found: using threshold %.2f", results_path, DEFAULT_THRESHOLD)
        return DEFAULT_THRESHOLD
    results = pd.read_csv(results_path)
    return float(results.loc[results["test_block"] == block, "threshold"].iloc[0])


def _csi(observed, predicted) -> float:
    observed, predicted = np.asarray(observed, bool), np.asarray(predicted, bool)
    hits = (observed & predicted).sum()
    total = (observed | predicted).sum()
    return float(hits / total) if total else float("nan")


def plot_case_study(samples, longitudes, latitudes, shape, case_time: pd.Timestamp, output_path) -> None:
    samples = samples.assign(block=assign_blocks(samples["day"]))
    frame = samples[samples["time"] == case_time]
    if frame.empty:
        raise ValueError(f"No samples at {case_time}; available times start at {samples['time'].min()}")
    block = int(frame["block"].iloc[0])
    feature_columns = FEATURE_SETS[FEATURE_SET]
    log.info("Training %s without block %d to forecast %s", FEATURE_SET, block, case_time)
    model = fit_model(samples[samples["block"] != block], feature_columns)
    frame = frame.assign(probability=model.predict_proba(frame[feature_columns])[:, 1])
    threshold = threshold_for_block(block)

    forecast = frame["probability"] >= threshold
    outcome = np.select(
        [frame["label"] & forecast, frame["label"] & ~forecast, ~frame["label"] & forecast], [1, 2, 3], default=0
    )
    frame = frame.assign(outcome=outcome)
    outcome_grid = np.ma.masked_equal(to_grid(frame, "outcome", shape), 0)
    outcome_grid = np.ma.masked_invalid(outcome_grid)

    figure = plt.figure(figsize=(18, 6.5))
    axis = new_map_axis(figure, (1, 3, 1))
    image = axis.pcolormesh(longitudes, latitudes, np.ma.masked_invalid(to_grid(frame, "ir_min", shape)),
                            cmap="Greys_r", vmin=200, vmax=300, shading="nearest", transform=PROJECTION)
    figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.04, shrink=0.8, label="Coldest IR 10.5 µm brightness temperature in cell (K)")
    flashing = frame[frame["flash_now"]]
    axis.scatter(flashing["longitude"], flashing["latitude"], s=6, c="red", marker="s", transform=PROJECTION)
    axis.set_title(f"Input at t = {case_time:%Y-%m-%d %H:%M} UTC\nred: cells with lightning at t")

    axis = new_map_axis(figure, (1, 3, 2))
    image = axis.pcolormesh(longitudes, latitudes, np.ma.masked_invalid(to_grid(frame, "probability", shape)),
                            cmap="viridis", vmin=0, vmax=1, shading="nearest", transform=PROJECTION)
    figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.04, shrink=0.8, label="Forecast probability of lightning at t + 30 min")
    axis.set_title("Gradient boosting forecast (out-of-sample)")

    axis = new_map_axis(figure, (1, 3, 3))
    colours = ["#2ca25f", "#e6550d", "#3182bd"]
    axis.pcolormesh(longitudes, latitudes, outcome_grid, cmap=ListedColormap(colours), vmin=0.5, vmax=3.5,
                    shading="nearest", transform=PROJECTION)
    axis.legend(
        handles=[Patch(color=colour, label=label)
                 for colour, label in zip(colours, ["hit", "miss", "false alarm"])],
        loc="lower left", fontsize=9,
    )
    persistence_csi = _csi(frame["label"], frame["flash_now"])
    axis.set_title(f"Outcome at t + 30 min (threshold {threshold:.2f})\n"
                   f"CSI {_csi(frame['label'], forecast):.2f} | persistence CSI {persistence_csi:.2f}")

    figure.suptitle("Demonstration only: a single case, not representative of the model's average skill", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_frequency_map(samples, longitudes, latitudes, shape, output_path) -> None:
    frequency = samples.groupby(["cell_row", "cell_col"], as_index=False)["label"].mean()
    figure = plt.figure(figsize=(8, 8))
    axis = new_map_axis(figure, (1, 1, 1))
    image = axis.pcolormesh(longitudes, latitudes, np.ma.masked_invalid(to_grid(frequency, "label", shape)),
                            cmap="YlOrRd", norm=PowerNorm(0.5), shading="nearest", transform=PROJECTION)
    figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.04, shrink=0.8,
                    label="Fraction of cycles with lightning in the cell")
    n_days = samples["day"].nunique()
    axis.set_title(f"Lightning frequency over the {n_days} selected days\n"
                   "(days chosen for their activity: not a climatology)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_diurnal_cycle(samples, output_path) -> None:
    hourly = (samples.assign(hour=samples["time"].dt.hour + samples["time"].dt.minute / 60)
              .groupby(["day", "hour"])["label"].mean().unstack("day"))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(hourly.index, 100 * hourly, color="grey", alpha=0.4, linewidth=1)
    axis.plot(hourly.index, 100 * hourly.mean(axis=1), color="red", linewidth=2.5, label="mean over days")
    axis.set_xlabel("Hour (UTC) of the forecast time t (label refers to t + 30 min)")
    axis.set_ylabel("Cells with lightning (%)")
    axis.set_title("Diurnal cycle of lightning cells (grey: single days)")
    axis.set_xlim(0, 24)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_maps(case_time_arg: str | None) -> None:
    samples = pd.read_parquet(SAMPLES_PATH)
    samples["current_lightning"] = samples["flash_now"].astype("float32")
    shape = cell_grid_shape(samples)
    longitudes, latitudes = coordinate_grids(samples, shape)

    if case_time_arg:
        case_time = pd.Timestamp(case_time_arg)
    else:
        case_time = samples.groupby("time")["label"].sum().idxmax()

    case_path = RESULTS_DIR / f"map_case_study_{case_time:%Y%m%dT%H%M}.png"
    plot_case_study(samples, longitudes, latitudes, shape, case_time, case_path)
    log.info("Written %s", case_path)
    frequency_path = RESULTS_DIR / "map_lightning_frequency.png"
    plot_frequency_map(samples, longitudes, latitudes, shape, frequency_path)
    log.info("Written %s", frequency_path)
    diurnal_path = RESULTS_DIR / "diurnal_cycle.png"
    plot_diurnal_cycle(samples, diurnal_path)
    log.info("Written %s", diurnal_path)


# ----------------------------------------------------------------------- main

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--time", help="forecast time of the case study, e.g. 2026-07-15T15:00 (UTC)")
    parser.add_argument("--only", nargs="+", choices=["reliability", "maps"], default=["reliability", "maps"],
                        help="which plots to produce (default: all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    status = 0
    if "reliability" in args.only and not run_reliability():
        status = 1
    if "maps" in args.only:
        run_maps(args.time)
    return status


if __name__ == "__main__":
    sys.exit(main())

