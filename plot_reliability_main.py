"""Plot reliability diagrams (and forecast-frequency histograms) from results/*_reliability.csv."""
import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import CONFIG, project_path

log = logging.getLogger(__name__)

RESULTS_DIR = project_path(CONFIG["paths"]["results_dir"])
OUTPUT_PATH = RESULTS_DIR / "reliability.png"
SUFFIX = "_reliability.csv"
MIN_COUNT = 30  # bins with fewer samples are too noisy: drawn faded


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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = sorted(RESULTS_DIR.glob(f"*{SUFFIX}"))
    if not paths:
        log.error("No *%s files in %s: run train_main.py / cnn_main.py first", SUFFIX, RESULTS_DIR)
        return 1
    tables = {path.name.removesuffix(SUFFIX): pd.read_csv(path) for path in paths}
    plot_reliability(tables, OUTPUT_PATH)
    log.info("Written %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())

