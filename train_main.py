import logging
import sys
from pathlib import Path

import pandas as pd

from src.labels import EXTRA_FEATURE_COLUMNS, FEATURE_COLUMNS
from src.train import leave_one_day_out, pooled_scores

PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLES_PATH = PROJECT_ROOT / "data" / "samples.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" 

FCI_NEIGHBOURHOOD = ["ir_min_neighbourhood_min", "ir_min_cooling_neighbourhood_min"]
LIGHTNING_INTENSITY = ["flash_sum_now", "flash_sum_past"]
FEATURE_SETS = {
    "fci_only": FEATURE_COLUMNS,
    "fci_plus_current_lightning": FEATURE_COLUMNS + ["current_lightning"],
    "fci_plus_neighbourhood": FEATURE_COLUMNS + FCI_NEIGHBOURHOOD,
    "fci_plus_lightning_intensity": FEATURE_COLUMNS + LIGHTNING_INTENSITY,
    "all_features": FEATURE_COLUMNS + ["current_lightning"] + EXTRA_FEATURE_COLUMNS,
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples = pd.read_parquet(SAMPLES_PATH)
    samples["current_lightning"] = samples["flash_now"].astype("float32")
    RESULTS_DIR.mkdir(exist_ok=True)

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        fold_results = leave_one_day_out(samples, feature_columns)
        fold_results.to_csv(RESULTS_DIR / f"{feature_set_name}.csv", index=False)
        print(f"\n=== {feature_set_name} ===")
        print(fold_results[["test_day", "threshold", "pr_auc", "model_csi", "persistence_csi"]].round(3))
        for prefix in ("model", "persistence"):
            print(prefix, {name: round(value, 3) for name, value in pooled_scores(fold_results, prefix).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())