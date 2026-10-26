import logging
import sys

import pandas as pd

from src.config import CONFIG, project_path
from src.labels import EXTRA_FEATURE_COLUMNS, FEATURE_COLUMNS
from src.train import leave_one_block_out, pooled_scores

SAMPLES_PATH = project_path(CONFIG["paths"]["samples"])
RESULTS_DIR = project_path(CONFIG["paths"]["results_dir"])

# Feature groups referenced by name in config.yaml (train.feature_sets)
FEATURE_GROUPS = {
    "base": FEATURE_COLUMNS,
    "current_lightning": ["current_lightning"],
    "fci_neighbourhood": ["ir_min_neighbourhood_min", "ir_min_cooling_neighbourhood_min"],
    "lightning_intensity": ["flash_sum_now", "flash_sum_past"],
    "extra": EXTRA_FEATURE_COLUMNS,
}
FEATURE_SETS = {
    name: [column for group in groups for column in FEATURE_GROUPS[group]]
    for name, groups in CONFIG["train"]["feature_sets"].items()
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    samples = pd.read_parquet(SAMPLES_PATH)
    samples["current_lightning"] = samples["flash_now"].astype("float32")
    RESULTS_DIR.mkdir(exist_ok=True)

    for feature_set_name, feature_columns in FEATURE_SETS.items():
        #fold_results = leave_one_day_out(samples, feature_columns)
        fold_results = leave_one_block_out(samples, feature_columns)
        fold_results.to_csv(RESULTS_DIR / f"{feature_set_name}.csv", index=False)
        fold_results.attrs["reliability"].to_csv(RESULTS_DIR / f"{feature_set_name}_reliability.csv", index=False)
        print(f"\n=== {feature_set_name} ===")
        #print(fold_results[["test_day", "threshold", "pr_auc", "roc_auc", "brier", "brier_skill", "model_csi", "persistence_csi"]].round(3))
        print(fold_results[["test_block","test_days", "threshold", "pr_auc", "roc_auc", "brier", "brier_skill", "model_csi", "persistence_csi"]].round(3))
        for prefix in ("model", "persistence"):
            print(prefix, {name: round(value, 3) for name, value in pooled_scores(fold_results, prefix).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())