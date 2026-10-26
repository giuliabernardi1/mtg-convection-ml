import logging
import sys

from src.cnn import run_leave_one_block_out
from src.config import CONFIG, project_path
from src.train import pooled_scores

PROCESSED_DIR = project_path(CONFIG["paths"]["processed_dir"])
RESULTS_DIR = project_path(CONFIG["paths"]["results_dir"])
EPOCHS = CONFIG["cnn"]["epochs"]
SUBSAMPLE_STEP = CONFIG["cnn"]["subsample_step"]
INPUT_SETS = CONFIG["cnn"]["input_sets"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(exist_ok=True)
    for set_name, use_lightning in INPUT_SETS.items():
        fold_results = run_leave_one_block_out(
            PROCESSED_DIR, use_lightning, epochs=EPOCHS, subsample_step=SUBSAMPLE_STEP
        )
        fold_results.to_csv(RESULTS_DIR / f"{set_name}.csv", index=False)
        print(f"\n=== {set_name} ===")
        print(fold_results[["test_block", "test_days", "threshold", "pr_auc", "model_csi", "persistence_csi"]].round(3))
        for prefix in ("model", "persistence"):
            print(prefix, {name: round(value, 3) for name, value in pooled_scores(fold_results, prefix).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())