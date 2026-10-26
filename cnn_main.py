import logging
import sys
from pathlib import Path

from src.cnn import run_leave_one_block_out
from src.train import pooled_scores

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
EPOCHS = 12
SUBSAMPLE_STEP = 1
INPUT_SETS = {"cnn_fci_only": False, "cnn_fci_plus_lightning": True}


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