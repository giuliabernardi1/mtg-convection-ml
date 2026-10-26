import numpy as np
import pandas as pd

from src.train import assign_blocks, leave_one_block_out, leave_one_day_out


def make_synthetic_samples(sample_count_per_day: int = 3000) -> pd.DataFrame:
    random_generator = np.random.default_rng(0)
    day_frames = []
    for day in ("2026-07-01", "2026-07-10", "2026-07-20"):
        cooling = random_generator.normal(0, 5, sample_count_per_day)
        day_frames.append(pd.DataFrame({
            "day": day,
            "ir_mean": random_generator.normal(250, 10, sample_count_per_day),
            "ir_cooling": cooling,
            "label": cooling < -7,                  # the label is a clear function of one feature
            "flash_now": random_generator.random(sample_count_per_day) < 0.01,
        }))
    return pd.concat(day_frames, ignore_index=True)

"""
def test_leave_one_day_out_learns_a_clear_signal_and_reports_persistence():
    samples = make_synthetic_samples()

    results = leave_one_day_out(samples, ["ir_mean", "ir_cooling"])

    assert list(results["test_day"]) == ["2026-07-18", "2026-07-19", "2026-07-20"]
    assert (results["model_csi"] > 0.8).all()
    assert (results["persistence_csi"] < 0.1).all()      # random flash_now cannot predict the label
    assert {"pr_auc", "threshold", "model_pod", "persistence_far"} <= set(results.columns)
"""

def test_assign_blocks_merges_days_within_the_gap_and_splits_beyond_it():
    days = pd.Series(["2026-07-18", "2026-07-19", "2026-07-19", "2026-07-22", "2026-07-24", "2026-07-30"])

    block_ids = assign_blocks(days, max_gap_days=2)

    assert list(block_ids) == [0, 0, 0, 1, 1, 2]


def test_leave_one_block_out_learns_a_clear_signal_and_reports_persistence():
    samples = make_synthetic_samples()  # change its three days to 

    results = leave_one_block_out(samples, ["ir_mean", "ir_cooling"])

    assert len(results) == 3
    assert (results["model_csi"] > 0.8).all()
    assert (results["persistence_csi"] < 0.1).all()
    assert {"test_block", "test_days", "pr_auc", "threshold"} <= set(results.columns)