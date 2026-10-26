"""Verification scores for rare-event forecasts: categorical (POD, FAR, CSI) and probabilistic."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

def categorical_scores(observed, predicted) -> dict[str, float]:
    """POD, FAR and CSI from boolean arrays; a ratio with an empty denominator is NaN."""
    observed = np.asarray(observed, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    hits = int((observed & predicted).sum())
    misses = int((observed & ~predicted).sum())
    false_alarms = int((~observed & predicted).sum())

    def safe_ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else float("nan")

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "pod": safe_ratio(hits, hits + misses),                      # probability of detection
        "far": safe_ratio(false_alarms, hits + false_alarms),        # false alarm ratio
        "csi": safe_ratio(hits, hits + misses + false_alarms),       # critical success index
    }


def probabilistic_scores(observed, probabilities) -> dict[str, float]:
    """Brier score, Brier skill score and ROC-AUC of event probabilities.

    The skill score uses the climatological base rate of the evaluated sample as reference
    (BS_ref = p * (1 - p)); it is NaN when the event never or always occurs. ROC-AUC is NaN
    when only one class is present.
    """
    observed = np.asarray(observed, dtype=bool).ravel()
    probabilities = np.asarray(probabilities, dtype="float64").ravel()
    brier = float(brier_score_loss(observed, probabilities))
    base_rate = observed.mean()
    reference_brier = base_rate * (1 - base_rate)
    has_both_classes = 0 < observed.sum() < observed.size
    return {
        "brier": brier,
        "brier_skill": 1 - brier / reference_brier if reference_brier else float("nan"),
        "roc_auc": float(roc_auc_score(observed, probabilities)) if has_both_classes else float("nan"),
    }


def reliability_table(observed, probabilities, n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: per probability bin, mean forecast, observed frequency and count.

    Bins are equal-width on [0, 1]; empty bins are omitted. Counts allow pooling several folds.
    """
    observed = np.asarray(observed, dtype=bool).ravel()
    probabilities = np.asarray(probabilities, dtype="float64").ravel()
    bin_index = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        in_bin = bin_index == bin_id
        if in_bin.any():
            rows.append({
                "bin": bin_id,
                "count": int(in_bin.sum()),
                "positives": int(observed[in_bin].sum()),
                "mean_forecast": float(probabilities[in_bin].mean()),
                "observed_frequency": float(observed[in_bin].mean()),
            })
    return pd.DataFrame(rows, columns=["bin", "count", "positives", "mean_forecast", "observed_frequency"])


def pool_reliability(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-fold reliability tables by summing counts per bin."""
    combined = pd.concat(tables, ignore_index=True)
    combined["forecast_sum"] = combined["mean_forecast"] * combined["count"]
    pooled = combined.groupby("bin", as_index=False)[["count", "positives", "forecast_sum"]].sum()
    pooled["mean_forecast"] = pooled["forecast_sum"] / pooled["count"]
    pooled["observed_frequency"] = pooled["positives"] / pooled["count"]
    return pooled.drop(columns="forecast_sum")
