"""Categorical verification scores for rare-event forecasts."""
import numpy as np


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