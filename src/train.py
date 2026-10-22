"""Leave-one-day-out evaluation of a gradient boosting nowcast against persistence."""
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from src.evaluation import categorical_scores

log = logging.getLogger(__name__)

PROBABILITY_THRESHOLDS = np.arange(0.02, 0.90, 0.02)


def fit_model(train_samples: pd.DataFrame, feature_columns: list[str]) -> HistGradientBoostingClassifier:
    """Fixed hyperparameters, no early stopping (a random validation split would leak neighbouring cells)."""
    model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.1, max_iter=150,
        l2_regularization=1.0, early_stopping=False, random_state=0,
    )
    model.fit(train_samples[feature_columns], train_samples["label"])
    return model


def choose_threshold(train_samples: pd.DataFrame, feature_columns: list[str]) -> float:
    """Probability threshold maximising CSI on out-of-fold predictions (inner leave-one-day-out)."""
    out_of_fold_probabilities = pd.Series(index=train_samples.index, dtype="float64")
    for inner_day in train_samples["day"].unique():
        is_inner_day = train_samples["day"] == inner_day
        inner_model = fit_model(train_samples[~is_inner_day], feature_columns)
        out_of_fold_probabilities[is_inner_day] = inner_model.predict_proba(
            train_samples.loc[is_inner_day, feature_columns]
        )[:, 1]
    csi_per_threshold = [
        categorical_scores(train_samples["label"], out_of_fold_probabilities >= threshold)["csi"]
        for threshold in PROBABILITY_THRESHOLDS
    ]
    return float(PROBABILITY_THRESHOLDS[int(np.nanargmax(csi_per_threshold))])


def leave_one_day_out(samples: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """One row per test day: model scores and persistence scores on the same samples."""
    fold_results = []
    for test_day in sorted(samples["day"].unique()):
        is_test_day = samples["day"] == test_day
        train_samples, test_samples = samples[~is_test_day], samples[is_test_day]

        threshold = choose_threshold(train_samples, feature_columns)
        model = fit_model(train_samples, feature_columns)
        probabilities = model.predict_proba(test_samples[feature_columns])[:, 1]

        model_scores = categorical_scores(test_samples["label"], probabilities >= threshold)
        persistence_scores = categorical_scores(test_samples["label"], test_samples["flash_now"])
        fold_results.append({
            "test_day": test_day,
            "threshold": threshold,
            "pr_auc": average_precision_score(test_samples["label"], probabilities),
            **{f"model_{name}": value for name, value in model_scores.items()},
            **{f"persistence_{name}": value for name, value in persistence_scores.items()},
        })
        log.info(
            "%s | threshold %.2f | CSI model %.3f vs persistence %.3f",
            test_day, threshold, model_scores["csi"], persistence_scores["csi"],
        )
    return pd.DataFrame(fold_results)


def pooled_scores(fold_results: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Scores from counts summed over all test days (days with more positives weigh more)."""
    hits = fold_results[f"{prefix}_hits"].sum()
    misses = fold_results[f"{prefix}_misses"].sum()
    false_alarms = fold_results[f"{prefix}_false_alarms"].sum()
    return {
        "pod": hits / (hits + misses),
        "far": false_alarms / (hits + false_alarms),
        "csi": hits / (hits + misses + false_alarms),
    }