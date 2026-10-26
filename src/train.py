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


def assign_blocks(days: pd.Series, max_gap_days: int = 2) -> pd.Series:
    """Days separated by at most max_gap_days share a block (one weather episode)."""
    sorted_days = pd.Series(pd.to_datetime(sorted(days.unique())))
    starts_new_block = sorted_days.diff().dt.days.fillna(0) > max_gap_days
    block_ids = starts_new_block.cumsum()
    block_by_day = dict(zip(sorted_days.dt.strftime("%Y-%m-%d"), block_ids))
    return days.map(block_by_day)


def choose_threshold(train_samples: pd.DataFrame, feature_columns: list[str]) -> float:
    """Probability threshold maximising CSI on out-of-fold predictions (inner leave-one-block-out)."""
    out_of_fold_probabilities = pd.Series(index=train_samples.index, dtype="float64")
    for inner_block in train_samples["block"].unique():
        is_inner_block = train_samples["block"] == inner_block
        inner_model = fit_model(train_samples[~is_inner_block], feature_columns)
        out_of_fold_probabilities[is_inner_block] = inner_model.predict_proba(
            train_samples.loc[is_inner_block, feature_columns]
        )[:, 1]
    csi_per_threshold = [
        categorical_scores(train_samples["label"], out_of_fold_probabilities >= threshold)["csi"]
        for threshold in PROBABILITY_THRESHOLDS
    ]
    return float(PROBABILITY_THRESHOLDS[int(np.nanargmax(csi_per_threshold))])


def leave_one_block_out(
    samples: pd.DataFrame, feature_columns: list[str], max_gap_days: int = 2
) -> pd.DataFrame:
    """One row per test block: model scores and persistence scores on the same samples."""
    samples = samples.assign(block=assign_blocks(samples["day"], max_gap_days))
    fold_results = []
    for test_block in sorted(samples["block"].unique()):
        is_test_block = samples["block"] == test_block
        train_samples, test_samples = samples[~is_test_block], samples[is_test_block]

        threshold = choose_threshold(train_samples, feature_columns)
        model = fit_model(train_samples, feature_columns)
        probabilities = model.predict_proba(test_samples[feature_columns])[:, 1]

        model_scores = categorical_scores(test_samples["label"], probabilities >= threshold)
        persistence_scores = categorical_scores(test_samples["label"], test_samples["flash_now"])
        test_days = sorted(test_samples["day"].unique())
        fold_results.append({
            "test_block": test_block,
            "test_days": f"{test_days[0]}..{test_days[-1]} ({len(test_days)} days)",
            "threshold": threshold,
            "pr_auc": average_precision_score(test_samples["label"], probabilities),
            **{f"model_{name}": value for name, value in model_scores.items()},
            **{f"persistence_{name}": value for name, value in persistence_scores.items()},
        })
        log.info(
            "block %s (%s) | threshold %.2f | CSI model %.3f vs persistence %.3f",
            test_block, fold_results[-1]["test_days"], threshold,
            model_scores["csi"], persistence_scores["csi"],
        )
    return pd.DataFrame(fold_results)


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
        "pod": hits / (hits + misses), # probability of detection
        "far": false_alarms / (hits + false_alarms), # false alarm ratio
        "csi": hits / (hits + misses + false_alarms), # critical success index

    }