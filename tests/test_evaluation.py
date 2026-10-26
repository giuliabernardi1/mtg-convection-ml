import math

from src.evaluation import categorical_scores, pool_reliability, probabilistic_scores, reliability_table


def test_scores_on_a_hand_computed_example():
    observed = [True, True, True, False, False, False, False]
    predicted = [True, True, False, True, False, False, False]

    scores = categorical_scores(observed, predicted)

    assert (scores["hits"], scores["misses"], scores["false_alarms"]) == (2, 1, 1)
    assert math.isclose(scores["pod"], 2 / 3)
    assert math.isclose(scores["far"], 1 / 3)
    assert math.isclose(scores["csi"], 2 / 4)


def test_scores_are_nan_when_there_is_nothing_to_divide():
    scores = categorical_scores([False, False], [False, False])

    assert math.isnan(scores["pod"])
    assert math.isnan(scores["far"])
    assert math.isnan(scores["csi"])


def test_probabilistic_scores_on_a_hand_computed_example():
    observed = [True, False, False, False]
    probabilities = [0.8, 0.2, 0.0, 0.4]

    scores = probabilistic_scores(observed, probabilities)

    assert math.isclose(scores["brier"], (0.04 + 0.04 + 0.0 + 0.16) / 4)
    assert math.isclose(scores["brier_skill"], 1 - 0.06 / (0.25 * 0.75))
    assert scores["roc_auc"] == 1.0


def test_probabilistic_scores_are_nan_when_the_event_never_occurs():
    scores = probabilistic_scores([False, False], [0.1, 0.2])

    assert math.isnan(scores["brier_skill"])
    assert math.isnan(scores["roc_auc"])


def test_reliability_table_bins_forecasts_and_pools_folds():
    table = reliability_table([True, False, True, True], [0.05, 0.05, 0.95, 1.0], n_bins=10)

    assert list(table["bin"]) == [0, 9]
    assert list(table["count"]) == [2, 2]
    assert list(table["observed_frequency"]) == [0.5, 1.0]

    pooled = pool_reliability([table, table])
    assert list(pooled["count"]) == [4, 4]
    assert list(pooled["observed_frequency"]) == [0.5, 1.0]
