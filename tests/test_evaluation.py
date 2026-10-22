import math

from src.evaluation import categorical_scores


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