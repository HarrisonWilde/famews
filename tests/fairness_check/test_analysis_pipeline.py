from pathlib import Path

import numpy as np
import pytest

from famews.fairness_check.metrics import (get_corrected_npv,
                                           get_corrected_precision, get_npv,
                                           get_precision)
from famews.fairness_check.utils.compute_threshold_score import \
    GetThresholdScore
from famews.pipeline import PipelineState


def test_prevalence_correction_binary_metrics():
    y_true = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    y_pred = np.random.choice([0, 1], size=100, p=[0.6, 0.4])
    prev_baseline_precision = 0.5
    prev_baseline_npv = 0.1
    prev = 0.3
    precision = get_precision(y_true, y_pred)
    corrected_precision = get_corrected_precision(
        y_true, y_pred, prev_baseline_precision, prev
    )
    assert (
        corrected_precision >= precision
    ), f"Prevalence correction lead to a smaller precision: {corrected_precision} vs {precision}"
    npv = get_npv(y_true, y_pred)
    corrected_npv = get_corrected_npv(y_true, y_pred, prev_baseline_npv, prev)
    assert (
        corrected_npv >= npv
    ), f"Prevalence correction lead to a smaller NPV: {corrected_npv} vs {npv}"


@pytest.mark.parametrize(
    "name_threshold",
    ["recall_0.25", "precision_0.40", "npv_0.625", "fpr_0.42", "event_recall_0.5"],
)
def test_compute_threshold(name_threshold: str):
    state = PipelineState()
    state.name_threshold = name_threshold
    state.predictions = {
        1: (
            np.array([0.6, 0.1, 0.1, 0, 0, 0.3, 0.3]),
            np.array([0, 1, 1, np.nan, np.nan, 0, 0]),
        ),
        2: (
            np.array([0.2, 0.15, 0.25, 0.4, 0.6, 0, 0, 0, 0, 0.45, 0.45]),
            np.array(
                [
                    0,
                    0,
                    0,
                    1,
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    0,
                    0,
                ]
            ),
        ),
    }
    state.event_bounds = {1: [(3, 5)], 2: [(5, 9)]}
    state.horizon = 2 * 5 / 60
    state.max_len = 20
    state.timestep = 5
    get_threshold_stage = GetThresholdScore(state)
    get_threshold_stage.run()
    if get_threshold_stage.metric_name == "recall":
        assert (
            state.threshold > 0.4
        ), f"Threshold for target recall at 0.25 is wrong, expected 0.4 but got {state.threshold}"
    elif get_threshold_stage.metric_name == "precision":
        assert (
            state.threshold > 0.3
        ), f"Threshold for target precision at 0.40 is wrong, expected 0.3 but got {state.threshold}"
    elif get_threshold_stage.metric_name == "npv":
        assert (
            state.threshold > 0.4
        ), f"Threshold for target NPV at 0.625 is wrong, expected 0.4 but got {state.threshold}"
    elif get_threshold_stage.metric_name == "fpr":
        assert (
            state.threshold > 0.3
        ), f"Threshold for target recall at 0.42 is wrong, expected 0.3 but got {state.threshold}"
    elif get_threshold_stage.metric_name == "event_recall":
        assert (
            state.threshold > 0.1
        ), f"Threshold for target event-based recall at 0.5 is wrong, expected 0.1 but got {state.threshold}"
