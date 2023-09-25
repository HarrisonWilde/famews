from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from famews.fairness_check.metrics import (
    get_corrected_npv,
    get_corrected_precision,
    get_npv,
    get_precision,
)
from famews.scripts.run_fairness_analysis import main as fairness_main


def test_prevalence_correction_binary_metrics():
    y_true = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    y_pred = np.random.choice([0, 1], size=100, p=[0.6, 0.4])
    prev_baseline_precision = 0.5
    prev_baseline_npv = 0.1
    prev = 0.3
    precision = get_precision(y_true, y_pred)
    corrected_precision = get_corrected_precision(y_true, y_pred, prev_baseline_precision, prev)
    assert (
        corrected_precision >= precision
    ), f"Prevalence correction lead to a smaller precision: {corrected_precision} vs {precision}"
    npv = get_npv(y_true, y_pred)
    corrected_npv = get_corrected_npv(y_true, y_pred, prev_baseline_npv, prev)
    assert (
        corrected_npv >= npv
    ), f"Prevalence correction lead to a smaller NPV: {corrected_npv} vs {npv}"
