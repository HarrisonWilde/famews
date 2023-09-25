import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc

from famews.fairness_check.metrics import (
    get_corrected_npv,
    get_corrected_precision,
    get_error_calibration,
    get_event_recall,
    get_fpr,
    get_negative_class_score,
    get_npv,
    get_positive_class_score,
    get_precision,
    get_precision_recall_all,
    get_recall,
    get_roc,
)


def compute_metrics_binary(
    y_true: np.array,
    y_pred: np.array,
    predictions_dict: Dict[int, Tuple[np.array, np.array]] = None,
    list_pids: List[int] = None,
    event_bounds: Dict[int, list] = None,
    threshold: float = None,
    prev_baseline_precision: float = None,
    prev_baseline_npv: float = None,
    prev_group: float = None,
    horizon: int = 12,
    max_len: int = 2016,
    timestep: int = 5,
    metrics_to_check: List[str] = [
        "recall",
        "precision",
        "npv",
        "fpr",
        "corrected_precision",
        "corrected_npv",
        "event_recall",
    ],
    name_suffix: str = None,
) -> Dict[str, float]:
    """Compute metrics based on binarised predictions.

    Parameters
    ----------
    y_true : np.array
        Label - array of 0/1
    y_pred : np.array
        Binary predictions - array of 0/1
    predictions_dict : Dict[int, Tuple[np.array, np.array]], optional
        Predictions - dictionary {pid: (pred, label)} with pred an array of float between 0 and 1 and label an array of 0/1, by default None
    list_pids : List[int], optional
        List of pids to consider to compute the metrics, by default None
    event_bounds : Dict[int, list], optional
        CF event bounds - dictionary {pid: [start_cf, stop_cf]}, by default None
    threshold: float, optional
        threshold on score, by default None
    prev_baseline_precision : float, optional
        Prevalence baseline for precision correction, by default None
    prev_baseline_npv : float, optional
        Prevalence baseline for NPV correction, by default None
    prev_group : float, optional
        Prevalence for the cohort, by default None
    horizon : int, optional
        Prediction range in hour, by default 12
    max_len: int, optional
        max. considered length of stay for the predictions (on timestep min grid), by default 2016
    timestep: int, optional
        Timestep of time series in minutes, by default 5
    metrics_to_check : List[str], optional
        List of metrics to check, by default ["recall", "precision", "npv", "fpr", "corrected_precision", "corrected_npv", "event_recall"]
    name_suffix : str, optional
        Suffix to append to the metric names, by default None

    Returns
    -------
    Dict[str, float]
        Dictionary of metric values {metric_name: value}
    """
    metrics = {}

    if "recall" in metrics_to_check:
        metrics[f"recall{name_suffix}"] = get_recall(y_true, y_pred)
    if "precision" in metrics_to_check:
        metrics[f"precision{name_suffix}"] = get_precision(y_true, y_pred)
    if "npv" in metrics_to_check:
        metrics[f"npv{name_suffix}"] = get_npv(y_true, y_pred)
    if "fpr" in metrics_to_check:
        metrics[f"fpr{name_suffix}"] = get_fpr(y_true, y_pred)
    if "corrected_precision" in metrics_to_check:
        if prev_baseline_precision and prev_group:
            metrics[f"corrected_precision{name_suffix}"] = get_corrected_precision(
                y_true, y_pred, prev_baseline_precision, prev_group
            )
        else:
            logging.warning(
                "prev_baseline_precision and prev_group have to be given to compute the corrected precision"
            )
    if "corrected_npv" in metrics_to_check:
        if prev_baseline_npv and prev_group:
            metrics[f"corrected_npv{name_suffix}"] = get_corrected_npv(
                y_true, y_pred, prev_baseline_npv, prev_group
            )
        else:
            logging.warning(
                "prev_baseline_npv and prev_group have to be given to compute the corrected NPV"
            )
    if (
        "event_recall" in metrics_to_check
        and predictions_dict is not None
        and list_pids is not None
        and event_bounds is not None
        and threshold is not None
    ):
        metrics[f"event_recall{name_suffix}"] = get_event_recall(
            predictions_dict,
            list_pids,
            event_bounds,
            threshold,
            horizon=horizon,
            max_len=max_len,
            timestep=timestep,
        )
    return metrics


def compute_metrics_score(
    y_true: np.array,
    y_pred: np.array,
    predictions_dict: Dict[int, Tuple[np.array, np.array]] = None,
    list_pids: Dict[int, int] = None,
    event_bounds: Dict[int, list] = None,
    horizon: int = 12,
    max_len: int = 2016,
    timestep: int = 5,
    prev_baseline_precision: float = None,
    prev_group: float = None,
    store_curve: bool = False,
    nbins_calibration: int = 5,  # sklearn default value
    metrics_to_check: List[str] = [
        "positive_class_score",
        "negative_class_score",
        "auroc",
        "auprc",
        "corrected_auprc",
        "event_auprc",
        "corrected_event_auprc",
        "calibration_error",
    ],
) -> Dict[str, float]:
    """Compute metrics based on score (for metrics that don't require binary input).

    Parameters
    ----------
    y_true : np.array
        Label - array of 0/1
    y_pred : np.array
        Predictions - array of float between 0 and 1
    predictions_dict : Dict[int, Tuple[np.array, np.array]], optional
        Predictions - dictionary {pid: (pred, label)} with pred an array of float between 0 and 1 and label an array of 0/1, by default None
    list_pids : List[int], optional
        List of pids to consider to compute the metrics, by default None
    event_bounds : Dict[int, list], optional
         CF event bounds - dictionary {pid: [start_cf, stop_cf]}, by default None
    horizon : int, optional
        Prediction range in hour, by default 12
    max_len: int, optional
        max. considered length of stay for the predictions (on timestep min grid), by default 2016
    timestep: int, optional
        Timestep of time series in minutes, by default 5
    prev_baseline_precision : float, optional
        Prevalence baseline for precision correction, by default None
    prev_group : float, optional
        Prevalence for the cohort, by default None
    store_curve: bool, optional
        whether to store the curves, by default False
    nbins_calibration: int, optional
        number of bins to generate the calibration curve, by default 5
    metrics_to_check : List[str], optional
        List of metrics to check, by default ["positive_class_score", "negative_class_score", "auroc", "auprc", "corrected_auprc", "event_auprc", "corrected_event_auprc", "calibration_error"]

    Returns
    -------
    Dict[str, float]
        Dictionary of metric values {metric_name: value}
    """
    metrics = {}
    curves = {}
    if "calibration_error" in metrics_to_check:
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=nbins_calibration)
        error = get_error_calibration(prob_true, prob_pred)
        metrics["calibration_error"] = error
        if store_curve:
            curves["calibration"] = {"prob_true": prob_true, "prob_pred": prob_pred, "error": error}
    if "positive_class_score" in metrics_to_check:
        metrics["positive_class_score"] = get_positive_class_score(y_true, y_pred)
    if "negative_class_score" in metrics_to_check:
        metrics["negative_class_score"] = get_negative_class_score(y_true, y_pred)
    if "auroc" in metrics_to_check:
        fpr, tpr = get_roc(y_true, y_pred)
        metrics["auroc"] = auc(fpr, tpr)
        if store_curve:
            curves["ROC"] = {"fpr": fpr, "tpr": tpr}
    do_correction = None
    do_event_based = None
    if "corrected_event_auprc" in metrics_to_check or (
        "corrected_auprc" in metrics_to_check and "event_auprc" in metrics_to_check
    ):
        do_correction = True
        do_event_based = True
    elif "corrected_auprc" in metrics_to_check:
        do_correction = True
        do_event_based = False
    elif "event_auprc" in metrics_to_check:
        do_correction = False
        do_event_based = True
    elif "auprc" in metrics_to_check:
        do_correction = False
        do_event_based = False
    if do_correction is not None and do_event_based is not None:
        precision, recall, corrected_precision, event_based_recall = get_precision_recall_all(
            y_true,
            y_pred,
            predictions_dict,
            list_pids,
            event_bounds,
            prev_baseline_precision,
            prev_group,
            do_correction,
            do_event_based,
            horizon,
            max_len,
            timestep,
        )
        if store_curve:
            curves["PRC"] = {
                "precision": precision,
                "recall": recall,
                "corrected_precision": corrected_precision,
                "event_recall": event_based_recall,
            }
    if "auprc" in metrics_to_check:
        metrics["auprc"] = auc(recall, precision)
    if "corrected_auprc" in metrics_to_check:
        if prev_baseline_precision is not None and prev_group is not None:
            metrics["corrected_auprc"] = auc(recall, corrected_precision)
        else:
            logging.warning(
                "prev_baseline_precision and prev_group have to be given to compute the corrected precision"
            )
    if (
        "event_auprc" in metrics_to_check
        and predictions_dict is not None
        and list_pids is not None
        and event_bounds is not None
    ):
        metrics["event_auprc"] = auc(event_based_recall, precision)
    if "corrected_event_auprc" in metrics_to_check and (
        predictions_dict is not None and list_pids is not None and event_bounds is not None
    ):
        if prev_baseline_precision is not None and prev_group is not None:
            metrics["corrected_event_auprc"] = auc(event_based_recall, corrected_precision)
        else:
            logging.warning(
                "prev_baseline_precision and prev_group have to be given to compute the corrected precision"
            )
    if store_curve:
        return metrics, curves
    return metrics, None
