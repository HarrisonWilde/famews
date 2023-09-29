from typing import Dict, List, Tuple

import numpy as np

from famews.fairness_check.metrics_wrapper import (
    compute_metrics_binary,
    compute_metrics_score,
)


def compute_prevalence_groups(
    labels_group: Dict[str, np.array]
) -> Tuple[Dict[str, float], float, float]:
    """Compute the prevalence for each group.

    Parameters
    ----------
    labels_group : Dict[str, np.array]
        Labels for each cohort {cat: label}

    Returns
    -------
    Tuple[Dict[str, float], float, float]
        dictionary of prevalence {cat: group prevalence}, prevalence baseline for precision, prevalence baseline for NPV
    """
    # labels_dict has the form: {cat: labels}
    dict_prevalence = {}
    for cat, label in labels_group.items():
        dict_prevalence[cat] = np.sum(label) / len(label)
    prev_baseline_precision = max(dict_prevalence.values())
    prev_baseline_npv = min(dict_prevalence.values())
    return dict_prevalence, prev_baseline_precision, prev_baseline_npv


def map_predictions_group(
    predictions: Dict[str, Tuple[np.array, np.array]], map_group_pid: Dict[str, List[int]]
) -> Tuple[Dict[str, np.array], Dict[str, np.array],]:
    """Construct map between grouping category and different predictions and labels objects.

    Parameters
    ----------
    predictions : Dict[str, Tuple[np.array, np.array]]
        {pid: (pred, label)}
    map_group_pid : Dict[str, List[int]]
        {cat: [pids]}

    Returns
    -------
    Tuple[ Dict[str, np.array], Dict[str, np.array], ]
        {cat: labels}, {cat: preds}
    """
    labels_array_group = {}
    preds_array_group = {}
    for cat, list_pids in map_group_pid.items():
        labels_array, preds_array = filter_predictions_pid(predictions, list_pids)
        labels_array_group[cat] = labels_array
        preds_array_group[cat] = preds_array
    return labels_array_group, preds_array_group


def filter_predictions_pid(
    predictions: Dict[str, Tuple[np.array, np.array]], list_pids: List[str]
) -> Tuple[np.array, np.array]:
    """Filter the predictions to have only preds and labels for patients in list_pids.

    Parameters
    ----------
    predictions : Dict[str, Tuple[np.array, np.array]]
        {pid: (pred, label)}
    list_pids : List[str]
        list of pids

    Returns
    -------
    Tuple[ np.array, np.array]
        concatenated labels without nan for the list of pids, concatenated preds without nan for the list of pids,
    """
    preds_array = []
    labels_array = []
    for pid in list_pids:
        pred = predictions[pid][0]
        label = predictions[pid][1]
        mask_nan = np.isnan(label)
        preds_array.append(pred[~mask_nan])
        labels_array.append(label[~mask_nan])
    preds_array = np.concatenate(preds_array)
    labels_array = np.concatenate(labels_array)
    return labels_array, preds_array


def compute_metrics(
    predictions_dict: Dict[int, np.array],
    list_pids: List[int],
    labels_array: np.array,
    preds_array: np.array,
    threshold: float,
    event_bounds: Dict[int, list] = None,
    prev_baseline_precision: float = None,
    prev_baseline_npv: float = None,
    prev_group: float = None,
    horizon: int = 12,
    max_len: int = 2016,
    timestep: int = 5,
    name_suffix: str = None,
    store_curve: bool = False,
    nbins_calibration: int = 5,  # sklearn default
    metrics_binary_to_check: List[str] = [
        "recall",
        "precision",
        "npv",
        "fpr",
        "corrected_precision",
        "corrected_npv",
        "event_recall",
    ],
    metrics_score_to_check: List[str] = [
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
    """Compute metrics for a list of patients.

    Parameters
    ----------
    predictions_dict : Dict[int, Tuple[np.array, np.array]]
        Predictions - dictionary {pid: (pred, label)} with pred an array of float between 0 and 1 and label an array of 0/1
    list_pids : List[int]
        List of pids to consider to compute the metrics
    labels_array: np.array
        concatenated labels without nan for the list of pids
    preds_array: np.array
        concatenated preds without nan for the list of pids
    threshold : float
        threshold on score
    event_bounds : Dict[int, list], optional
        dictionary of cf event bounds {pid: [(start_cf, stop_cf)]}, by default None
    prev_baseline_precision : float, optional
        prevalence baseline to correct precision, by default None
    prev_baseline_npv : float, optional
        prevalence baseline to correct NPV, by default None
    prev_group : float, optional
        prevalence for the cohort, by default None
    horizon : int, optional
        prediction range in hour, by default 12
    max_len: int, optional
        max. considered length of stay for the predictions (on 5 min grid), by default 2016
    timestep: int, optional
        Timestep of time series in minutes, by default 5
    name_suffix : str, optional
        name suffix for binary metrics, by default None
    store_curve: bool, optional
        whether to store the curves, optional
    nbins_calibration: int, optional
        number of bins to generate the calibration curve, by default 5
    metrics_binary_to_check : List[str], optional
        list of binary metrics names, by default [ "recall", "precision", "npv", "fpr", "corrected_precision", "corrected_npv", "event_recall"]
    metrics_score_to_check : List[str], optional
        list of score metrics names, by default [ "positive_class_score", "negative_class_score", "auroc", "auprc", "corrected_auprc", "event_auprc", "corrected_event_auprc", "calibration_error"]

    Returns
    -------
    Dict[str, float]
        metrics {metric_name: value}
    """
    binary_preds_array = (preds_array >= threshold).astype(int)
    metrics_binary = compute_metrics_binary(
        labels_array,
        binary_preds_array,
        predictions_dict,
        list_pids,
        event_bounds,
        threshold,
        prev_baseline_precision,
        prev_baseline_npv,
        prev_group,
        horizon,
        max_len,
        timestep,
        metrics_binary_to_check,
        name_suffix,
    )
    metrics_score, curves = compute_metrics_score(
        labels_array,
        preds_array,
        predictions_dict,
        list_pids,
        event_bounds,
        horizon,
        max_len,
        timestep,
        prev_baseline_precision,
        prev_group,
        store_curve,
        nbins_calibration,
        metrics_score_to_check,
    )
    return {**metrics_binary, **metrics_score}, curves
