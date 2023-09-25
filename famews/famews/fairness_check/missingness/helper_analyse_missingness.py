from typing import Dict, List, Tuple

import numpy as np

from famews.fairness_check.metrics_wrapper import (
    compute_metrics_binary,
    compute_metrics_score,
)


def map_intensity_rate_category(
    rate: float, category_intensity: Dict[str, Tuple[float, float]]
) -> str:
    """Map the intensity of measurement rate to its corresponding category.

    Parameters
    ----------
    rate : float
        rate of measurement
    category_intensity : Dict_description_[str, Tuple[float, float]]
        Dictionary defining the categories of intensity of measurement {intensity_category: (lower, upper]}

    Returns
    -------
    str
        Intensity of measurement category
    """
    for name, (lower, upper) in category_intensity.items():
        if lower < rate and rate <= upper:
            return name


def get_predictions_no_msrt(
    predictions: Dict[str, Tuple[np.array, np.array]],
    bootstrap_pids: List[int],
    pids_no_msrt: List[int],
) -> Tuple[np.array, np.array]:
    """Select predictions and labels of patients without measurement.

    Parameters
    ----------
    predictions : Dict[str, Tuple[np.array, np.array]]
        Predictions dictionary
    bootstrap_pids : List[int]
        List of patient ids
    pids_no_msrt : List[int]
        PIDs of patients without measurement

    Returns
    -------
    Tuple[np.array, np.array]
        Array of predictions, array of labels
    """
    preds_array = []
    labels_array = []
    for pid in bootstrap_pids:
        if pid in pids_no_msrt:
            pred = predictions[pid][0]
            label = predictions[pid][1]
            mask_nan = np.isnan(label)
            preds_array.append(pred[~mask_nan])
            labels_array.append(label[~mask_nan])
    preds_array = np.concatenate(preds_array)
    labels_array = np.concatenate(labels_array)
    return preds_array, labels_array


def get_predictions_msrt_ind(
    predictions: Dict[str, Tuple[np.array, np.array]],
    bootstrap_pids: List[int],
    dict_pids_msrt_ind: Dict[int, Dict[str, List[int]]],
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Select predictions and labels for patients: with measurement and with missing measurement.

    Parameters
    ----------
    predictions : Dict[str, Tuple[np.array, np.array]]
        Predictions dictionary
    bootstrap_pids : List[int]
        List of patients ids
    dict_pids_msrt_ind : List[int]
        Map pid to its list of indices for each missingness category (with_msrt, missing_msrt)

    Returns
    -------
    Tuple[np.array, np.array]
        Array of predictions with measurement, array of labels with measurement,
        array of predictions missing measurement, array of labels missing measurement
    """
    preds_w_msrt = []
    labels_w_msrt = []
    preds_missing_msrt = []
    labels_missing_msrt = []
    for pid in bootstrap_pids:
        if pid in dict_pids_msrt_ind:
            dict_inds = dict_pids_msrt_ind[pid]
            ind_w_msrt = dict_inds["with_msrt"]
            ind_missing_msrt = dict_inds["missing_msrt"]
            pred = predictions[pid][0]
            label = predictions[pid][1]
            mask_nan = np.isnan(label)
            pred_w_msrt = pred[ind_w_msrt]
            label_w_msrt = label[ind_w_msrt]
            mask_nan_w_msrt = mask_nan[ind_w_msrt]
            preds_w_msrt.append(pred_w_msrt[~mask_nan_w_msrt])
            labels_w_msrt.append(label_w_msrt[~mask_nan_w_msrt])
            pred_missing_msrt = pred[ind_missing_msrt]
            label_missing_msrt = label[ind_missing_msrt]
            mask_nan_missing_msrt = mask_nan[ind_missing_msrt]
            preds_missing_msrt.append(pred_missing_msrt[~mask_nan_missing_msrt])
            labels_missing_msrt.append(label_missing_msrt[~mask_nan_missing_msrt])
    preds_w_msrt = np.concatenate(preds_w_msrt)
    labels_w_msrt = np.concatenate(labels_w_msrt)
    preds_missing_msrt = np.concatenate(preds_missing_msrt)
    labels_missing_msrt = np.concatenate(labels_missing_msrt)
    return preds_w_msrt, labels_w_msrt, preds_missing_msrt, labels_missing_msrt


def compute_prevalence_timepoint_categories(
    labels_tp: Dict[str, np.array]
) -> Tuple[Dict[str, float], float, float]:
    """Compute the prevalence for each missingness category.

    Parameters
    ----------
    labels_group : Dict[str, np.array]
        Labels for each missingness category {cat: label}

    Returns
    -------
    Tuple[Dict[str, float], float, float]
        dictionary of prevalence {cat: group prevalence}, prevalence baseline for precision, prevalence baseline for NPV
    """
    # labels_dict has the form: {cat: labels}
    dict_prevalence = {}
    for cat, label in labels_tp.items():
        dict_prevalence[cat] = np.sum(label) / len(label)
    prev_baseline_precision = max(dict_prevalence.values())
    prev_baseline_npv = min(dict_prevalence.values())
    return dict_prevalence, prev_baseline_precision, prev_baseline_npv


def compute_metrics_timepoint(
    labels_array: np.array,
    preds_array: np.array,
    threshold: float,
    prev_baseline_precision: float = None,
    prev_baseline_npv: float = None,
    prev_group: float = None,
    name_suffix: str = None,
    metrics_binary_to_check: List[str] = [
        "recall",
        "precision",
        "npv",
        "fpr",
        "corrected_precision",
        "corrected_npv",
    ],
    metrics_score_to_check: List[str] = [
        "positive_class_score",
        "negative_class_score",
        "auroc",
        "auprc",
        "corrected_auprc",
    ],
) -> Dict[str, float]:
    """Compute metrics for a missingness category.

    Parameters
    ----------
    labels_array: np.array
        concatenated labels without nan for a set of time points
    preds_array: np.array
        concatenated preds without nan for a set of time points
    threshold : float
        threshold on score
    prev_baseline_precision : float, optional
        prevalence baseline to correct precision, by default None
    prev_baseline_npv : float, optional
        prevalence baseline to correct NPV, by default None
    prev_group : float, optional
        prevalence for the missingness category, by default None
    name_suffix : str, optional
        name suffix for binary metrics, by default None
    metrics_binary_to_check : List[str], optional
        list of binary metrics names, by default [ "recall", "precision", "npv", "fpr", "corrected_precision", "corrected_npv"]
    metrics_score_to_check : List[str], optional
        list of score metrics names, by default [ "positive_class_score", "negative_class_score", "auroc", "auprc", "corrected_auprc"]

    Returns
    -------
    Dict[str, float]
        metrics {metric_name: value}
    """
    binary_preds_array = (preds_array >= threshold).astype(int)
    metrics_binary = compute_metrics_binary(
        labels_array,
        binary_preds_array,
        prev_baseline_precision=prev_baseline_precision,
        prev_baseline_npv=prev_baseline_npv,
        prev_group=prev_group,
        metrics_to_check=metrics_binary_to_check,
        name_suffix=name_suffix,
    )
    metrics_score, _ = compute_metrics_score(
        labels_array,
        preds_array,
        prev_baseline_precision=prev_baseline_precision,
        prev_group=prev_group,
        store_curve=False,
        metrics_to_check=metrics_score_to_check,
    )
    return {**metrics_binary, **metrics_score}
