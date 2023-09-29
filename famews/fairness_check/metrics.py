import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.metrics._ranking import _binary_clf_curve


def get_event_recall(
    predictions_dict: Dict[int, Tuple[np.array, np.array]],
    list_pids: List[int],
    event_bounds: Dict[int, list],
    threshold: float,
    horizon: int = 12,
    max_len: int = 2016,
    timestep: int = 5,
) -> float:
    """Compute event-based recall.

    Parameters
    ----------
    predictions_dict : Dict[int, Tuple[np.array, np.array]]
        Predictions - dictionary {pid: (pred, label)} with pred an array of float between 0 and 1 and label an array of 0/1
    list_pids : List[int]
        List of pids to consider to compute the metrics
    event_bounds : Dict[int, list]
        CF event bounds - dictionary {pid: [start_cf, stop_cf]}
    threshold: float
        Threshold on score
    horizon : int, optional
        Prediction range in hour, by default 12
    max_len: int, optional
        max. considered length of stay for the predictions (on timestep min grid), by default 2016
    timestep: int, optional
        time step for the time series in minutes, by default 5

    Returns
    -------
    float
        Event-based recall
    """
    nb_captured_event = 0
    nb_event = 0
    range_pred = int(horizon * 60 / timestep)  # should the resolution be a parameter as well?
    for pid in list_pids:
        if pid not in event_bounds:
            continue
        pred = predictions_dict[pid][0]
        label = predictions_dict[pid][1]
        event_bound_pid = event_bounds[pid]
        for (start, stop) in event_bound_pid:
            if start > max_len + range_pred:
                break
            if start == 0:  # event is at the start of the stay -> cannot predict
                continue
            nb_event += 1
            pred_event = pred[max(0, start - range_pred) : min(start, max_len)]
            label_event = label[max(0, start - range_pred) : min(start, max_len)]
            mask_nan = np.isnan(label_event)
            pred_event = (pred_event[~mask_nan] >= threshold).astype(int)
            if sum(pred_event) > 0:
                nb_captured_event += 1
    return nb_captured_event / nb_event


def get_precision(y_true: np.array, y_pred: np.array) -> float:
    """Compute precision tp/(tp+fp).
    If tp+fp is zero then throw a warning and return zero.
    Parameters
    ----------
    y_true : np.array
        Label - array of 0/1
    y_pred : np.array
        Binary predictions - array of 0/1

    Returns
    -------
    float
        precision
    """
    tp = np.logical_and(y_true == y_pred, y_pred == 1).sum()
    fp = np.logical_and(y_true != y_pred, y_pred == 1).sum()
    if tp + fp == 0:
        warnings.warn(
            "No positive class found in y_pred, "
            "precision is set to zero."  # one can use logging.captureWarnings(True) to capture these warning in the log
        )
    return tp / (tp + fp)


def get_recall(y_true: np.array, y_pred: np.array) -> float:
    """Compute recall tp/(tp+fn).
    If tp+fn is zero then throw a warning and return zero.
    Parameters
    ----------
    y_true : np.array
        Label - array of 0/1
    y_pred : np.array
        Binary predictions - array of 0/1

    Returns
    -------
    float
        recall
    """
    tp = np.logical_and(y_true == y_pred, y_pred == 1).sum()
    p = y_true.sum()
    if p == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to zero."  # one can use logging.captureWarnings(True) to capture these warning in the log
        )
    return tp / p


def get_fpr(y_true: np.array, y_pred: np.array) -> float:
    """Compute FPR (false positive rate): fp / total number of negatives

    Parameters
    ----------
    y_true : np.array
        Label - array of 0/1
    y_pred : np.array
        Binary predictions - array of 0/1

    Returns
    -------
    float
        False positive rate
    """
    fp = np.logical_and(y_true != y_pred, y_pred == 1).sum()
    return fp / (len(y_true) - y_true.sum())


def get_npv(y_true: np.array, y_pred: np.array) -> float:
    """Compute NPV (negative predictive value): tn / total negative calls

    Parameters
    ----------
    y_true : np.array
        Label - array of 0/1
    y_pred : np.array
        Binary predictions - array of 0/1

    Returns
    -------
    float
        Negative predictive value
    """
    tn = np.logical_and(y_true == y_pred, y_pred == 0).sum()
    return tn / (len(y_pred) - y_pred.sum())


def get_corrected_precision(
    y_true: np.array, y_pred: np.array, prev_baseline: float, prev_group: float
) -> float:
    """Compute corrected precision TP / (TP+s*FP)

    Parameters
    ----------
    y_true : np.array
        Label - array of 0/1
    y_pred : np.array
        Binary predictions - array of 0/1
    prev_baseline : float
        prevalence baseline
    prev_group : float
        prevalence for the cohort

    Returns
    -------
    float
        corrected precision
    """
    tp = np.logical_and(y_true == y_pred, y_pred == 1).sum()
    fp = np.logical_and(y_true != y_pred, y_pred == 1).sum()
    s = (1 / prev_baseline - 1) / (1 / prev_group - 1)
    return tp / (tp + s * fp)


def get_corrected_npv(
    y_true: np.array, y_pred: np.array, prev_baseline: float, prev_group: float
) -> float:
    """Compute corrected NPV TN / (TN+s*FN)

    Parameters
    ----------
    y_true : np.array
        Label - array of 0/1
    y_pred : np.array
        Binary predictions - array of 0/1
    prev_baseline : float
        prevalence baseline
    prev_group : float
        prevalence for the cohort

    Returns
    -------
    float
        corrected NPV
    """
    tn = np.logical_and(y_true == y_pred, y_pred == 0).sum()
    fn = np.logical_and(y_true != y_pred, y_pred == 0).sum()
    s = (1 / prev_group - 1) / (1 / prev_baseline - 1)
    return tn / (tn + s * fn)


def get_positive_class_score(y_true: np.array, y_pred: np.array) -> float:
    """Compute average predicted score for positive target

    Parameters
    ----------
    y_true : np.array
        Label - array of 0/1
    y_pred : np.array
        Predictions - array of float between 0 and 1

    Returns
    -------
    float
        average score on positive class
    """
    return np.mean(y_pred[np.nonzero(y_true)])


def get_negative_class_score(y_true: np.array, y_pred: np.array) -> float:
    """Compute average predicted score for negative target

    Parameters
    ----------
    y_true : np.array
        Label - array of 0/1
    y_pred : np.array
        Predictions - array of float between 0 and 1

    Returns
    -------
    float
        average score on negative class
    """
    return np.mean(y_pred[np.nonzero(y_true == 0)])


def get_error_calibration(prob_true: np.array, prob_pred: np.array) -> float:
    """Compute calibration error.

    Parameters
    ----------
    prob_true : np.array
        proportion of positive samples in each bin (used to draw the calibration curve)
    prob_pred : np.array
        mean predicted probability in each bin (used to draw the calibration curve)

    Returns
    -------
    float
        calibration error
    """
    error = np.abs(prob_true - prob_pred)
    bin_width = 1 / len(error)
    return np.sum(error * bin_width)


def get_roc(y_true: np.array, y_pred: np.array):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return fpr, tpr


def _get_event_recall_thresholds(
    predictions_dict: Dict[int, Tuple[np.array, np.array]],
    list_pids: List[int],
    event_bounds: Dict[int, list],
    thresholds: np.array,
    horizon: int = 12,
    max_len: int = 2016,
    timestep: int = 5,
) -> np.array:
    """Compute the event-based recall for each threshold.

    Parameters
    ----------
    predictions_dict : Dict[int, Tuple[np.array, np.array]]
        Predictions - dictionary {pid: (pred, label)} with pred an array of float between 0 and 1 and label an array of 0/1
    list_pids : List[int]
        List of pids to consider to compute the metrics
    event_bounds : Dict[int, list]
        CF event bounds - dictionary {pid: [start_cf, stop_cf]}
    thresholds : np.array
        array of thresholds
    horizon : int, optional
        Prediction range in hour, by default 12
    max_len : int, optional
        max. considered length of stay for the predictions (on timestep min grid), by default 2016
    timestep: int, optional
        Timestep of the time series in minutes, by default 5

    Returns
    -------
    np.array
        array of event-based recall values
    """
    nb_captured_event = np.zeros(len(thresholds))
    nb_event = 0
    range_pred = int(horizon * 60 / timestep)
    for pid in list_pids:
        if pid not in event_bounds:
            continue
        pred = predictions_dict[pid][0]
        label = predictions_dict[pid][1]
        event_bound_pid = event_bounds[pid]
        for (start, stop) in event_bound_pid:
            if start > max_len + range_pred:
                break
            if start == 0:  # event is at the start of the stay -> cannot predict
                continue
            nb_event += 1
            pred_event = pred[max(0, start - range_pred) : min(start, max_len)]
            label_event = label[max(0, start - range_pred) : min(start, max_len)]
            mask_nan = np.isnan(label_event)
            pred_event = pred_event[~mask_nan]
            if pred_event.size:
                nb_captured_event += (np.max(pred_event) >= thresholds).astype(int)
    event_based_recall = nb_captured_event / nb_event
    event_based_recall = np.append(event_based_recall, 0)
    return event_based_recall


def get_precision_recall_all(
    y_true: np.array,
    y_pred: np.array,
    predictions_dict: Dict[int, Tuple[np.array, np.array]] = None,
    list_pids: List[int] = None,
    event_bounds: Dict[int, list] = None,
    prev_baseline: float = None,
    prev_group: float = None,
    do_correction: bool = False,
    do_event_based: bool = False,
    horizon: int = 12,
    max_len: int = 2016,
    timestep: int = 5,
) -> Tuple[np.array, np.array, Optional[np.array], Optional[np.array]]:
    """Compute the precision and recall for each threshold, If do_correction then also compute the corrected precision.
    If do_event_based then also compute the event-based recall.
    The code is inspired from the sklearn.metrics.precision_recall_curve source code: https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/metrics/_ranking.py#L851

    Parameters
    ----------
    y_true : np.array
        _description_
    y_pred : np.array
        _description_
    predictions_dict : Dict[int, Tuple[np.array, np.array]], optional
        Predictions - dictionary {pid: (pred, label)} with pred an array of float between 0 and 1 and label an array of 0/1, by default None
    list_pids : List[int], optional
        List of pids to consider to compute the metrics, by default None
    event_bounds : Dict[int, list], optional
        CF event bounds - dictionary {pid: [start_cf, stop_cf]}, by default None
    prev_baseline : float, optional
        prevalence baseline for precision, by default None
    prev_group : float, optional
        prevalence for the cohort, by default None
    do_correction : bool, optional
        whether to compute precision with corrected prevalence, by default False
    do_event_based : bool, optional
        whether to compute event-based recall, by default False
    horizon : int, optional
        Prediction range in hour, by default 12
    max_len : int, optional
        max. considered length of stay for the predictions (on timestep min grid), by default 2016
    timestep: int, optional
        Time step for the time series in minutes, by default 5

    Returns
    -------
    Tuple[np.array, np.array, Optional[np.array], Optional[np.array]]
        precision array, recall array, corrected precision array, event-based recall array
    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true,
        y_pred,
    )
    # filter out unnecessary points
    if len(fps) > 2:
        # Drop thresholds corresponding to points where true positives (tps)
        # do not change from the previous or subsequent point. This will keep
        # only the first and last point for each tps value. All points
        # with the same tps value have the same recall and thus x coordinate.
        # They appear as a vertical line on the plot.
        optimal_idxs = np.where(
            np.concatenate([[True], np.logical_or(np.diff(tps[:-1]), np.diff(tps[1:])), [True]])
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]
    # compute precision
    ps = tps + fps
    # Initialize the result array with zeros to make sure that precision[ps == 0]
    # does not contain uninitialized values.
    precision = np.zeros_like(tps)
    np.divide(tps, ps, out=precision, where=(ps != 0))
    sl = slice(None, None, -1)
    thresholds = thresholds[sl]
    precision = np.hstack((precision[sl], 1))
    # compute recall
    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."  # one can use logging.captureWarnings(True) to capture these warning in the log
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]
    recall = np.hstack((recall[sl], 0))
    # compute corrected precision
    if do_correction and prev_baseline is not None and prev_group is not None:
        s = (1 / prev_baseline - 1) / (1 / prev_group - 1)
        num = tps + s * fps  # numerator for corrected precision
        corrected_precision = np.zeros_like(tps)
        np.divide(tps, num, out=corrected_precision, where=(num != 0))
        corrected_precision = np.hstack((corrected_precision[sl], 1))
    else:
        corrected_precision = None
    # compute event_based recall
    if (
        do_event_based
        and predictions_dict is not None
        and list_pids is not None
        and event_bounds is not None
    ):
        event_based_recall = _get_event_recall_thresholds(
            predictions_dict,
            list_pids,
            event_bounds,
            thresholds,
            horizon=horizon,
            max_len=max_len,
            timestep=timestep,
        )
    else:
        event_based_recall = None
    return precision, recall, corrected_precision, event_based_recall
