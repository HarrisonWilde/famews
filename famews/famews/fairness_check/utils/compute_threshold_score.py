import logging
from typing import Dict, List, Tuple

import gin
import numpy as np
from sklearn.metrics import precision_recall_curve

from famews.fairness_check.metrics import get_event_recall, get_fpr, get_npv
from famews.fairness_check.model_performance.helper_analyse_performance import (
    filter_predictions_pid,
)
from famews.pipeline import PipelineState, StatefulPipelineStage


def find_threshold_event_recall(
    target_event_recall: float,
    predictions_dict: Dict[int, Tuple[np.array, np.array]],
    list_pids: List[int],
    event_bounds: Dict[int, list],
    horizon: int = 12,
    max_len: int = 2016,
    timestep: int = 5,
) -> float:
    """Find threshold with target value on event-based recall.

    Parameters
    ----------
    target_event_recall : float
        Target value on event-based recall
    predictions_dict : Dict[int, Tuple[np.array, np.array]]
        Predictions - dictionary {pid: (pred, label)} with pred an array of float between 0 and 1 and label an array of 0/1
    list_pids : List[int]
        List of pids to consider to compute the metrics
    event_bounds : Dict[int, list]
        CF event bounds - dictionary {pid: [start_cf, stop_cf]}
    horizon : int, optional
        Prediction range in hour, by default 12
    max_len: int, optional
        max. considered length of stay for the predictions (on 5 min grid), by default 2016
    timestep: int, optional
        Timestep of time series in minutes, by default 5


    Returns
    -------
    float
        Threshold
    """
    best_threshold = 0
    best_event_recall = 1
    for depth in [0.1, 0.01, 0.001]:
        thresholds = [best_threshold + depth * 10 - j * depth for j in range(1, 10)]
        event_recalls = [
            get_event_recall(
                predictions_dict, list_pids, event_bounds, thres, horizon, max_len, timestep
            )
            for thres in thresholds
        ]
        thresholds += [best_threshold]
        event_recalls += [best_event_recall]
        i = 0
        while event_recalls[i] < target_event_recall:
            i += 1
        best_event_recall = event_recalls[i]
        best_threshold = thresholds[i]
    return best_threshold


def find_threshold_precision_recall(
    target_metric: float,
    metric_name: str,
    labels_array: np.array,
    preds_array: np.array,
) -> float:
    """Find threshold with target value on metric_name, either precision or recall.

    Parameters
    ----------
    target_fpr : float
        Target value on precision or recall
    metric_name: str
        Precision or Recall
    labels_array : np.array
        Label - array of 0/1
    preds_array : np.array
        Predictions - array of float between 0 and 1

    Returns
    -------
    float
        Threshold
    """
    precision, recall, thresholds = precision_recall_curve(labels_array, preds_array)
    if metric_name == "recall":
        for j, val in enumerate(recall[::-1]):
            if val >= target_metric:
                return thresholds[::-1][j]

    elif metric_name == "precision":
        for j, val in enumerate(precision):
            if val >= target_metric:
                return thresholds[j]


def find_threshold_npv(
    target_npv: float,
    labels_array: np.array,
    preds_array: np.array,
) -> float:
    """Find threshold with target value on NPV.

    Parameters
    ----------
    target_fpr : float
        Target value on NPV
    labels_array : np.array
        Label - array of 0/1
    preds_array : np.array
        Predictions - array of float between 0 and 1

    Returns
    -------
    float
        Threshold
    """
    best_threshold = 0
    best_npv = 1
    for depth in [0.1, 0.01, 0.001]:
        thresholds = [best_threshold + depth * 10 - j * depth for j in range(1, 10)]
        npvs = [get_npv(labels_array, preds_array >= thres) for thres in thresholds]
        thresholds += [best_threshold]
        npvs += [best_npv]
        i = 0
        while npvs[i] < target_npv:
            i += 1
        best_npv = npvs[i]
        best_threshold = thresholds[i]
    return best_threshold


def find_threshold_fpr(
    target_fpr: float,
    labels_array: np.array,
    preds_array: np.array,
) -> float:
    """Find threshold with target value on FPR.

    Parameters
    ----------
    target_fpr : float
        Target value on FPR
    labels_array : np.array
        Label - array of 0/1
    preds_array : np.array
        Predictions - array of float between 0 and 1

    Returns
    -------
    float
        Threshold
    """
    best_threshold = 0
    best_fpr = 1
    for depth in [0.1, 0.01, 0.001]:
        thresholds = [best_threshold + depth * 10 - j * depth for j in range(1, 10)]
        fprs = [get_fpr(labels_array, preds_array >= thres) for thres in thresholds]
        thresholds += [best_threshold]
        fprs += [best_fpr]
        i = 0
        while fprs[i] < target_fpr:
            i += 1
        best_fpr = fprs[i]
        best_threshold = thresholds[i]
    return thresholds[i - 1]


@gin.configurable("GetThresholdScore", denylist=["state"])
class GetThresholdScore(StatefulPipelineStage):

    name = "Get Threshold Score"

    def __init__(self, state: PipelineState, num_workers: int = 1, **kwargs):
        """_summary_

        Parameters
        ----------
        state : PipelineState
            Pipeline state
        num_workers : int, optional
            Number of workers, by default 1
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        split_name_threshold = self.state.name_threshold.rsplit("_", 1)
        self.metric_name = split_name_threshold[0]
        self.target_metric = float(split_name_threshold[1])

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains threshold

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        return hasattr(self.state, "threshold") and self.state.threshold is not None

    def run(self):
        """Given a target metric value, compute the corresponding threshold on prediction.

        Raises
        ------
        ValueError
            Unsupported metric
        """
        labels_array, preds_array = filter_predictions_pid(
            self.state.predictions, list(self.state.predictions.keys())
        )
        if self.metric_name == "event_recall":
            self.state.threshold = find_threshold_event_recall(
                self.target_metric,
                self.state.predictions,
                list(self.state.predictions.keys()),
                self.state.event_bounds,
                self.state.horizon,
                self.state.max_len,
                self.state.timestep,
            )
        elif self.metric_name == "precision" or self.metric_name == "recall":
            self.state.threshold = find_threshold_precision_recall(
                self.target_metric,
                self.metric_name,
                labels_array,
                preds_array,
            )
        elif self.metric_name == "npv":
            self.state.threshold = find_threshold_npv(self.target_metric, labels_array, preds_array)
        elif self.metric_name == "fpr":
            self.state.threshold = find_threshold_fpr(self.target_metric, labels_array, preds_array)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported metric for threshold computation: {self.metric_name}"
            )
        logging.info(f"[{self.__class__.__name__}] Threshold on score: {self.state.threshold}")
