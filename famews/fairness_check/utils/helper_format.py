PRETTY_NAME_METRICS_SCORE = {
    "positive_class_score": "Avg. score on positive class",
    "negative_class_score": "Avg. score on negative class",
    "auroc": "AUROC ↑",
    "auprc": "AUPRC ↑",
    "corrected_auprc": "Corrected AUPRC ↑",
    "event_auprc": "Event-based AUPRC ↑",
    "corrected_event_auprc": "Corrected event-based AUPRC ↑",
    "calibration_error": "Calibration error ↓",
}


def get_pretty_name_metric(metric_name: str) -> str:
    """Return name of metric with a nice format to be displayed

    Parameters
    ----------
    metric_name : str
        metric name

    Returns
    -------
    str
        metric name ready to be displayed
    """
    if metric_name.startswith("precision"):
        return "Precision ↑"
    if metric_name.startswith("recall"):
        return "Recall ↑"
    if metric_name.startswith("corrected_precision"):
        return "Corrected precision ↑"
    if metric_name.startswith("npv"):
        return "NPV ↑"
    if metric_name.startswith("corrected_npv"):
        return "Corrected NPV ↑"
    if metric_name.startswith("fpr"):
        return "FPR ↓"
    if metric_name.startswith("event_recall"):
        return "Event-based recall ↑"
    if metric_name in PRETTY_NAME_METRICS_SCORE:
        return PRETTY_NAME_METRICS_SCORE[metric_name]
    return metric_name


def display_name_metric_in_sentence(metric_name: str):
    """Return name of metric with a nice format to be displayed within a sentence.
    To this end, we remove the arrow at the end of the string if there is any.

    Parameters
    ----------
    metric_name : str
        metric name

    Returns
    -------
    str
        metric name ready to be displayed
    """
    if metric_name[-1] == "↑" or metric_name[-1] == "↓":
        return metric_name[:-2]
    return metric_name


def get_pretty_name_medvars(medvar: str, suffix: str) -> str:
    if suffix == "":
        return medvar
    if suffix == "_not_inevent":
        return medvar + " - Not in event"
    if suffix == "_never_inevent":
        return medvar + " - Never in event"
