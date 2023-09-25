METRICS_LOW = ["fpr", "negative_class_score", "calibration_error"]

ERROR_BOX = {
    "calibration": {"xy": (0.01, 0.95), "alignment": (0, 0.95)},
    "auroc": {"xy": (0.7, 0), "alignment": (0, 0)},
    "auprc": {"xy": (0.02, 0), "alignment": (0, 0)},
}

PRETTY_NAME_AXIS = {
    "prob_true": "Fraction of positives",
    "prob_pred": "Mean predicted probability",
    "tpr": "True positive rate",
    "fpr": "False positive rate",
    "precision": "Precision",
    "recall": "Recall",
    "event_recall": "Event-based recall",
    "corrected_precision": "Corrected precision",
}
