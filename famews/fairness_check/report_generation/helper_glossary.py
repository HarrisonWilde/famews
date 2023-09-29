from typing import Dict, List, Tuple

from reportlab.platypus import Paragraph

from famews.fairness_check.utils.helper_format import get_pretty_name_metric

METRICS_DEF = {
    "Recall": "<i>TP/P</i>",
    "Precision": "<i>TP/(TP+FP)</i>",
    "NPV": "Negative predictive value, <i>TN/(TN+FN)</i>",
    "FPR": "False positive rate, <i>FP/(FP+TN)</i>",
    "TPR": "True positive rate, <i>TP/(TP+FN)</i>",
    "Corrected precision": "Precision corrected for the cohort prevalence of positive labels, <i>TP/(TP+s*FP)</i> with <i>s</i> the correcting factor that depends on the cohort prevalence and the maximum prevalence for the grouping.",
    "Corrected NPV": "NPV corrected for the cohort prevalence of positive labels, <i>TN/(TN+s*FN)</i> with <i>s</i> the correcting factor that depends on the cohort prevalence and the minimum prevalence for the grouping.",
    "Event-based recall": "Number of detected events over the total number of events.",
    "Avg. score on positive class": "for all positive labels, average of the output scores.",
    "Avg. score on negative class": "for all negative labels, average of the output scores.",
    "ROC curve": "Receiver operating characteristic curve, x-axis FPR, y-axis TPR.",
    "PR curve": "Precision-recall curve, x-axis recall, y-axis precision. It can be drawn also for event-based recall and corrected precision.",
    "AUROC": "Area under the ROC curve.",
    "AUPRC": "Area under the PR curve. It can be computed for the PR curve drawn with event-based recall and/or corrected precision.",
    "Calibration curve": "Illustrates how well the probabilistic predictions of the model are calibrated (whether they can be interpreted as true probabilities), x-axis mean predicted probabilities, y-axis frequency of positive labels. The perfect calibration line (dashed line in the figures) acts as a reference.",
    "Calibration error": "Area between the calibration curve and the perfect calibration line.",
}


def generate_glossary_general() -> list:
    story = []
    story.append(
        Paragraph(
            "<b>Event:</b> Failure or more generally health condition that the model aims to predict. We assume that it has some duration."
        )
    )
    story.append(
        Paragraph(
            "<b>Grouping / Group name:</b> This refers to an attribute used to form the cohorts of patients."
        )
    )
    story.append(
        Paragraph(
            "<b>Category:</b> (abbreviation: <b>Cat.</b>) This refers to the value taken by the grouping attribute, it characterizes a specific cohort. It can also be used to directly designate a cohort."
        )
    )
    story.append(
        Paragraph(
            "<b>Cohort:</b> This is used to designate a particular category of patients (i.e. a set of patients that share a common grouping attribute value)."
        )
    )
    story.append(
        Paragraph(
            "<b>Macro-average:</b> Consider a grouping with <i>n</i> categories, and each category <i>i</i> has a metric value <i>m_i</i>, then the macro-average is <i>(m_1 + m_2 + ... + m_n)/n</i>."
        )
    )
    story.append(
        Paragraph(
            "<b>Delta:</b> (abbreviation: <b>Δ</b>) Each stage is associated with certain metrics, the delta for a metric and a cohort corresponds to the absolute difference in median metric between patients of this cohort and the rest of the patients."
        )
    )
    story.append(
        Paragraph(
            "<b>Threshold on score:</b> Binary classifier outputs probability between 0 and 1, to obtain a binary output the user has to decide on a threshold value below which the output class will be 0 and above which it will be 1."
        )
    )
    return story


def generate_glossary_model_performance(list_metrics: List[str]) -> list:
    story = []
    story.append(Paragraph("<b>Metrics Definitions:</b>"))
    story.append(
        Paragraph(
            (
                "<b>P</b> number of positive labels, <b>N</b> number of negative labels, <b>TP</b> number of correctly predicted positive labels, <b>TN</b> number of correctly predicted negative labels, <b>FP</b> number of instances with true negative labels but that were incorrectly predicted as positive by the model, <b>FN</b> number of instances with true positive labels but that were incorrectly predicted as negative by the model."
            )
        )
    )
    story.append(Paragraph("<b>↑</b>: Means that the larger the metric value, the better it is."))
    story.append(Paragraph("<b>↓</b>: Means that the lower the metric value, the better it is."))
    already_auprc = False
    for metric in list_metrics:
        metric_story, already_auprc = add_metric_glossary(metric, already_auprc)
        story += metric_story
    story.append(
        Paragraph(
            "<b>Ratio of significantly worse metrics:</b> For a specific category of patients, it refers to the number of metrics for which the category is significantly worse off compared to the rest of the population divided by the total number of metrics."
        )
    )
    story.append(
        Paragraph(
            "<b>Worst ratio:</b> Refers to the largest <b>ratio of significantly worse metrics</b> (for a grouping or for the overall analysis)."
        )
    )
    story.append(
        Paragraph(
            "<b>Worst delta:</b> Refers to the largest <b>delta</b> in performance metrics (for a grouping or for the overall analysis)."
        )
    )
    return story


def generate_glossary_timegap() -> list:
    story = []
    story.append(
        Paragraph(
            "<b>Time gap:</b> Amount of time between the trigger of the first correct alarm and the event occurence."
        )
    )
    story.append(
        Paragraph(
            "<b>Start event:</b> Considered split of the alarm horizon. We split the alarm horizon into different windows (chosen by the user) based on how much time in advance the alarm can be triggered. The available prediction horizon can not be longer than the time between the start of the considered event and the start of the stay or between the start of the considered event and the time when the previous event finished."
        )
    )
    return story


def generate_glossary_medvars() -> list:
    story = []
    story.append(
        Paragraph(
            "<b>Not in event:</b> Refers to the median value computed on time points when patients aren't undergoing an event."
        )
    )
    story.append(
        Paragraph(
            "<b>Never in event:</b> Refers to the median value computed for patients without any event during their stay."
        )
    )
    return story


def generate_glossary_feature_importance(k: int) -> list:
    story = []
    story.append(
        Paragraph(
            "<b>Feature importance:</b> Approximates how useful is a feature for the prediction task. We use SHAP values to estimate it."
        )
    )
    story.append(
        Paragraph(
            "<b>RBO (Rank-biased overlap):</b> Similarity measure between two lists that focuses more on the head of the list (i.e it penalizes more mismatches that occur at the beginning). We use this measure to compare two feature rankings."
        )
    )
    story.append(
        Paragraph(
            "<b>General feature ranking:</b> Refers to the ranking of features based on their importance (from the most important to the least important), obtained on the entire set of patients. In contrast to cohort-based rankings, that are obtained on a specific cohort of patients."
        )
    )
    story.append(
        Paragraph(
            "<b>Delta of inverse rank:</b> For a feature that has rank <i>rk_0</i> in the cohort-based ranking and <i>rk_all</i> in the general ranking, it is defined as <i>|1/rk_0 - 1/rk_all|</i>. If it is big enough, we consider the change in rank of the feature from the general to the cohort-based ranking to be significant."
        )
    )
    story.append(
        Paragraph(
            f"<b>Top {k} (cohort)</b>: refers to the first {k} features of the general (or cohort-based) ranking."
        )
    )
    return story


def generate_glossary_missingness(
    intensity_msrt_cats: Dict[str, Tuple[float, float]],
    list_metrics_missingness: List[str],
    list_metrics_performance: List[str],
) -> list:
    story = []
    story.append(Paragraph("<b>Performance metrics definitions:</b>"))
    have_unseen_metric = False
    already_auprc = False
    for metric in list_metrics_missingness:
        if metric not in list_metrics_performance:
            metric_story, already_auprc = add_metric_glossary(metric, already_auprc)
            story += metric_story
            have_unseen_metric = True
    if not have_unseen_metric:
        story.append(
            Paragraph(
                "All metrics have already been defined in the <b>Model Performance Analysis concepts</b>."
            )
        )
    story.append(Paragraph("<b>Intensity of measurement categories</b>"))
    story.append(
        Paragraph("<i>no_msrt</i>: Refers to patients without any measurement for a variable.")
    )
    for cat, (lower, upper) in intensity_msrt_cats.items():
        if cat == "no_msrt":
            continue
        else:
            story.append(
                Paragraph(
                    f"<i>{cat}</i>: Refers to patients with between {int(lower*100)}% (not included) and {int(upper*100)}% of valid measurements (over the number of expected measurements)."
                )
            )
        story.append(
            Paragraph(
                "The number of expected measurements is computed from the medical variable's expected sampling interval <i>t_e</i> (input from the user) and the patient's length of stay <i>los</i> as <i>los</i> / <i>t_e</i>."
            )
        )
    story.append(Paragraph("<b>Missingness categories:</b>"))
    story.append(
        Paragraph(
            "<i>no_msrt</i>: Refers to patients without any measurement for a variable (before full data imputation)."
        )
    )
    story.append(
        Paragraph(
            "<i>missing_msrt</i>: Refers to data points without valid measurement for a variable (before full data imputation but after forward propagation of measurements based on the variable's expected sampling interval)."
        )
    )
    story.append(
        Paragraph(
            "<i>with_msrt</i>: Refers to data points with valid measurements for a variable (before full data imputation but after forward propagation of measurements based on the variable's expected sampling interval)."
        )
    )
    story.append(
        Paragraph(
            "<b>Dependent/Independent:</b> Refers to the result of the Chi-squared independence test."
        )
    )
    return story


def add_metric_glossary(metric: str, already_auprc: bool) -> list:
    story = []
    pretty_name_metric = get_pretty_name_metric(metric)
    if "auroc" in metric:
        story.append(Paragraph(f"<b>ROC curve</b>: {METRICS_DEF['ROC curve']}"))
        story.append(Paragraph(f"<b>AUROC</b>: {METRICS_DEF['AUROC']}"))
    elif "calibration_error" in metric:
        story.append(Paragraph(f"<b>Calibration curve</b>: {METRICS_DEF['Calibration curve']}"))
        story.append(Paragraph(f"<b>Calibration error</b>: {METRICS_DEF['Calibration error']}"))
    elif "auprc" in metric:
        if not already_auprc:
            story.append(Paragraph(f"<b>PR curve</b>: {METRICS_DEF['PR curve']}"))
            story.append(Paragraph(f"<b>AUPRC</b>: {METRICS_DEF['AUPRC']}"))
            already_auprc = True
    elif "↑" in pretty_name_metric or "↓" in pretty_name_metric:
        pretty_name_metric = pretty_name_metric[:-2]
        story.append(Paragraph(f"<b>{pretty_name_metric}</b>: {METRICS_DEF[pretty_name_metric]}"))
    else:
        story.append(Paragraph(f"<b>{pretty_name_metric}</b>: {METRICS_DEF[pretty_name_metric]}"))
    return story, already_auprc
