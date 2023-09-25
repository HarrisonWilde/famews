from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from famews.fairness_check.model_performance.constants import METRICS_LOW
from famews.fairness_check.utils.helper_format import get_pretty_name_metric
from famews.fairness_check.utils.stat_tests import run_mann_whitney_u


def get_delta(
    df: pd.DataFrame, metric: str, hue: str, group1: Union[str, bool], group2: Union[str, bool]
) -> float:
    """Compute the absolute difference between the median metric for group1 and for group2.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the metrics for each cohort
    metric : str
        metric name
    hue : str
        grouping name
    group1 : Union[str, bool]
        name first category
    group2 : Union[str, bool]
        name second category

    Returns
    -------
    float
        delta
    """
    median_metric = df[[hue, metric]].groupby(hue).median()[metric]
    return np.abs(median_metric[group1] - median_metric[group2])


def generate_table_binary_group(
    metrics_df: pd.DataFrame,
    list_metrics: List[str],
    hue_group: str,
    group1: Union[str, bool],
    group2: Union[str, bool],
    significance_level: float = 0.001,
    filter_delta: float = 0,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, float]]]:
    """Generate table presenting the results of the statistical test and delta analysis for a binary grouping.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe containing metrics for each cohort that we want to compare
    list_metrics : List[str]
        List of metrics that we want to compare
    hue_group : str
        Grouping name
    group1 : Union[str, bool]
        name of first category
    group2 : Union[str, bool]
        name of second category
    significance_level : float, optional
        Signifiance level for the statistical test (before correction), by default 0.001
    filter_delta: float, optional
        Minimum delta value to display the results of the stat test in the table, by default 0

    Returns
    -------
    pd.DataFrame
        Table containing results of statistical tests and delta of metrics
    Dict[str, Tuple[str, float]]
        Dictionary mapping each metric to the category and corresponding delta, if the metric was significantly worse for this category
    """
    alpha = significance_level / (len(list_metrics) * 2)
    pval_dict = {
        "Metric": [],
        "Cohort with the worst metric": [],
        "P-value": [],
        "Delta": [],
    }
    worst_cats = {metric: [] for metric in list_metrics}  # (cat, delta)
    for metric in list_metrics:
        pretty_name_metric = get_pretty_name_metric(metric)
        pvalue_1 = run_mann_whitney_u(
            metrics_df, metric, hue_group, group1, group2, hyp="greater"
        ).pvalue
        pvalue_2 = run_mann_whitney_u(
            metrics_df, metric, hue_group, group2, group1, hyp="greater"
        ).pvalue
        delta = get_delta(metrics_df, metric, hue_group, group1, group2)
        if pvalue_1 < alpha and delta > filter_delta:
            pval_dict["Metric"].append(pretty_name_metric)
            pval_dict["P-value"].append(np.format_float_scientific(pvalue_1, precision=2))
            pval_dict["Delta"].append(round(delta, 3))
            if sum(
                metric.startswith(m) for m in METRICS_LOW
            ):  # check whether it is preferable for the metric to be low or high
                worst_cats[metric].append((group1, delta))
                pval_dict["Cohort with the worst metric"].append(group1)
            else:
                worst_cats[metric].append((group2, delta))
                pval_dict["Cohort with the worst metric"].append(group2)
        elif pvalue_2 < alpha and delta > filter_delta:
            pval_dict["Metric"].append(pretty_name_metric)
            pval_dict["P-value"].append(np.format_float_scientific(pvalue_2, precision=2))
            pval_dict["Delta"].append(round(delta, 3))
            if sum(
                metric.startswith(m) for m in METRICS_LOW
            ):  # check whether it is preferable for the metric to be low or high
                worst_cats[metric].append((group2, delta))
                pval_dict["Cohort with the worst metric"].append(group2)
            else:
                worst_cats[metric].append((group1, delta))
                pval_dict["Cohort with the worst metric"].append(group1)
    return pd.DataFrame(pval_dict), worst_cats


def generate_table_multicat_group(
    metrics_df: pd.DataFrame,
    list_metrics: List[str],
    hue_group: str,
    cats_group: List[str],
    significance_level: float = 0.001,
    filter_delta: float = 0,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, float]]]:
    """Generate table presenting the results of the statistical test and delta analysis for a grouping with more than 2 categories.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe containing metrics for each cohort that we want to compare
    list_metrics : List[str]
        List of metrics that we want to compare
    hue_group : str
        Grouping name
    cats_group : List[str]
        List of categories
    significance_level : float, optional
        Signifiance level for the statistical test (before correction), by default 0.001
    filter_delta: float, optional
        Minimum delta value to display the results of the stat test in the table, by default 0

    Returns
    -------
    pd.DataFrame
        Table containing results of statistical tests and delta of metrics
    Dict[str, Tuple[str, float]]
        Dictionary mapping each metric to the category and corresponding delta, if the metric was significantly worse for this category
    """
    pval_dict = {
        "Metric": [],
        "Category": [],
        "Cohort vs. rest": [],
        "P-value": [],
        "Delta": [],
    }
    alpha = significance_level / (len(list_metrics) * 2 * len(cats_group))
    worst_cats = {metric: [] for metric in list_metrics}  # (cat, delta)
    for metric in list_metrics:
        pretty_name_metric = get_pretty_name_metric(metric)
        for cat in cats_group:
            pvalue_greater = run_mann_whitney_u(
                metrics_df, metric, hue_group, cat, f"Not {cat}", hyp="greater"
            ).pvalue
            pvalue_less = run_mann_whitney_u(
                metrics_df, metric, hue_group, cat, f"Not {cat}", hyp="less"
            ).pvalue
            delta = get_delta(metrics_df, metric, hue_group, cat, f"Not {cat}")
            if pvalue_greater < alpha and delta > filter_delta:
                pval_dict["Metric"].append(pretty_name_metric)
                pval_dict["Category"].append(cat)
                pval_dict["P-value"].append(np.format_float_scientific(pvalue_greater, precision=2))
                pval_dict["Delta"].append(round(delta, 3))
                if sum(
                    metric.startswith(m) for m in METRICS_LOW
                ):  # check whether it is preferable for the metric to be low
                    worst_cats[metric].append((cat, delta))
                    pval_dict["Cohort vs. rest"].append("worse")
                else:
                    pval_dict["Cohort vs. rest"].append("better")
            elif pvalue_less < alpha and delta > filter_delta:
                pval_dict["Metric"].append(pretty_name_metric)
                pval_dict["Category"].append(cat)
                pval_dict["P-value"].append(np.format_float_scientific(pvalue_less, precision=2))
                pval_dict["Delta"].append(round(delta, 3))
                if not sum(
                    metric.startswith(m) for m in METRICS_LOW
                ):  # check whether it is preferable for the metric to be high
                    worst_cats[metric].append((cat, delta))
                    pval_dict["Cohort vs. rest"].append("worse")
                else:
                    pval_dict["Cohort vs. rest"].append("better")
    return pd.DataFrame(pval_dict), worst_cats


def get_worst_metrics_performance(
    worst_cats_metrics: Dict[str, List[Tuple[str, float]]], k: int = 3
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """For each metric, get the top k worst performing categories (high delta and significantly worse metric distribution).

    Parameters
    ----------
    worst_cats_metrics : Dict[str, List[Tuple[str, float]]]
        Dictionary mapping each metric to a lsit of categories and the corresponding performance delta {metric: [(category, delta)]}
    k : int, optional
        Number of categories to return per metric, by default 3

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, List[str]]]
        Data frame displaying the top k worst categories per metric, dictionary mapping each metric  to the top k worst categories
    """
    res = []
    dict_worst_cats = {}
    for metric, list_cats in worst_cats_metrics.items():
        list_cats.sort(key=lambda t: t[1], reverse=True)
        worst_cats_res = [f"{cat} ({round(delta, 3)})" for (cat, delta) in list_cats[:k]]
        dict_worst_cats[metric] = [cat for (cat, _) in list_cats[:k]]
        if len(worst_cats_res) < k:
            worst_cats_res += ["-" for _ in range(k - len(worst_cats_res))]
        res.append([get_pretty_name_metric(metric)] + worst_cats_res)
    return (
        pd.DataFrame(res, columns=["Metric"] + [f"Cohort {i+1} (Δ)" for i in range(k)]),
        dict_worst_cats,
    )
