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
        Dataframe containing the metrics for each missingness category
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


def generate_table_perf_missingness(
    metrics_df: pd.DataFrame,
    list_metrics: List[str],
    other_missigness_cats: List[str],
    significance_level: float = 0.001,
    filter_delta: float = 0,
) -> pd.DataFrame:
    """Generate table presenting the results of the statistical test and delta analysis for the impact on
    performance of missingness.
    We compare the categories in other_missingness_cats against the with_msrt category.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe containing metrics for each missingness category
    list_metrics : List[str]
        List of metrics that we want to compare
    other_missingness_cats: List[str]
        List of missingness categories, other than 'with_msrt'
    significance_level : float, optional
        Signifiance level for the statistical test (before correction), by default 0.001
    filter_delta: float, optional
        Minimum delta value to display the results of the stat test in the table, by default 0

    Returns
    -------
    pd.DataFrame
        Table containing results of statistical tests and delta of metrics
    """
    alpha = significance_level / (len(list_metrics) * 2)
    pval_dict = {
        "Metric": [],
        "Missingness category": [],
        "Category vs with msrt": [],
        "P-value": [],
        "Delta": [],
    }
    # worst_cats = {metric: [] for metric in list_metrics}  # (cat, delta)
    group_ref = "with_msrt"
    for metric in list_metrics:
        pretty_name_metric = get_pretty_name_metric(metric)
        for missingness_cat in other_missigness_cats:
            pvalue_1 = run_mann_whitney_u(
                metrics_df, metric, "cat", group_ref, missingness_cat, hyp="greater"
            ).pvalue
            pvalue_2 = run_mann_whitney_u(
                metrics_df, metric, "cat", missingness_cat, group_ref, hyp="greater"
            ).pvalue
            delta = get_delta(metrics_df, metric, "cat", group_ref, missingness_cat)
            if pvalue_1 < alpha and delta > filter_delta:
                pval_dict["Metric"].append(pretty_name_metric)
                pval_dict["Missingness category"].append(missingness_cat)
                pval_dict["P-value"].append(np.format_float_scientific(pvalue_1, precision=2))
                pval_dict["Delta"].append(round(delta, 3))
                if sum(
                    metric.startswith(m) for m in METRICS_LOW
                ):  # check whether it is preferable for the metric to be low or high
                    # worst_cats[metric].append((group1, delta))
                    pval_dict["Category vs with msrt"].append("better")
                else:
                    # worst_cats[metric].append((group2, delta))
                    pval_dict["Category vs with msrt"].append("worse")
            elif pvalue_2 < alpha and delta > filter_delta:
                pval_dict["Metric"].append(pretty_name_metric)
                pval_dict["Missingness category"].append(missingness_cat)
                pval_dict["P-value"].append(np.format_float_scientific(pvalue_2, precision=2))
                pval_dict["Delta"].append(round(delta, 3))
                if sum(
                    metric.startswith(m) for m in METRICS_LOW
                ):  # check whether it is preferable for the metric to be low or high
                    # worst_cats[metric].append((group2, delta))
                    pval_dict["Category vs with msrt"].append("worse")
                else:
                    # worst_cats[metric].append((group1, delta))
                    pval_dict["Category vs with msrt"].append("better")
    return pd.DataFrame(pval_dict)  # , worst_cats
