from typing import Dict, List, Union

import numpy as np
import pandas as pd

from famews.fairness_check.utils.stat_tests import run_mann_whitney_u


def get_delta(
    df: pd.DataFrame, hue: str, group1: Union[str, bool], group2: Union[str, bool]
) -> float:
    """Compute the absolute difference between the median time gap for group1 and for group2.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the metrics for each cohort
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
    median_timegap = df[[hue, "time_gap"]].groupby(hue).median()["time_gap"]
    return np.abs(median_timegap[group1] - median_timegap[group2])


def generate_table_binary_group(
    timegap_df: pd.DataFrame,
    list_group_start_event: List[str],
    hue_group: str,
    group1: Union[str, bool],
    group2: Union[str, bool],
    significance_level: float = 0.001,
    filter_delta: float = 0,
) -> pd.DataFrame:
    """Generate table presenting the results of the statistical test and delta analysis for a binary grouping.

    Parameters
    ----------
    timegap_df : pd.DataFrame
        Dataframe containing time gap for each cohort that we want to compare
    list_group_start_event : List[str]
        List of groups for start of event that we want to compare
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
    """
    alpha = significance_level / (len(list_group_start_event) * 2)
    pval_dict = {
        "Start event": [],
        "Cohort with the worst time gap": [],
        "P-value": [],
        "Delta (in minutes)": [],
    }
    worst_cats = {gp: [] for gp in list_group_start_event}  # (cat, delta)
    for gp in list_group_start_event:
        timegap_gp_df = timegap_df[timegap_df["group_start_event"] == gp]
        contains_group1 = (timegap_gp_df[hue_group] == group1).any()
        contains_group2 = (timegap_gp_df[hue_group] == group2).any()
        if contains_group1 and contains_group2:
            pvalue_1 = run_mann_whitney_u(
                timegap_gp_df, "time_gap", hue_group, group1, group2, hyp="greater"
            ).pvalue
            pvalue_2 = run_mann_whitney_u(
                timegap_gp_df, "time_gap", hue_group, group2, group1, hyp="greater"
            ).pvalue
            delta = get_delta(timegap_gp_df, hue_group, group1, group2)
        else:
            pvalue_1 = 1
            pvalue_2 = 1
            delta = 0
        if pvalue_1 < alpha and delta > filter_delta:
            pval_dict["Start event"].append(gp)
            pval_dict["Cohort with the worst time gap"].append(group2)
            pval_dict["P-value"].append(np.format_float_scientific(pvalue_1, precision=2))
            pval_dict["Delta (in minutes)"].append(round(delta, 3))
            worst_cats[gp].append((group2, delta))
        elif pvalue_2 < alpha and delta > filter_delta:
            pval_dict["Start event"].append(gp)
            pval_dict["Cohort with the worst time gap"].append(group1)
            pval_dict["P-value"].append(np.format_float_scientific(pvalue_2, precision=2))
            pval_dict["Delta (in minutes)"].append(round(delta, 3))
            worst_cats[gp].append((group1, delta))
    return pd.DataFrame(pval_dict), worst_cats


def generate_table_multicat_group(
    timegap_df: pd.DataFrame,
    list_group_start_event: List[str],
    hue_group: str,
    cats_group: List[str],
    significance_level: float = 0.001,
    filter_delta: float = 0,
) -> pd.DataFrame:
    """Generate table presenting the results of the statistical test and delta analysis for a grouping with more than 2 categories.

    Parameters
    ----------
    timegap_df : pd.DataFrame
        Dataframe containing time gap for each cohort that we want to compare
    list_group_start_event : List[str]
        List of groups for start of event that we want to compare
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
    """
    pval_dict = {
        "Start event": [],
        "Category": [],
        "Cohort vs. rest": [],
        "P-value": [],
        "Delta (in minutes)": [],
    }
    alpha = significance_level / (len(list_group_start_event) * 2 * len(cats_group))
    worst_cats = {gp: [] for gp in list_group_start_event}  # (cat, delta)
    for gp in list_group_start_event:
        timegap_gp_df = timegap_df[timegap_df["group_start_event"] == gp]
        for cat in cats_group:
            contains_cat = (timegap_gp_df[hue_group] == cat).any()
            contains_not_cat = (timegap_gp_df[hue_group] == f"Not {cat}").any()
            if contains_cat and contains_not_cat:
                pvalue_greater = run_mann_whitney_u(
                    timegap_gp_df, "time_gap", hue_group, cat, f"Not {cat}", hyp="greater"
                ).pvalue
                pvalue_less = run_mann_whitney_u(
                    timegap_gp_df, "time_gap", hue_group, cat, f"Not {cat}", hyp="less"
                ).pvalue
                delta = get_delta(timegap_gp_df, hue_group, cat, f"Not {cat}")
            else:
                pvalue_greater = 1
                pvalue_less = 1
                delta = 0
            if pvalue_greater < alpha and delta > filter_delta:
                pval_dict["Start event"].append(gp)
                pval_dict["Category"].append(cat)
                pval_dict["Cohort vs. rest"].append("better")
                pval_dict["P-value"].append(np.format_float_scientific(pvalue_greater, precision=2))
                pval_dict["Delta (in minutes)"].append(round(delta, 3))
            elif pvalue_less < alpha and delta > filter_delta:
                pval_dict["Start event"].append(gp)
                pval_dict["Category"].append(cat)
                pval_dict["Cohort vs. rest"].append("worse")
                pval_dict["P-value"].append(np.format_float_scientific(pvalue_less, precision=2))
                pval_dict["Delta (in minutes)"].append(round(delta, 3))
                worst_cats[gp].append((cat, delta))
    return pd.DataFrame(pval_dict), worst_cats


def get_worst_timegap(worst_cats_timegap: Dict[str, list], k: int = 3):
    res = []
    dict_worst_cats = {}
    for gp, list_cats in worst_cats_timegap.items():
        list_cats.sort(key=lambda t: t[1], reverse=True)
        worst_cats_res = [f"{cat} ({round(delta, 3)})" for (cat, delta) in list_cats[:k]]
        dict_worst_cats[gp] = [cat for (cat, _) in list_cats[:k]]
        if len(worst_cats_res) < k:
            worst_cats_res += ["-" for _ in range(k - len(worst_cats_res))]
        res.append([gp] + worst_cats_res)
    return (
        pd.DataFrame(
            res,
            columns=["Start event"] + [f"Cohort {i+1} (Δ in minutes)" for i in range(k)],
        ),
        dict_worst_cats,
    )
