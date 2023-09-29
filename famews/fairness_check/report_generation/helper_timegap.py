import logging
from typing import Dict, List, Tuple

import pandas as pd

from famews.fairness_check.timegap_alarm_event.build_table import (
    generate_table_binary_group,
    generate_table_multicat_group,
)


def get_stat_test_table_timegap(
    timegap_group_df: pd.DataFrame,
    type_table_groups: Dict[str, List[str]],
    group_name: str,
    cats: List[str],
    list_group_start_event: List[str],
    worst_cats_timegap: Dict[str, list],
    significance_level: float,
    filter_delta: float,
) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """Run statistical test for time gap.

    Parameters
    ----------
    timegap_group_df : pd.DataFrame
        Dataframe time gap
    type_table_groups : Dict[str, List[str]]
        Dictionary mapping each type of comparison to a list of groupings
    group_name : str
        Group name
    cats : List[str]
        List of categories
    list_group_start_event : List[str]
        List of group of start event window
    worst_cats_timegap : Dict[str, list]
        Dictionary mapping each group_start_event to the list of categories and delta
    significance_level : float
        Significance level
    filter_delta : float
        Filter on delta

    Returns
    -------
    pd.DataFrame
        Dataframe of statistical test results
    Dict[str, list]]
        worst_cats_timegap with the delta for new categories
    """
    if group_name in type_table_groups["binary_group"]:
        timegap_df = timegap_group_df[timegap_group_df["group"] == group_name].rename(
            columns={"cat": group_name}
        )
        table, worst_cats = generate_table_binary_group(
            timegap_df,
            list_group_start_event,
            group_name,
            cats[0],
            cats[1],
            significance_level,
            filter_delta,
        )
    elif group_name in type_table_groups["multicat_group"]:
        timegap_df = timegap_group_df[
            (timegap_group_df["group"] == group_name)
            | (timegap_group_df["group"] == f"{group_name}_bool")
        ].rename(columns={"cat": group_name})
        table, worst_cats = generate_table_multicat_group(
            timegap_df,
            list_group_start_event,
            group_name,
            cats,
            significance_level,
            filter_delta,
        )
    else:
        logging.warning(
            f"[Model Performance] Grouping {group_name} has no table type to generate the statistical test table."
        )
        return pd.DataFrame(), worst_cats_timegap
    return table, {gp: worst_cats_timegap[gp] + worst_cats[gp] for gp in list_group_start_event}


def get_table_summary_timegap_group(
    timegap_df: pd.DataFrame, list_group_start_event: List[str], cats_size: Dict[str, int]
) -> pd.DataFrame:
    res = {
        "Start event": [],
        "Macro-average (in minutes)": [],
        "Minimum (category)": [],
        "For minority category": [],
    }
    for gp in list_group_start_event:
        df = timegap_df[timegap_df["group_start_event"] == gp].groupby("cat")["time_gap"].median()
        avg_timegap = df.mean()
        min_timegap = df.min()
        min_cat = df.index[df.argmin()]
        minority_cat = min(cats_size, key=cats_size.get)
        minority_timegap = df[minority_cat]
        res["Start event"].append(gp)
        res["Macro-average (in minutes)"].append(round(avg_timegap, 3))
        res["Minimum (category)"].append(f"{round(min_timegap, 3)} ({min_cat})")
        res["For minority category"].append(round(minority_timegap, 3))
    return pd.DataFrame(res)


def get_summary_timegap_mean(timegap_df_group: pd.DataFrame, group_start_event: str) -> str:
    timegap_window = timegap_df_group[timegap_df_group["group_start_event"] == group_start_event]
    mean_timegap = timegap_window.groupby("cat")["time_gap"].median().mean()
    return f"For event starting in the window {group_start_event}, the overall macro-averaged median time gap is {round(mean_timegap, 1)} (in minutes)."
