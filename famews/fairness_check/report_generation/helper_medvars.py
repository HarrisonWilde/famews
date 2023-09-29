import logging
from typing import Dict, List, Tuple

import pandas as pd

from famews.fairness_check.medical_variables.build_table import (
    generate_table_binary_group,
    generate_table_multicat_group,
)


def get_stat_test_table_medvars(
    medvars_group_df: pd.DataFrame,
    type_table_groups: Dict[str, List],
    group_name: str,
    cats: List[str],
    list_medvars: List[str],
    var_suffixes: List[str],
    worst_cats_medvars: Dict[str, list],
    significance_level: float,
    filter_delta: float,
) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """_summary_

    Parameters
    ----------
    medvars_group_df : pd.DataFrame
        Dataframe of median medical variables
    type_table_groups : Dict[str, List]
        Dictionary mapping comparison type to list of groupings
    group_name : str
        Group name
    cats : List[str]
        List of categories
    list_medvars : List[str]
        List of medical variables
    var_suffixes: List[str]
        List of suffixes
    worst_cats_medvars : Dict[str, list]
        Dictionary mapping a medical variable to the list of categories and corresponding deltas
    significance_level : float
        Significance level
    filter_delta : float
        Filter on delta

    Returns
    -------
    pd.DataFrame
        dataframe with result of statistical test (p-value, delta)
    Dict[str, list]
        Updated worst_cats_medvars with the considered categories
    """
    if group_name in type_table_groups["binary_group"]:
        medvars_df = medvars_group_df[medvars_group_df["group"] == group_name].rename(
            columns={"cat": group_name}
        )
        table, worst_cats = generate_table_binary_group(
            medvars_df,
            list_medvars,
            var_suffixes,
            group_name,
            cats[0],
            cats[1],
            significance_level,
            filter_delta,
        )
    elif group_name in type_table_groups["multicat_group"]:
        medvars_df = medvars_group_df[
            (medvars_group_df["group"] == group_name)
            | (medvars_group_df["group"] == f"{group_name}_bool")
        ].rename(columns={"cat": group_name})
        table, worst_cats = generate_table_multicat_group(
            medvars_df,
            list_medvars,
            var_suffixes,
            group_name,
            cats,
            significance_level,
            filter_delta,
        )
    else:
        logging.warning(
            f"[Medical Variables] Group {group_name} has no table type to generate the statistical test table."
        )
        return pd.DataFrame(), worst_cats_medvars
    return table, {
        (var, suffix): worst_cats_medvars[(var, suffix)] + worst_cats[(var, suffix)]
        for var in list_medvars
        for suffix in var_suffixes
    }
