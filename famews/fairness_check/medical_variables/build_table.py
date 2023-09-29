from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from famews.fairness_check.utils.helper_format import get_pretty_name_medvars
from famews.fairness_check.utils.stat_tests import run_mann_whitney_u


def get_delta(
    df: pd.DataFrame, medvar: str, hue: str, group1: Union[str, bool], group2: Union[str, bool]
) -> float:
    """Compute the absolute difference between the median medical variable for group1 and for group2.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the metrics for each group
    medvar : str
        Medical variable name
    hue : str
        Grouping name
    group1 : Union[str, bool]
        Name first category
    group2 : Union[str, bool]
        Name second category

    Returns
    -------
    float
        delta
    """
    median_medvar = df[[hue, medvar]].groupby(hue).median()[medvar]
    return np.abs(median_medvar[group1] - median_medvar[group2])


def generate_table_binary_group(
    medvars_df: pd.DataFrame,
    list_medvars: List[str],
    var_suffixes: List[str],
    hue_group: str,
    group1: Union[str, bool],
    group2: Union[str, bool],
    significance_level: float = 0.001,
    filter_delta: float = 0,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, float]]]:
    """Generate table presenting the results of the statistical test and delta analysis for a binary grouping.

    Parameters
    ----------
    medvars_df : pd.DataFrame
        Dataframe containing median medical variables for each cohort that we want to compare
    list_medvars : List[str]
        List of medical variables
    var_suffixes: List[str]
        List of suffixes
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
        Table containing results of statistical tests and delta of medical variables
    Dict[str, Tuple[str, float]]
        Dictionary mapping each medical variable to the category and corresponding delta, if the medical variable median value was significantly different for this category
    """
    alpha = significance_level / (len(list_medvars) * len(var_suffixes) * 2)
    pval_dict = {
        "Medical Variable": [],
        "Cohort with greater median value": [],
        "P-value": [],
        "Delta": [],
    }
    worst_cats = {
        (var, suffix): [] for var in list_medvars for suffix in var_suffixes
    }  # (cat, delta)
    for var in list_medvars:
        for suffix in var_suffixes:
            medvar = var + suffix
            pretty_name_medvar = get_pretty_name_medvars(var, suffix)
            pvalue_1 = run_mann_whitney_u(
                medvars_df, medvar, hue_group, group1, group2, hyp="greater"
            ).pvalue
            pvalue_2 = run_mann_whitney_u(
                medvars_df, medvar, hue_group, group2, group1, hyp="greater"
            ).pvalue
            delta = get_delta(medvars_df, medvar, hue_group, group1, group2)
            if pvalue_1 < alpha and delta > filter_delta:
                pval_dict["Medical Variable"].append(pretty_name_medvar)
                pval_dict["Cohort with greater median value"].append(group1)
                pval_dict["P-value"].append(np.format_float_scientific(pvalue_1, precision=2))
                pval_dict["Delta"].append(round(delta, 3))
                worst_cats[(var, suffix)].append((hue_group, delta))
            elif pvalue_2 < alpha and delta > filter_delta:
                pval_dict["Medical Variable"].append(pretty_name_medvar)
                pval_dict["Cohort with greater median value"].append(group2)
                pval_dict["P-value"].append(np.format_float_scientific(pvalue_2, precision=2))
                pval_dict["Delta"].append(round(delta, 3))
                worst_cats[(var, suffix)].append((hue_group, delta))
    return pd.DataFrame(pval_dict), worst_cats


def generate_table_multicat_group(
    medvars_df: pd.DataFrame,
    list_medvars: List[str],
    var_suffixes: List[str],
    hue_group: str,
    cats_group: List[str],
    significance_level: float = 0.001,
    filter_delta: float = 0,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, float]]]:
    """Generate table presenting the results of the statistical test and delta analysis for a grouping with more than 2 categories.

    Parameters
    ----------
    medvars_df : pd.DataFrame
        Dataframe containing median medical variables for each cohort that we want to compare
    list_medvars : List[str]
        List of medical variables
    var_suffixes: List[str]
        List of suffixes
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
        Table containing results of statistical tests and delta of medical variables
    Dict[str, Tuple[str, float]]
        Dictionary mapping each medical variable to the category and corresponding delta, if the medical variable median value was significantly different for this category
    """
    pval_dict = {
        "Medical Variable": [],
        "Category": [],
        "Cohort vs. rest": [],
        "P-value": [],
        "Delta": [],
    }
    alpha = significance_level / (len(list_medvars) * len(var_suffixes) * 2 * len(cats_group))
    worst_cats = {
        (var, suffix): [] for var in list_medvars for suffix in var_suffixes
    }  # (cat, delta)
    for var in list_medvars:
        for suffix in var_suffixes:
            medvar = var + suffix
            pretty_name_medvar = get_pretty_name_medvars(var, suffix)
            for cat in cats_group:
                pvalue_greater = run_mann_whitney_u(
                    medvars_df, medvar, hue_group, cat, f"Not {cat}", hyp="greater"
                ).pvalue
                pvalue_less = run_mann_whitney_u(
                    medvars_df, medvar, hue_group, cat, f"Not {cat}", hyp="less"
                ).pvalue
                delta = get_delta(medvars_df, medvar, hue_group, cat, f"Not {cat}")
                if pvalue_greater < alpha and delta > filter_delta:
                    pval_dict["Medical Variable"].append(pretty_name_medvar)
                    pval_dict["Category"].append(cat)
                    pval_dict["Cohort vs. rest"].append("greater")
                    pval_dict["P-value"].append(
                        np.format_float_scientific(pvalue_greater, precision=2)
                    )
                    pval_dict["Delta"].append(round(delta, 3))
                    worst_cats[(var, suffix)].append((cat, delta))
                elif pvalue_less < alpha and delta > filter_delta:
                    pval_dict["Medical Variable"].append(pretty_name_medvar)
                    pval_dict["Category"].append(cat)
                    pval_dict["Cohort vs. rest"].append("less")
                    pval_dict["P-value"].append(
                        np.format_float_scientific(pvalue_less, precision=2)
                    )
                    pval_dict["Delta"].append(round(delta, 3))
                    worst_cats[(var, suffix)].append((cat, delta))
    return pd.DataFrame(pval_dict), worst_cats


def get_worst_medvars_delta(
    worst_cats_medvars: Dict[str, List[Tuple[str, float]]],
    k: int = 3,
    medvars_units: Dict[str, str] = {},
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """For each medical variable, get the top k worst performing categories (high delta and significantly different distribution).

    Parameters
    ----------
    worst_cats_medvars : Dict[str, List[Tuple[str, float]]]
        Dictionary mapping each medical variable to a lsit of categories and the corresponding performance delta {medvar: [(category, delta)]}
    k : int, optional
        Number of categories to return per medical variable, by default 3
    medvars_units: Dict[str, str], optional
        Dictionary storing the units of the medical variables (only for display purpose), by default {}

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, List[str]]]
        Data frame displaying the top k worst categories per medical variable, dictionary mapping each medical variable  to the top k worst categories
    """
    res = []
    dict_worst_cats = {}
    for medvar, list_cats in worst_cats_medvars.items():
        list_cats.sort(key=lambda t: t[1], reverse=True)
        worst_cats_res = [f"{cat} ({round(delta, 3)})" for (cat, delta) in list_cats[:k]]
        dict_worst_cats[medvar] = [cat for (cat, _) in list_cats[:k]]
        if len(worst_cats_res) < k:
            worst_cats_res += ["-" for _ in range(k - len(worst_cats_res))]
        if medvar[0] in medvars_units:
            res.append(
                [f"{get_pretty_name_medvars(*medvar)} ({medvars_units[medvar[0]]})"]
                + worst_cats_res
            )
        else:
            res.append([f"{get_pretty_name_medvars(*medvar)}"] + worst_cats_res)
    return (
        pd.DataFrame(res, columns=["Medical Variable"] + [f"Cohort {i+1} (Δ)" for i in range(k)]),
        dict_worst_cats,
    )
