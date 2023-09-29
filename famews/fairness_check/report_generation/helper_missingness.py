from typing import Dict, List, Tuple

import pandas as pd

from famews.fairness_check.utils.helper_format import display_name_metric_in_sentence
from famews.fairness_check.utils.stat_tests import run_chi_square_independence_test


def get_stat_test_intensity_msrt(
    intensity_msrt_df: pd.DataFrame, var: str, group_name: str, cats: List[str], alpha: float
) -> Tuple[bool, str]:
    """Run the Chi-squared independence test, assessing the dependence between intensify of
    measurement and patient's categories.

    Parameters
    ----------
    intensity_msrt_df : pd.DataFrame
        Intensity of measurement data frame
    var : str
        Variable name
    group_name : str
        Group name
    cats : List[str]
        List of categories
    alpha : float
        Significance level

    Returns
    -------
    Tuple[bool, str]
        whether the intensitfy of measurement is dependent on the grouping, result string
    """
    stat_test_res = run_chi_square_independence_test(
        intensity_msrt_df, var, group_name, filter2=cats
    )
    if stat_test_res.pvalue < alpha:
        return (
            True,
            f"The intensity of measurements of {var} and {group_name} attributes are <u>dependent</u>.",
        )
    else:
        False, f"The intensity of measurements of {var} and {group_name} attributes are <u>independent</u>."


def summarize_stat_test_perf(
    df_stat_test: pd.DataFrame, other_missigness_cats: List[str], nb_metrics: int
) -> Dict[str, str]:
    """Generate summary string for each missingness category stating the ratio of
    metrics significantly worse than with_msrt and the corresponding largest delta.

    Parameters
    ----------
    df_stat_test : pd.DataFrame
        Statistical test dataframe
    other_missigness_cats : List[str]
        List of missingness categories different from with_msrt
    nb_metrics : int
        Number of metrics analyzed

    Returns
    -------
    Dict[str, str]
        Map missingness category to its summary string
    """
    summaries = {}
    for cat in other_missigness_cats:
        df_worse = df_stat_test[
            (df_stat_test["Missingness category"] == cat)
            & (df_stat_test["Category vs with msrt"] == "worse")
        ]
        worse_delta = df_worse["Delta"].max()
        worse_metric = df_worse.iloc[df_worse["Delta"].argmax()]["Metric"]
        summaries[
            cat
        ] = f"<b>{cat}</b>: {round(len(df_worse)/nb_metrics*100, 1)}% of metrics are worse than for with measurement time points, with the biggest delta {round(worse_delta, 3)} for metric {display_name_metric_in_sentence(worse_metric)}."
    return summaries


def summarize_intensity_msrt(
    intensity_msrts_df: pd.DataFrame,
    var: str,
    intensity_stat_test: Dict[str, Tuple[bool, str]],
    name_lower_intensity: str,
) -> pd.DataFrame:
    """Generate summary table for the intensity of measurement analysis.
    It lists groupings that are dependent on the intensity of measurement and the corresponding
    patients' category with the largest rate of no measurement and the largest rate of low intensity
    of measurement"

    Parameters
    ----------
    intensity_msrts_df : pd.DataFrame
        Intensity of measurement dataframe
    var : str
        Variable name
    intensity_stat_test : Dict[str, Tuple[bool, str]]
        Dictionary of the results of the independence test on intensity of measurement
    name_lower_intensity : str
        Name of the category for the lower intensity of measurement (that is not no_msrt)

    Returns
    -------
    pd.DataFrame
        Summary table
    """
    res = {
        "Group name": [],
        "Category with the biggest rate of no_msrt": [],
        f"Category with the biggest rate of {name_lower_intensity}": [],
    }
    for group_name, (test_res, _) in intensity_stat_test.items():
        if test_res:
            df = intensity_msrts_df.groupby(group_name)[var].value_counts(normalize=True)
            df = df.mul(100)
            df = df.rename("percent").reset_index()
            worst_cat_no_msrt = df[df[var] == "no_msrt"].iloc[
                df[df[var] == "no_msrt"]["percent"].argmax()
            ][group_name]
            worst_cat_lower_intensity = df[df[var] == name_lower_intensity].iloc[
                df[df[var] == name_lower_intensity]["percent"].argmax()
            ][group_name]
            res["Group name"].append(group_name)
            res["Category with the biggest rate of no_msrt"].append(worst_cat_no_msrt)
            res[f"Category with the biggest rate of {name_lower_intensity}"].append(
                worst_cat_lower_intensity
            )
    return pd.DataFrame(res)
