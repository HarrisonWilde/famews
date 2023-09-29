import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from famews.fairness_check.model_performance.build_table import (
    generate_table_binary_group,
    generate_table_multicat_group,
)
from famews.fairness_check.model_performance.constants import METRICS_LOW
from famews.fairness_check.model_performance.draw_graph import draw_prc_group
from famews.fairness_check.report_generation.utils import (
    df2table,
    double_fig2image,
    fig2image,
)
from famews.fairness_check.utils.helper_format import (
    display_name_metric_in_sentence,
    get_pretty_name_metric,
)


def get_PRC_figure(
    metrics_df: pd.DataFrame,
    curves_group: Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]],
    group_name: str,
    cats: List[str],
    list_metrics: List[str],
    event_based: bool,
    colors: List[str],
    figsize_curves: Tuple[int, int],
):
    """_summary_

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe for a grouping
    curves_group : Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]]
        Dictionary containing curve points for each grouping, category and curve type
    group_name : str
        Group name
    cats : List[str]
        List of categories
    list_metrics : List[str]
        List of metrics
    event_based : bool
        Whether we are drawing the curve for event-based recall
    colors : List[str]
        List of colors
    figsize_curves : Tuple[int, int]
        Figure size

    Returns
    -------
    Reportlab images
    """
    if event_based:
        prefix = "event_"
        prefix_title = "event-based "
    else:
        prefix = ""
        prefix_title = ""
    if f"{prefix}auprc" in list_metrics:
        fig_prc = draw_prc_group(
            metrics_df,
            curves_group,
            f"{prefix}recall",
            "precision",
            group_name,
            cats,
            colors,
            figsize_curves,
            f"Precision / {prefix_title}recall curve",
        )
    else:
        fig_prc = None
    if f"corrected_{prefix}auprc" in list_metrics:
        fig_corr_prc = draw_prc_group(
            metrics_df,
            curves_group,
            f"{prefix}recall",
            "corrected_precision",
            group_name,
            cats,
            colors,
            figsize_curves,
            f"Corrected precision / {prefix_title}recall curve",
        )
    else:
        fig_corr_prc = None
    if fig_prc and fig_corr_prc:
        return double_fig2image(fig_prc, fig_corr_prc)
    elif fig_prc:
        return fig2image(fig_prc)
    elif fig_corr_prc:
        return fig2image(fig_corr_prc)
    return None


def get_stat_test_table_performance(
    metrics_group_df: pd.DataFrame,
    type_table_groups: Dict[str, List],
    group_name: str,
    cats: List[str],
    list_metrics: List[str],
    worst_cats_metrics: Dict[str, list],
    significance_level: float,
    filter_delta: float,
) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """_summary_

    Parameters
    ----------
    metrics_group_df : pd.DataFrame
        Dataframe of metrics
    type_table_groups : Dict[str, List]
        Dictionary mapping comparison type to list of groupings
    group_name : str
        Group name
    cats : List[str]
        List of categories
    list_metrics : List[str]
        List of metrics
    worst_cats_metrics : Dict[str, list]
        Dictionary mapping a metric to the list of categories and corresponding deltas
    significance_level : float
        Significance level
    filter_delta : float
        Filter on delta

    Returns
    -------
    pd.DataFrame
        dataframe with result of statistical test (p-value, delta)
    Dict[str, list]
        Updated worst_cats_metrics with the considered categories
    """
    if group_name in type_table_groups["binary_group"]:
        metrics_df = metrics_group_df[metrics_group_df["group"] == group_name].rename(
            columns={"cat": group_name}
        )
        table, worst_cats = generate_table_binary_group(
            metrics_df,
            list_metrics,
            group_name,
            cats[0],
            cats[1],
            significance_level,
            filter_delta,
        )
    elif group_name in type_table_groups["multicat_group"]:
        metrics_df = metrics_group_df[
            (metrics_group_df["group"] == group_name)
            | (metrics_group_df["group"] == f"{group_name}_bool")
        ].rename(columns={"cat": group_name})
        table, worst_cats = generate_table_multicat_group(
            metrics_df,
            list_metrics,
            group_name,
            cats,
            significance_level,
            filter_delta,
        )
    else:
        logging.warning(
            f"[Model Performance] Grouping {group_name} has no table type to generate the statistical test table."
        )
        return pd.DataFrame(), worst_cats_metrics
    return table, {
        metric: worst_cats_metrics[metric] + worst_cats[metric] for metric in list_metrics
    }


def get_table_summarized_metrics_group(
    metrics_df: pd.DataFrame, list_metrics: List[str], cats_size: Dict[str, int]
) -> Tuple[str, pd.DataFrame]:
    """Construct summary table for a grouping, displaying for each metric the macro-average,
    the minimum and the value for the minority group.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe
    list_metrics : List[str]
        List of metrics
    cats_size : Dict[str, int]
        Dictionary mapping a category of patients to its size

    Returns
    -------
    Tuple[str, pd.DataFrame]
        String presenting minority category, Summary table
    """
    res = {
        "Metric": [],
        "Macro-average": [],
        "Worst value (category)": [],
        "For minority category": [],
    }
    for metric in list_metrics:
        df = metrics_df.groupby("cat")[metric].median()
        avg_metric = df.mean()
        if sum(metric.startswith(m) for m in METRICS_LOW):
            worst_metric = df.max()
            worst_cat = df.index[df.argmax()]
        else:
            worst_metric = df.min()
            worst_cat = df.index[df.argmin()]
        minority_cat = min(cats_size, key=cats_size.get)
        minority_metric = df[minority_cat]
        res["Metric"].append(get_pretty_name_metric(metric))
        res["Macro-average"].append(round(avg_metric, 3))
        res["Worst value (category)"].append(f"{round(worst_metric, 3)} ({worst_cat})")
        res["For minority category"].append(round(minority_metric, 3))
    return f"The minority category is {minority_cat}.", pd.DataFrame(res)


def generate_summary_string_worst_ratio_group(
    info_worst_ratio: Dict[str, Union[str, float]]
) -> str:
    """Generate summary string for the ratio of significantly worse metrics analysis for a grouping.

    Parameters
    ----------
    info_worst_ratio : Dict[str, Union[str, float]]
        Dictionary containing results of the ratio of significantly worse metrics analysis for a grouping

    Returns
    -------
    str
        Summary string
    """
    return f"<u>Worst ratio:</u> {round(info_worst_ratio['ratio']*100, 1)}% for category {info_worst_ratio['cat']} with the biggest delta {round(info_worst_ratio['delta'], 3)} on {display_name_metric_in_sentence(info_worst_ratio['metric'])}."


def generate_summary_string_worst_delta_group(
    info_worst_delta: Dict[str, Union[str, float]]
) -> str:
    """Generate summary string on the largest delta for the ratio of significantly worse metrics analysis for a grouping.

    Parameters
    ----------
    info_worst_delta : Dict[str, Union[str, float]]
        Dictionary containing results of the delta analysis for s grouping

    Returns
    -------
    str
        Summary string
    """
    if info_worst_delta is None:
        return "<u>Worst delta</u> is the same as above."
    return f"<u>Worst delta:</u> {round(info_worst_delta['delta'], 3)} on {display_name_metric_in_sentence(info_worst_delta['metric'])} for category {info_worst_delta['cat']}."


def generate_summary_string_worst_ratio_all(
    info_worst_ratio: Dict[str, Dict[str, Union[str, float]]]
) -> str:
    """Generate summary string for the overall ratio of significantly worse metrics analysis.

    Parameters
    ----------
    info_worst_ratio : Dict[str, Dict[str, Union[str, float]]]
        Dictionary containing results of the ratio of significantly worse metrics analysis


    Returns
    -------
    str
        Summary string
    """
    dict_worst_ratio = {group_name: dc["ratio"] for group_name, dc in info_worst_ratio.items()}
    worst_group = max(dict_worst_ratio, key=dict_worst_ratio.get)
    info_worst_group = info_worst_ratio[worst_group]
    return f"<u>Worst ratio:</u> {round(dict_worst_ratio[worst_group]*100, 1)}% for category {info_worst_group['cat']} ({worst_group}) with the biggest delta {round(info_worst_group['delta'], 3)} on {display_name_metric_in_sentence(info_worst_group['metric'])}."


def generate_summary_string_worst_delta_all(
    info_worst_ratio: Dict[str, Dict[str, Union[str, float]]],
    info_worst_delta: Dict[str, Dict[str, Union[str, float]]],
) -> str:
    """Generate summary string on the largest delta for the overall ratio of significantly worse metrics analysis.

    Parameters
    ----------
    info_worst_ratio : Dict[str, Dict[str, Union[str, float]]]
        Dictionary containing results of the ratio of significantly worse metrics analysis

    info_worst_delta : Dict[str, Dict[str, Union[str, float]]]
        Dictionary containing results of the delta analysis

    Returns
    -------
    str
        Summary string
    """
    if not info_worst_delta:
        return "<u>Worst delta</u> is the same as above."
    dict_delta_worst_ratio = {
        group_name: dc["delta"] for group_name, dc in info_worst_ratio.items()
    }
    delta_worst_ratio = max(dict_delta_worst_ratio.values())
    dict_worst_delta = {group_name: dc["delta"] for group_name, dc in info_worst_delta.items()}
    worst_delta = max(dict_worst_delta.values())
    if worst_delta <= delta_worst_ratio:
        return "<u>Worst delta</u> is the same as above."
    worst_group = max(dict_worst_delta, key=dict_worst_delta.get)
    info_worst_group = info_worst_delta[worst_group]
    return f"<u>Worst delta:</u> {round(worst_delta, 3)} on {display_name_metric_in_sentence(info_worst_group['metric'])} for category {info_worst_group['cat']} ({worst_group})."


def generate_table_ratio_stat_test(
    groups: Dict[str, List[str]],
    stat_test_groups: Dict[str, pd.DataFrame],
    type_table_groups: Dict[str, List],
    nb_metrics: int,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]], Tuple[str, str]]:
    """Run the aggregated view on ratio of significanlty worse metrics

    Parameters
    ----------
    groups : Dict[str, List[str]]
        Groupings definition {group_name: [category]}
    stat_test_groups : Dict[str, pd.DataFrame]
        Results of statistical test for each grouping
    type_table_groups : Dict[str, List]
        Type of comparison to perform for each group {comparison_type: [grouping]}
    nb_metrics : int
        Number of metrics

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[str, str]], Tuple[str, str]]
    - Map from group name to summary table
    - Map from group name to summary strings
    - Global summary strings

    """

    summary_strings_group = {}
    info_worst_ratio_group = {}
    info_worst_delta_group = {}
    tables_ratio_group = {}
    for group_name, cats in groups.items():
        df_stat_test = stat_test_groups[group_name]
        is_binary = group_name in type_table_groups["binary_group"]
        ratio_cats, info_worst_ratio, info_worst_delta = get_ratio_index_group(
            df_stat_test, cats, is_binary, nb_metrics
        )
        info_worst_ratio_group[group_name] = info_worst_ratio
        if info_worst_delta is not None:
            info_worst_delta_group[group_name] = info_worst_delta
        summary_strings_group[group_name] = (
            generate_summary_string_worst_ratio_group(info_worst_ratio),
            generate_summary_string_worst_delta_group(info_worst_delta),
        )
        tables_ratio_group[group_name] = pd.DataFrame(
            {cat: [f"{round(ratio*100, 1)}%"] for cat, ratio in ratio_cats.items()}
        )
    global_summary_string = (
        generate_summary_string_worst_ratio_all(info_worst_ratio_group),
        generate_summary_string_worst_delta_all(info_worst_ratio_group, info_worst_delta_group),
    )
    return tables_ratio_group, summary_strings_group, global_summary_string


def get_ratio_index_group(
    df_stat_test: pd.DataFrame, cats: List[str], is_binary: bool, nb_metrics: int
) -> Tuple[Dict[str, float], Dict[str, Union[str, float]], Dict[str, Union[str, float]]]:
    """_summary_

    Parameters
    ----------
    df_stat_test : pd.DataFrame
        Result statistical test
    cats : List[str]
        List of categories
    is_binary : bool
        Whether the grouping is binary
    nb_metrics : int
        Number of metrics

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, Union[str, float]], Dict[str, Union[str, float]]]
        - Map from category to ratio of significantly worse metrics
        - Dictionary of summary information
        - Dictionary of summary information for the delta
    """
    ratio_cats = {}
    for cat in cats:
        if is_binary:
            ratio_cats[cat] = sum(df_stat_test["Cohort with the worst metric"] == cat) / nb_metrics
        else:
            ratio_cats[cat] = (
                sum(
                    (df_stat_test["Category"] == cat) & (df_stat_test["Cohort vs. rest"] == "worse")
                )
                / nb_metrics
            )
    worst_ratio = max(ratio_cats.values())
    worst_cat = max(ratio_cats, key=ratio_cats.get)
    if is_binary:
        df_worst_cat = df_stat_test[df_stat_test["Cohort with the worst metric"] == worst_cat]
    else:
        df_worst_cat = df_stat_test[
            (df_stat_test["Category"] == worst_cat) & (df_stat_test["Cohort vs. rest"] == "worse")
        ]
    delta_worst_cat = df_worst_cat["Delta"].max()
    metric_worst_cat = df_worst_cat["Metric"].iloc[df_worst_cat["Delta"].argmax()]
    if is_binary:
        df_worse = df_stat_test
    else:
        df_worse = df_stat_test[df_stat_test["Cohort vs. rest"] == "worse"]
    if delta_worst_cat < df_worse["Delta"].max():
        worst_delta = df_worse["Delta"].max()
        cat_worst_delta = df_worse["Category"].iloc[df_worse["Delta"].argmax()]
        metric_worst_delta = df_worse["Metric"].iloc[df_worse["Delta"].argmax()]
        return (
            ratio_cats,
            {
                "ratio": worst_ratio,
                "cat": worst_cat,
                "metric": metric_worst_cat,
                "delta": delta_worst_cat,
            },
            {"cat": cat_worst_delta, "metric": metric_worst_delta, "delta": worst_delta},
        )
    return (
        ratio_cats,
        {
            "ratio": worst_ratio,
            "cat": worst_cat,
            "metric": metric_worst_cat,
            "delta": delta_worst_cat,
        },
        None,
    )
