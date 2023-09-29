from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.platypus.tables import Table as TableType

from famews.fairness_check.feature_importance.helper_rbo import (
    compare_ranking_per_group,
    get_critical_invdelta_features,
    get_critical_rbo_value,
    get_invdelta_value,
)
from famews.fairness_check.report_generation.utils import TABLE_STYLE, df2table


def get_rbo_table(
    feat_ranking_all: List[str],
    feat_ranking_per_group: Dict[str, Dict[str, List[str]]],
    feat_ranking_random: Dict[str, Dict[str, List[str]]],
    k: int,
) -> Tuple[pd.DataFrame, List[int], float]:
    """Generate RBO table for each cohort.

    Parameters
    ----------
    feat_ranking_all : List[str]
        Overall feature ranking
    feat_ranking_per_group : Dict[str, Dict[str, List[str]]]
        Feature ranking per cohort for each grouping
    feat_ranking_random : Dict[str, Dict[str, List[str]]]
        Synthetic feature rankings
    k : int
        Size of the head to focus one

    Returns
    -------
    Tuple[pd.DataFrame, List[int], float]
        rbo table: Dataframe mapping each cohort of each grouping its RBO value
        to_color_red: list of rows to color in red
        critical_rbo: Critical RBO value
    """
    critical_rbo = get_critical_rbo_value(feat_ranking_all, feat_ranking_random, k)
    res_comparison = compare_ranking_per_group(
        feat_ranking_all, feat_ranking_per_group, critical_rbo, k
    )
    dict_tables = {"Grouping": [], "Category": [], "RBO": []}
    to_color_red = []
    i_row = 0
    for group_name, res in res_comparison.items():
        for cat, (rbo, is_critical) in res.items():
            dict_tables["Grouping"].append(group_name)
            dict_tables["Category"].append(cat)
            dict_tables["RBO"].append(f"{round(rbo, 3)}")
            if is_critical:
                to_color_red.append(i_row)
            i_row += 1
    return pd.DataFrame(dict_tables), to_color_red, critical_rbo


def get_topk_ranking_critical_per_group(
    feat_ranking_all: List[str],
    feat_ranking_per_group: Dict[str, Dict[str, List[str]]],
    feat_ranking_random: Dict[str, Dict[str, List[str]]],
    k: int,
) -> Tuple[
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, List[int]]],
    Dict[str, Dict[str, List[int]]],
]:
    """_summary_

    Parameters
    ----------
    feat_ranking_all : List[str]
        Overall feature ranking
    feat_ranking_per_group : Dict[str, Dict[str, List[str]]]
        Feature ranking per cohort for each grouping
    feat_ranking_random : Dict[str, Dict[str, List[str]]]
        Synthetic feature rankings
    k : int
        Size of the head to focus one

    Returns
    -------
    Tuple[ Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]], ]
        _description_
    """
    critical_invdelta_features = get_critical_invdelta_features(
        feat_ranking_all, feat_ranking_random
    )
    feat_ranking_all_index = {feat: rk for rk, feat in enumerate(feat_ranking_all)}
    topk_ranking_per_group = {group_name: {} for group_name in feat_ranking_per_group.keys()}
    to_color_red_per_group = {group_name: {} for group_name in feat_ranking_per_group.keys()}
    to_color_blue_per_group = {group_name: {} for group_name in feat_ranking_per_group.keys()}
    for group_name in feat_ranking_per_group.keys():
        for cat in feat_ranking_per_group[group_name].keys():
            (
                topk_ranking_per_group[group_name][cat],
                to_color_red_per_group[group_name][cat],
                to_color_blue_per_group[group_name][cat],
            ) = get_topk_ranking_critical(
                feat_ranking_all_index,
                feat_ranking_per_group[group_name][cat],
                critical_invdelta_features,
                k,
            )
    return topk_ranking_per_group, to_color_red_per_group, to_color_blue_per_group


def get_topk_ranking_critical(
    feat_ranking_all_index: Dict[str, int],
    feat_ranking_cat: List[str],
    critical_invdelta_features: Dict[str, float],
    k: int,
) -> Tuple[List[str], List[int], List[int]]:
    """_summary_

    Parameters
    ----------
    feat_ranking_all_index : Dict[str, int]
       Overall feature ranking
    feat_ranking_cat : List[str]
        Feature ranking for a cohort
    critical_invdelta_features : Dict[str, float]
        Map from feature names to critical delta of inverse rank
    k : int
        Size of ranking to focus on

    Returns
    -------
    Tuple[List[str], List[int], List[int]]
        top k ranking for the cohort, list of ranks to color in red, list of ranks to color in blue
    """
    topk_ranking = []
    to_color_red = []
    to_color_blue = []
    for rk in range(k):
        feat = feat_ranking_cat[rk]
        rk_all = feat_ranking_all_index[feat]
        if rk == rk_all:
            topk_ranking.append(feat)
        else:
            invdelta = get_invdelta_value(rk_all, rk)
            is_critical = invdelta > critical_invdelta_features[feat]
            topk_ranking.append(format_moved_feat(feat, rk, rk_all))
            if is_critical:
                if rk_all >= k:
                    to_color_red.append(rk)
                else:
                    to_color_blue.append(rk)
    return topk_ranking, to_color_red, to_color_blue


def format_moved_feat(feat_name: str, rk: int, rk_all: int) -> str:
    """Format feature that moved rank compared

    Parameters
    ----------
    feat_name : str
        Feature name
    rk : int
        New rank of the feature
    rk_all : int
        Overall rank of the feature

    Returns
    -------
    str
        Formatted rank string
    """
    if rk < rk_all:
        direction = "↑"
    elif rk > rk_all:
        direction = "↓"
    else:
        direction = "="
    diff_rk = abs(rk - rk_all)
    return f"{feat_name} ({direction} {diff_rk})"


def get_topk_ranking_per_group(
    feat_ranking_all: List[str], feat_ranking_per_group: Dict[str, Dict[str, List[str]]], k: int
) -> Dict[str, Dict[str, List[str]]]:
    """Obatin the top k features for each cohort.

    Parameters
    ----------
    feat_ranking_all : List[str]
        Overall feature ranking
    feat_ranking_per_group : Dict[str, Dict[str, List[str]]]
        Feature ranking for each cohort of each grouping
    k : int
        Size of feature ranking to focus on

    Returns
    -------
    Dict[str, Dict[str, List[str]]]
        Map each cohort of each cohort to its top k ranking
    """
    feat_ranking_all_index = {feat: rk for rk, feat in enumerate(feat_ranking_all)}
    topk_ranking_per_group = {group_name: {} for group_name in feat_ranking_per_group.keys()}
    for group_name in feat_ranking_per_group.keys():
        for cat in feat_ranking_per_group[group_name].keys():
            topk_ranking_per_group[group_name][cat] = get_topk_ranking(
                feat_ranking_all_index, feat_ranking_per_group[group_name][cat], k
            )
    return topk_ranking_per_group


def get_topk_ranking(
    feat_ranking_all_index: Dict[str, int], feat_ranking_cat: List[str], k: int
) -> List[str]:
    """Get the top k feature for a cohort, with nice formatting.

    Parameters
    ----------
    feat_ranking_all_index : Dict[str, int]
        Overall feature ranking
    feat_ranking_cat : List[str]
        Cohort feature ranking
    k : int
        Size of feature ranking to focus on

    Returns
    -------
    List[str]
        Top k features, nicely formatted
    """
    topk_ranking = []
    for rk in range(k):
        feat = feat_ranking_cat[rk]
        rk_all = feat_ranking_all_index[feat]
        if rk == rk_all:
            topk_ranking.append(feat)
        else:
            topk_ranking.append(format_moved_feat(feat, rk, rk_all))
    return topk_ranking


def df2table_rbo(df: pd.DataFrame, to_color: List[int]) -> TableType:
    """Generate ReportLab table for RBO.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of RBO analysis
    to_color : List[int]
        Lsit of row indices to color

    Returns
    -------
    TableType
        ReportLab table for RBO
    """
    style_to_color = []
    for i_row in to_color:
        style_to_color.append(("TEXTCOLOR", (2, i_row + 1), (2, i_row + 1), colors.red))
    return df2table(df, TABLE_STYLE + style_to_color)


def generate_table_topk_feat(
    k: int,
    topk_ranking_all: List[str],
    topk_ranking_group: Dict[str, List[str]],
    to_color_red: Dict[str, List[int]] = None,
    to_color_blue: Dict[str, List[int]] = None,
    max_cols: int = 5,
) -> List[TableType]:
    """Generate the ReportLab tables presenting the top k features for each cohort.

    Parameters
    ----------
    k : int
        Size of feature ranking to focus on
    topk_ranking_all : List[str]
        Overall feature ranking
    topk_ranking_group : Dict[str, List[str]]
        Top k ranking for each cohort
    to_color_red : Dict[str, List[int]], optional
        List of rows to color in red for each cohort, by default None
    to_color_blue : Dict[str, List[int]], optional
        List of rows to color in blue for each cohort, by default None
    max_cols : int, optional
        Maximum number of columns to put side by side, by default 5

    Returns
    -------
    List[TableType]
        List of ReportLab tables presenting the top k features for each cohort
    """
    table_style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.white),
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
    ]
    columns_group = {
        f"Top {k} {cat}": topk_ranking_cat for cat, topk_ranking_cat in topk_ranking_group.items()
    }
    df_feat_ranking = pd.DataFrame({f"Top {k}": topk_ranking_all, **columns_group})
    list_tables = []
    cats_to_cols = ["all"] + list(topk_ranking_group.keys())
    for i in range(int(np.ceil(len(cats_to_cols) / max_cols))):
        df_sub = df_feat_ranking.iloc[:, i * max_cols : (i + 1) * max_cols]
        sub_cats_to_cols = cats_to_cols[i * max_cols : (i + 1) * max_cols]
        style_color = []
        if to_color_red is not None and to_color_blue is not None:
            for i_col, cat in enumerate(sub_cats_to_cols):
                if cat == "all":
                    continue
                for i_row in to_color_red[cat]:
                    style_color.append(
                        ("TEXTCOLOR", (i_col, i_row + 1), (i_col, i_row + 1), colors.red)
                    )
                for i_row in to_color_blue[cat]:
                    style_color.append(
                        ("TEXTCOLOR", (i_col, i_row + 1), (i_col, i_row + 1), colors.blue)
                    )
        list_tables.append(df2table(df_sub, table_style + style_color))
    return list_tables
