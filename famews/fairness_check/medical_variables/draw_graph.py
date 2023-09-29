from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

from famews.fairness_check.utils.helper_format import get_pretty_name_medvars


def plot_boxplot_medvars_pw(
    medvars_group_df: pd.DataFrame,
    group_name: str,
    cats: List[str],
    list_medvars: List[str],
    var_suffixes: List[str] = ["", "_not_inevent", "_never_inevent"],
    figsize: Tuple[int, int] = (4, 5),
    do_stat_test: bool = False,
    is_binary_group: bool = None,
    df_stat_test: pd.DataFrame = None,
    dict_worst_cats: Dict[str, List[str]] = None,
    medvars_xlim: Dict[str, Tuple[int, int]] = None,
    color_palette: List[str] = None,
) -> List[Figure]:
    """Plot the median of the different medical variables for each category of patients.

    Parameters
    ----------
    medvars_group_df : pd.DataFrame
        Dataframe of median medical variables
    group_name : str
        Group name
    cats : List[str]
        List of categories
    list_medvars : List[str]
       List of medical variables
    var_suffixes: List[str]
        List of variable suffixes
    figsize : Tuple[int, int], optional
        Figure size, by default (4, 5)
    do_stat_test : bool, optional
        Flag to state whether we perform statistical testing, by default False
    is_binary_group : bool, optional
        flag to state whether the grouping is binary (useful only in do_stat_test is True), by default None
    df_stat_test : pd.DataFrame, optional
       Dataframe with statistical test results, by default None
    dict_worst_cats : Dict[str, List[str]], optional
        Dictionary of top k worst categories for each group_start_event, by default None
    medvars_xlim : Dict[str, Tuple[int, int]], optional
        X-lim for each medical variable, by default None
    color_palette : List[str], optional
        List of colors, by default None

    Returns
    -------
    List[Figure]
        List of box plots
    """
    figs = []
    for var in list_medvars:
        fig, axs = plt.subplots(nrows=1, ncols=len(var_suffixes), figsize=figsize, sharey=True)
        for i in range(len(var_suffixes)):
            axs[i] = draw_boxplot_medvar(
                medvars_group_df,
                group_name,
                cats,
                var,
                var_suffixes[i],
                axs[i],
                do_stat_test,
                is_binary_group,
                df_stat_test,
                dict_worst_cats,
                medvars_xlim,
                color_palette,
            )
        figs.append(fig)
    return figs


def draw_boxplot_medvar(
    medvars_group_df: pd.DataFrame,
    group_name: str,
    cats: List[str],
    var: str,
    var_suffix: str,
    ax: Axes,
    do_stat_test: bool = False,
    is_binary_group: bool = None,
    df_stat_test: pd.DataFrame = None,
    dict_worst_cats: Dict[str, List[str]] = None,
    medvars_xlim: Dict[str, Tuple[float, float]] = None,
    color_palette: List[str] = None,
) -> Axes:
    """Draw box plot for a medical variable for each category.

    Parameters
    ----------
    medvars_group_df : pd.DataFrame
        Dataframe of median medical variables
    group_name : str
        Group name
    cats : List[str]
        List of categories
    var: str
        Medical variable name
    var_suffix: str:
        Suffix
    ax : Axes
        Axis
    do_stat_test : bool, optional
        Flag to state whether we perform statistical testing, by default False
    is_binary_group : bool, optional
        flag to state whether the grouping is binary (useful only in do_stat_test is True), by default None
    df_stat_test : pd.DataFrame, optional
       Dataframe with statistical test results, by default None
    dict_worst_cats : Dict[str, List[str]], optional
        Dictionary of top k worst categories for each group_start_event, by default None
    medvars_xlim : Dict[str, Tuple[int, int]], optional
        X-lim for each medical variable, by default None
    color_palette : List[str], optional
        List of colors, by default None

    Returns
    -------
    Axes
        Box plot
    """
    sns.boxplot(
        data=medvars_group_df,
        x=var + var_suffix,
        y=group_name,
        ax=ax,
        palette=color_palette,
    )
    pretty_name_medvar = get_pretty_name_medvars(var, var_suffix)
    ax.set_title(pretty_name_medvar)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    if medvars_xlim and var in medvars_xlim:
        ax.set(xlim=medvars_xlim[var])

    if do_stat_test:
        if is_binary_group:
            if pretty_name_medvar in list(df_stat_test["Medical Variable"]):
                for i, cat in enumerate(cats):
                    draw_star(
                        medvars_group_df,
                        var,
                        var_suffix,
                        dict_worst_cats,
                        group_name,
                        cat,
                        is_binary_group,
                        i,
                        ax,
                    )
        else:
            significantly_diff_cats = df_stat_test[
                df_stat_test["Medical Variable"] == pretty_name_medvar
            ]["Category"]
            for i, cat in enumerate(cats):
                if cat in list(significantly_diff_cats):
                    draw_star(
                        medvars_group_df,
                        var,
                        var_suffix,
                        dict_worst_cats,
                        group_name,
                        cat,
                        is_binary_group,
                        i,
                        ax,
                    )

    return ax


def draw_star(
    medvars_group_df: pd.DataFrame,
    var: str,
    var_suffix: str,
    dict_worst_cats: Dict[str, List[str]],
    group_name: str,
    cat: str,
    is_binary_group: bool,
    i_box: int,
    ax: Axes,
):
    """Draw a star next to the boxplot of the specific category if its median medical variable is statistically different.
    The star is red if the category is among the top k worst.

    Parameters
    ----------
    medvars_group_df : pd.DataFrame
        Dataframe of median medical variables
    var: str
        Medical variable
    var_suffix: str
        Variable suffix
    dict_worst_cats : Dict[str, List[str]]
        Dictionary of top k worst categories for each group_start_event
    group_name : str
        Group name
    cat : str
        Category name
    is_binary_group : bool
        flag to state whether the grouping is binary (useful only in do_stat_test is True)
    i_box : int
        Index of box
    ax : Axes
        Axis
    """
    int_ticks = ax.get_xticks()[1] - ax.get_xticks()[0]
    x_place = (
        medvars_group_df[medvars_group_df[group_name] == cat][var + var_suffix].quantile(0.75)
        + int_ticks * 0.05
    )
    if is_binary_group:
        y_place = i_box - 0.25
    else:
        y_place = i_box - 0.15
    if cat in dict_worst_cats[(var, var_suffix)]:
        color_star = "r"
    else:
        color_star = "k"
    ax.text(x_place, y_place, "*", fontweight="extra bold", c=color_star)
