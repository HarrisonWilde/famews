from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure


def plot_boxplot_timegap_pw(
    timegap_group_df: pd.DataFrame,
    group_name: str,
    cats: List[str],
    list_group_start_event: List[str],
    figsize: Tuple[int, int] = (4, 5),
    do_stat_test: bool = False,
    type_table_groups: Dict[str, List[str]] = None,
    df_stat_test: pd.DataFrame = None,
    dict_worst_cats: Dict[str, List[str]] = None,
    timegap_xlim: Dict[str, Tuple[int, int]] = None,
    color_palette: List[str] = None,
) -> List[Figure]:
    """Plot the median time gap for each category of patients and each group_start_event.

    Parameters
    ----------
    timegap_group_df : pd.DataFrame
        Dataframe of time gap
    group_name : str
        Group name
    cats : List[str]
        List of categories
    list_group_start_event : List[str]
       List of groups of start event windows
    figsize : Tuple[int, int], optional
        Figure size, by default (4, 5)
    do_stat_test : bool, optional
        Flag to state whether we perform statistical testing, by default False
    type_table_groups: Dict[str, List[str]], optional
        Dictionary matching each of two types (binary, multiple categories) of comparison to its list of groupings, by default None
    df_stat_test : pd.DataFrame, optional
       Dataframe with statistical test results, by default None
    dict_worst_cats : Dict[str, List[str]], optional
        Dictionary of top k worst categories for each group_start_event, by default None
    timegap_xlim : Dict[str, Tuple[int, int]], optional
        X-lim of time gap for each group_start_event, by default None
    color_palette : List[str], optional
        List of colors, by default None

    Returns
    -------
    List[Figure]
        List of box plots
    """
    figs = []
    for it in range(int(np.ceil(len(list_group_start_event) / 2))):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
        axs[0] = draw_boxplot_timegap(
            timegap_group_df,
            group_name,
            cats,
            list_group_start_event[it * 2],
            axs[0],
            do_stat_test,
            type_table_groups,
            df_stat_test,
            dict_worst_cats,
            timegap_xlim,
            color_palette,
        )
        if len(list_group_start_event) > it * 2 + 1:
            axs[1] = draw_boxplot_timegap(
                timegap_group_df,
                group_name,
                cats,
                list_group_start_event[it * 2 + 1],
                axs[1],
                do_stat_test,
                type_table_groups,
                df_stat_test,
                dict_worst_cats,
                timegap_xlim,
                color_palette,
            )
        else:
            fig.delaxes(axs[1])
        figs.append(fig)
    return figs


def draw_boxplot_timegap(
    timegap_group_df: pd.DataFrame,
    group_name: str,
    cats: List[str],
    group_start_event: str,
    ax: Axes,
    do_stat_test: bool = False,
    type_table_groups: Dict[str, List[str]] = None,
    df_stat_test: pd.DataFrame = None,
    dict_worst_cats: Dict[str, List[str]] = None,
    timegap_xlim: Dict[str, Tuple[float, float]] = None,
    color_palette: List[str] = None,
) -> Axes:
    """Draw box plot of time gap for each category and for the group_start_event.

    Parameters
    ----------
    timegap_group_df : pd.DataFrame
        Dataframe of time gap
    group_name : str
        Group name
    cats : List[str]
        List of categories
    group_start_event : str
        Group name of start event window
    ax : Axes
        Axis
    do_stat_test : bool, optional
        Flag to state whether we perform statistical testing, by default False
    type_table_groups: Dict[str, List[str]], optional
        Dictionary matching each of three types of comparison to its list of groupings, by default None
    df_stat_test : pd.DataFrame, optional
       Dataframe with statistical test results, by default None
    dict_worst_cats : Dict[str, List[str]], optional
        Dictionary of top k worst categories for each group_start_event, by default None
    timegap_xlim : Dict[str, Tuple[int, int]], optional
        X-lim of time gap for each group_start_event, by default None
    color_palette : List[str], optional
        List of colors, by default None

    Returns
    -------
    Axes
        Box plot
    """
    timegap_df = timegap_group_df[timegap_group_df["group_start_event"] == group_start_event]
    sns.boxplot(
        data=timegap_df,
        x="time_gap",
        y=group_name,
        ax=ax,
        palette=color_palette,
    )
    ax.set_title(f"Window start first event: {group_start_event}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    if timegap_xlim and group_start_event in timegap_xlim:
        ax.set(xlim=timegap_xlim[group_start_event])

    if do_stat_test:
        if group_name in type_table_groups["binary_group"]:
            if len(df_stat_test[df_stat_test["Start event"] == group_start_event]):
                cat = df_stat_test[df_stat_test["Start event"] == group_start_event].iloc[0][
                    "Cohort with the worst time gap"
                ]
                i = cats.index(cat)
                draw_star(
                    timegap_df,
                    dict_worst_cats,
                    group_start_event,
                    group_name,
                    cat,
                    True,
                    i,
                    ax,
                )
        else:
            significantly_diff_cats = df_stat_test[
                (df_stat_test["Start event"] == group_start_event)
                & (df_stat_test["Cohort vs. rest"] == "worse")
            ]["Category"]
            for i, cat in enumerate(cats):
                if cat in list(significantly_diff_cats):
                    draw_star(
                        timegap_df,
                        dict_worst_cats,
                        group_start_event,
                        group_name,
                        cat,
                        False,
                        i,
                        ax,
                    )

    return ax


def draw_star(
    timegap_df: pd.DataFrame,
    dict_worst_cats: Dict[str, List[str]],
    group_start_event: str,
    group_name: str,
    cat: str,
    is_binary_group: bool,
    i_box: int,
    ax: Axes,
):
    """Draw start next to boxplot of the specific category if its medin time gap is statistically different.
    The start is red if the category is among the top k worst for the specific group_start_event.

    Parameters
    ----------
    timegap_df : pd.DataFrame
        Dataframe of time gap
    dict_worst_cats : Dict[str, List[str]]
        Dictionary of top k worst categories for each group_start_event
    group_start_event : str
        Group name of start event window
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
        timegap_df[timegap_df[group_name] == cat]["time_gap"].quantile(0.75) + int_ticks * 0.05
    )
    if is_binary_group:
        y_place = i_box - 0.25
    else:
        y_place = i_box - 0.15
    if cat in dict_worst_cats[group_start_event]:
        color_star = "r"
    else:
        color_star = "k"
    ax.text(x_place, y_place, "*", fontweight="extra bold", c=color_star)
