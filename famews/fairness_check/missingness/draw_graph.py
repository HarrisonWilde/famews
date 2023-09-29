from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

from famews.fairness_check.utils.helper_format import get_pretty_name_metric


def draw_intensity_bar_plot(
    var: str,
    group_name: str,
    cats: List[str],
    intensity_msrts_df: pd.DataFrame,
    order_intensity: List[str],
    colors: List[str],
    figsize: Tuple[int, int],
) -> Figure:
    """Draw the bar plot of the intensity of measurements distribution per cohort of a grouping.

    Parameters
    ----------
    var : str
        Variable name
    group_name : str
        Group name
    cats : List[str]
        List of categories
    intensity_msrts_df : pd.DataFrame
        Dataframe of intensity of measurements
    order_intensity : List[str]
        Order of intensity of measurements categories
    colors : List[str]
        List of colors
    figsize : Tuple[int, int]
        Figure size

    Returns
    -------
    Figure
        Bar plot
    """
    df = intensity_msrts_df.groupby(group_name)[var].value_counts(normalize=True)
    df = df.mul(100)
    df = df.rename("percent").reset_index()
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        x=group_name,
        y="percent",
        hue=var,
        data=df,
        palette=colors,
        hue_order=order_intensity,
        order=cats,
    )
    for c, rate in enumerate(order_intensity):
        color = colors[c]
        plt.axhline(
            sum(intensity_msrts_df[var] == rate) / len(intensity_msrts_df) * 100,
            ls=":",
            c=color,
            lw=0.7,
        )
    plt.rcParams["font.size"] = 5
    for c in ax.containers:
        labels = [f"{round(v.get_height(), 2)}%" for v in c]
        ax.bar_label(c, labels=labels, label_type="edge")
    plt.rcParams["font.size"] = 7
    plt.legend(bbox_to_anchor=(1.12, 0.75), loc="upper right", title="Intensity msrt", fontsize=7)
    plt.xlabel(group_name)
    return fig


def plot_boxplot_missingness_metrics_pw(
    metrics_df: pd.DataFrame,
    missingness_cats: List[str],
    metrics_name: List[str],
    figsize: Tuple[int, int] = (4, 5),
    do_stat_test: bool = False,
    df_stat_test: pd.DataFrame = None,
    metrics_xlim: Dict[str, Tuple[float, float]] = None,
    color_palette: List[str] = None,
) -> List[Figure]:
    """Plot box plot of metrics two by two.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe
    missingness_cats: List[str]
        List of missingness categories
    metrics_name : List[str]
        List of metric names
    figsize : Tuple[int, int], optional
        Figure size, by default (4, 5)
    do_stat_test : bool, optional
        Flag stating whether we do statistical test, by default False
    df_stat_test : pd.DataFrame, optional
        Dataframe with results of statistical test, by default None
    metrics_xlim : Dict[str, Tuple[float, float]], optional
        Dictionary of x-axis limit for each metric, by default None
    color_palette : List[str], optional
        List of colors, by default None

    Returns
    -------
    List[Figure]
        List of box plot figures
    """
    figs = []
    for it in range(int(np.ceil(len(metrics_name) / 2))):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
        axs[0] = draw_boxplot_metric(
            metrics_df,
            metrics_name[it * 2],
            missingness_cats,
            axs[0],
            do_stat_test,
            df_stat_test,
            metrics_xlim,
            color_palette,
        )
        if len(metrics_name) > it * 2 + 1:
            axs[1] = draw_boxplot_metric(
                metrics_df,
                metrics_name[it * 2 + 1],
                missingness_cats,
                axs[1],
                do_stat_test,
                df_stat_test,
                metrics_xlim,
                color_palette,
            )
        else:
            fig.delaxes(axs[1])
        figs.append(fig)
    return figs


def draw_boxplot_metric(
    metrics_df: pd.DataFrame,
    metric: str,
    missingness_cats: List[str],
    ax: Axes,
    do_stat_test: bool = False,
    df_stat_test: pd.DataFrame = None,
    metrics_xlim: Dict[str, Tuple[float, float]] = None,
    color_palette: List[str] = None,
) -> Axes:
    """Draw the box plot for a metric.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe
    metric : str
        Metric name
    missingness_cats : List[str]
        List of missingness categories
    ax : Axes
        Axis
    do_stat_test : bool, optional
        Flag stating whether we do statistical test, by default False
    df_stat_test : pd.DataFrame, optional
        Dataframe with results of statistical test, by default None
    metrics_xlim : Dict[str, Tuple[float, float]], optional
        Dictionary of x-axis limit for each metric, by default None
    color_palette : List[str], optional
        List of colors, by default None

    Returns
    -------
    Axes
        Box plot figure for a metric
    """
    sns.boxplot(
        data=metrics_df, x=metric, y="cat", ax=ax, palette=color_palette, order=missingness_cats
    )
    pretty_name_metric = get_pretty_name_metric(metric)
    ax.set_title(pretty_name_metric)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    if metrics_xlim and metric in metrics_xlim:
        ax.set(xlim=metrics_xlim[metric])

    if do_stat_test:
        df_stat_test_metric = df_stat_test[df_stat_test["Metric"] == pretty_name_metric]
        if len(df_stat_test_metric):
            if "no_msrt" in df_stat_test_metric["Missingness category"].values:
                i_no_msrt = missingness_cats.index("no_msrt")
                draw_star(metrics_df, metric, "no_msrt", i_no_msrt, ax)
            if "missing_msrt" in df_stat_test_metric["Missingness category"].values:
                i_missing_msrt = missingness_cats.index("missing_msrt")
                draw_star(metrics_df, metric, "missing_msrt", i_missing_msrt, ax)

    return ax


def draw_star(
    metrics_df: pd.DataFrame,
    metric: str,
    missingness_cat: str,
    i_box: int,
    ax: Axes,
):
    """_summary_

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe
    metric : str
        Metrics name
    missingness_cat : str
        Missingness category
    i_box : int
        Index of box
    ax : Axes
        Axis
    """
    int_ticks = ax.get_xticks()[1] - ax.get_xticks()[0]
    x_place = (
        metrics_df[metrics_df["cat"] == missingness_cat][metric].quantile(0.75) + int_ticks * 0.05
    )
    y_place = i_box - 0.25
    ax.text(x_place, y_place, "*", fontweight="extra bold", c="k")
