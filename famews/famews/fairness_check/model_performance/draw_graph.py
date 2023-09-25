from typing import Dict, List, Tuple

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

from famews.fairness_check.model_performance.constants import (
    ERROR_BOX,
    PRETTY_NAME_AXIS,
)
from famews.fairness_check.utils.helper_format import get_pretty_name_metric


def draw_calibration_group(
    curves_dict: Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]],
    group_name: str,
    cats: List[str],
    color_palette: List[str],
    figsize: Tuple[int, int],
    graph_title: str = "",
) -> Figure:
    """Draw calibration curve for each category.

    Parameters
    ----------
    curves_dict : Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]]
        Dictionary containing curve points for each grouping, category and curve type
    group_name : str
        Group name
    cats : List[str]
        Categories
    color_palette : List[str]
        List of colors
    figsize : Tuple[int, int]
        Figure size
    graph_title : str, optional
        Graph title, by default ""

    Returns
    -------
    Figure
        Calibration curve figure
    """
    fig, ax = draw_curve_group(
        curves_dict,
        "calibration",
        "prob_pred",
        "prob_true",
        group_name,
        cats,
        color_palette,
        figsize,
        graph_title,
    )
    plt.legend(bbox_to_anchor=(1.12, 0.75), loc="upper right", title=group_name)
    agg_errors = agg_calibration_error(curves_dict, group_name, cats)
    fig = add_curve_errors(agg_errors, "Calibration Error", "calibration", color_palette, fig, ax)
    plt.plot(
        np.arange(0, 1.1, 0.1),
        np.arange(0, 1.1, 0.1),
        linestyle="dashed",
        color="black",
        linewidth=1,
    )
    return fig


def add_curve_errors(
    errors: Dict[str, Tuple[float, float]],
    error_name: str,
    error_type: str,
    color_palette: List[str],
    fig: Figure,
    ax: Axes,
) -> Figure:
    """_summary_

    Parameters
    ----------
    errors : Dict[str, Tuple[float, float]]
        Dictionary mapping each cohort to the mean and std of the corresponding curve error
    error_name : str
        Error name
    color_palette : Lsit[str]
        List of colors
    fig : Figure
        Figure
    ax : Axes
        Axis

    Returns
    -------
    Figure
        Figure with mean and std curve error per category printed
    """
    texts = [f"{round(error[0], 3)}±{round(error[1], 3)}" for error in errors.values()]
    Texts = [TextArea(f"{error_name}:", textprops=dict(weight="bold")), TextArea("")]
    for t, c in zip(texts, color_palette[: len(texts)]):
        Texts.append(TextArea(t, textprops=dict(color=c)))
    texts_vbox = VPacker(children=Texts, pad=0, sep=0)
    ann = AnnotationBbox(
        texts_vbox,
        ERROR_BOX[error_type]["xy"],
        xycoords=ax.transAxes,
        box_alignment=ERROR_BOX[error_type]["alignment"],
        bboxprops=dict(alpha=0),
    )
    ann.set_figure(fig)
    fig.artists.append(ann)
    return fig


def agg_calibration_error(
    curves_dict: Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]],
    group_name: str,
    cats: List[str],
) -> Dict[str, Tuple[float, float]]:
    """Get calibration error for each category.

    Parameters
    ----------
    curves_dict : Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]]
        Dictionary containing curve points for each grouping, category and curve type
    group_name : str
        Group name
    cats : List[str]
        Categories

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Dictionary mapping each category to the mean and std of the corresponding calibration error
    """
    errors = {cat: [] for cat in cats}
    for i in range(len(curves_dict[group_name][cats[0]])):
        for cat in cats:
            errors[cat].append(curves_dict[group_name][cat][i]["calibration"]["error"])
    return {cat: (np.mean(errors[cat]), np.std(errors[cat])) for cat in cats}


def agg_auc_group(
    metrics_df: pd.DataFrame, metric_name: str, group_name: str, cats: List[str]
) -> Dict[str, Tuple[float, float]]:
    """Get the mean and std AUC for each category.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe for a grouping
    metric_name : str
        Metric name
    group_name : str
        Group name
    cats : List[str]
        List of categories

    Returns
    -------
    Dict[str, Tuple[float, float]]
        Dictionary mapping category to mean and std AUC
    """
    metrics_df_grouped = metrics_df.groupby(group_name)
    mean_metric = metrics_df_grouped[metric_name].mean()
    std_metric = metrics_df_grouped[metric_name].std()
    return {cat: (mean_metric[cat], std_metric[cat]) for cat in cats}


def draw_roc_group(
    metrics_df: pd.DataFrame,
    curves_dict: Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]],
    group_name: str,
    cats: List[str],
    color_palette: List[str],
    figsize: Tuple[int, int],
    graph_title: str = "",
) -> Figure:
    """Draw ROC curve for each category.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe for a grouping
    curves_dict : Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]]
        Dictionary containing curve points for each grouping, category and curve type
    group_name : str
        Group name
    cats : List[str]
        Categories
    color_palette : List[str]
        List of colors
    figsize : Tuple[int, int]
        Figure size
    graph_title : str, optional
        Graph title, by default ""

    Returns
    -------
    Figure
        ROC curve figure
    """
    fig, ax = draw_curve_group(
        curves_dict,
        "ROC",
        "fpr",
        "tpr",
        group_name,
        cats,
        color_palette,
        figsize,
        graph_title,
    )
    agg_aucs = agg_auc_group(metrics_df, "auroc", group_name, cats)
    fig = add_curve_errors(
        agg_aucs, get_pretty_name_metric("auroc"), "auroc", color_palette, fig, ax
    )
    return fig


def draw_prc_group(
    metrics_df: pd.DataFrame,
    curves_dict: Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]],
    x_name: str,
    y_name: str,
    group_name: str,
    cats: List[str],
    color_palette: List[str],
    figsize: Tuple[int, int],
    graph_title: str = "",
) -> Figure:
    """Draw Precision-Recall curve for each category.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe for a grouping
    curves_dict : Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]]
        Dictionary containing curve points for each grouping, category and curve type
    x_name: str
        Type of recall (either event_recall or recall)
    y_name: str
        Type of precision (either corrected_precision or precision)
    group_name : str
        Group name
    cats : List[str]
        Categories
    color_palette : List[str]
        List of colors
    figsize : Tuple[int, int]
        Figure size
    graph_title : str, optional
        Graph title, by default ""

    Returns
    -------
    Figure
        PR curve figure
    """
    fig, ax = draw_curve_group(
        curves_dict,
        "PRC",
        x_name,
        y_name,
        group_name,
        cats,
        color_palette,
        figsize,
        graph_title,
        sl=slice(None, None, -1),
    )
    metric_name = "auprc"
    if x_name == "event_recall":
        metric_name = "event_" + metric_name
    if y_name == "corrected_precision":
        metric_name = "corrected_" + metric_name
    agg_aucs = agg_auc_group(metrics_df, metric_name, group_name, cats)
    fig = add_curve_errors(
        agg_aucs, get_pretty_name_metric(metric_name), "auprc", color_palette, fig, ax
    )
    return fig


def draw_curve_group(
    curves_dict: Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]],
    curve_type: str,
    x_name: str,
    y_name: str,
    group_name: str,
    cats: List[str],
    color_palette: List[str],
    figsize: Tuple[str, str],
    graph_title: str = "",
    sl: slice = slice(None, None, 1),
) -> Tuple[Figure, Axes]:
    """Draw curve specified by curve_type for each category.

    Parameters
    ----------
    curves_dict : Dict[str, Dict[str, List[Dict[str, Dict[str, np.array]]]]]
        Dictionary containing curve points for each grouping, category and curve type
    curve_type: str
        Type of curve (either calibration, ROC or PRC)
    x_name: str
        Metric on x axis
    y_name: str
        Metric on y axis
    group_name : str
        Group name
    cats : List[str]
        Categories
    color_palette : List[str]
        List of colors
    figsize : Tuple[int, int]
        Figure size
    graph_title : str, optional
        Graph title, by default ""

    Returns
    -------
    Figure
        ROC curve figure
    """
    fig, ax = plt.subplots(nrows=1, figsize=figsize)
    for c, cat in enumerate(cats):
        x_grid = None
        y_arr = []
        n_bootstrap = len(curves_dict[group_name][cat])
        for i in range(n_bootstrap):
            x = curves_dict[group_name][cat][i][curve_type][x_name][sl]
            y = curves_dict[group_name][cat][i][curve_type][y_name][sl]
            if x_grid is None:
                x_grid = x
            y_interp = np.interp(x_grid, x, y)
            y_arr.append(y_interp)
        mean_y = np.mean(y_arr, axis=0)
        std_y = np.std(y_arr, axis=0)
        y_lower = np.maximum(mean_y - std_y, 0)
        y_upper = np.minimum(mean_y + std_y, 1)
        plt.plot(x_grid[sl], mean_y[sl], label=cat, c=color_palette[c])
        plt.fill_between(x_grid[sl], y_lower[sl], y_upper[sl], alpha=0.2, color=color_palette[c])
    plt.title(graph_title)
    if x_name in PRETTY_NAME_AXIS:
        plt.xlabel(PRETTY_NAME_AXIS[x_name])
    else:
        plt.xlabel(x_name)
    if y_name in PRETTY_NAME_AXIS:
        plt.ylabel(PRETTY_NAME_AXIS[y_name])
    else:
        plt.ylabel(y_name)
    return fig, ax


def plot_boxplot_metrics_pw(
    metrics_df: pd.DataFrame,
    group_name: str,
    cats: List[str],
    metrics_name: List[str],
    figsize: Tuple[int, int] = (4, 5),
    do_stat_test: bool = False,
    type_table_groups: Dict[str, List[str]] = None,
    df_stat_test: pd.DataFrame = None,
    dict_worst_cats: Dict[str, List[str]] = None,
    metrics_xlim: Dict[str, Tuple[float, float]] = None,
    color_palette: List[str] = None,
) -> List[Figure]:
    """Plot box plot of metrics two by two.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe
    group_name : str
        Group name
    cats: List[str]
        List of categories
    metrics_name : List[str]
        List of metric names
    figsize : Tuple[int, int], optional
        Figure size, by default (4, 5)
    do_stat_test : bool, optional
        Flag stating whether we do statistical test, by default False
    type_table_groups: Dict[str, List[str]], optional
        Dictionary matching each of three types of comparison to its list of groupings
    df_stat_test : pd.DataFrame, optional
        Dataframe with results of statistical test, by default None
    dict_worst_cats : Dict[str, List[str]], optional
        Dictionary mapping each metric to the top k worst categories (in terms of delta in performance), by default None
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
            group_name,
            cats,
            axs[0],
            do_stat_test,
            type_table_groups,
            df_stat_test,
            dict_worst_cats,
            metrics_xlim,
            color_palette,
        )
        if len(metrics_name) > it * 2 + 1:
            axs[1] = draw_boxplot_metric(
                metrics_df,
                metrics_name[it * 2 + 1],
                group_name,
                cats,
                axs[1],
                do_stat_test,
                type_table_groups,
                df_stat_test,
                dict_worst_cats,
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
    group_name: str,
    cats: List[str],
    ax: Axes,
    do_stat_test: bool = False,
    type_table_groups: Dict[str, List[str]] = None,
    df_stat_test: pd.DataFrame = None,
    dict_worst_cats: Dict[str, List[str]] = None,
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
    group_name : str
        Group name
    cats : List[str]
        List of categories
    ax : Axes
        Axis
    do_stat_test : bool, optional
        Flag stating whether we do statistical test, by default False
    type_table_groups: Dict[str, List[str]], optional
        Dictionary matching each of three types of comparison to its list of groupings
    df_stat_test : pd.DataFrame, optional
        Dataframe with results of statistical test, by default None
    dict_worst_cats : Dict[str, List[str]], optional
        Dictionary mapping each metric to the top k worst categories (in terms of delta in performance), by default None
    metrics_xlim : Dict[str, Tuple[float, float]], optional
        Dictionary of x-axis limit for each metric, by default None
    color_palette : List[str], optional
        List of colors, by default None

    Returns
    -------
    Axes
        Box plot figure for a metric
    """
    sns.boxplot(data=metrics_df, x=metric, y=group_name, ax=ax, palette=color_palette)
    pretty_name_metric = get_pretty_name_metric(metric)
    ax.set_title(pretty_name_metric)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    if metrics_xlim and metric in metrics_xlim:
        ax.set(xlim=metrics_xlim[metric])

    if do_stat_test:
        if group_name in type_table_groups["binary_group"]:
            if len(df_stat_test[df_stat_test["Metric"] == pretty_name_metric]):
                cat = df_stat_test[df_stat_test["Metric"] == pretty_name_metric].iloc[0][
                    "Cohort with the worst metric"
                ]
                i = cats.index(cat)
                draw_star(metrics_df, dict_worst_cats, metric, group_name, cat, True, i, ax)
        else:
            significantly_diff_cats = df_stat_test[
                (df_stat_test["Metric"] == pretty_name_metric)
                & (df_stat_test["Cohort vs. rest"] == "worse")
            ]["Category"]
            for i, cat in enumerate(cats):
                if cat in list(significantly_diff_cats):
                    draw_star(metrics_df, dict_worst_cats, metric, group_name, cat, False, i, ax)

    return ax


def draw_star(
    metrics_df: pd.DataFrame,
    dict_worst_cats: Dict[str, List[str]],
    metric: str,
    group_name: str,
    cat: str,
    is_binary_group: bool,
    i_box: int,
    ax: Axes,
):
    """_summary_

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Metrics dataframe
    dict_worst_cats : Dict[str, List[str]]
        Dictionary mapping each metric to the top k worst categories (in terms of delta in performance)
    metric : str
        Metrics name
    group_name : str
        Group name
    cat : str
        List of categories
    is_binary_group : bool
        Flag stating whether the considered grouping is binary
    i_box : int
        Index of box
    ax : Axes
        Axis
    """
    int_ticks = ax.get_xticks()[1] - ax.get_xticks()[0]
    x_place = metrics_df[metrics_df[group_name] == cat][metric].quantile(0.75) + int_ticks * 0.05
    if is_binary_group:
        y_place = i_box - 0.25
    else:
        y_place = i_box - 0.15
    if cat in dict_worst_cats[metric]:
        color_star = "r"
    else:
        color_star = "k"
    ax.text(x_place, y_place, "*", fontweight="extra bold", c=color_star)
