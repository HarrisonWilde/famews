import logging
from pathlib import Path
from typing import Dict, List, Tuple

import gin
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Table,
)
from reportlab.platypus.tableofcontents import TableOfContents

from famews.fairness_check.medical_variables.build_table import get_worst_medvars_delta
from famews.fairness_check.medical_variables.draw_graph import plot_boxplot_medvars_pw
from famews.fairness_check.missingness.build_table import (
    generate_table_perf_missingness,
)
from famews.fairness_check.missingness.draw_graph import (
    draw_intensity_bar_plot,
    plot_boxplot_missingness_metrics_pw,
)
from famews.fairness_check.model_performance.build_table import (
    get_worst_metrics_performance,
)
from famews.fairness_check.model_performance.draw_graph import (
    draw_calibration_group,
    draw_roc_group,
    plot_boxplot_metrics_pw,
)
from famews.fairness_check.report_generation.helper_feat_importance import (
    df2table_rbo,
    generate_table_topk_feat,
    get_rbo_table,
    get_topk_ranking_critical_per_group,
    get_topk_ranking_per_group,
)
from famews.fairness_check.report_generation.helper_glossary import (
    generate_glossary_feature_importance,
    generate_glossary_general,
    generate_glossary_medvars,
    generate_glossary_missingness,
    generate_glossary_model_performance,
    generate_glossary_timegap,
)
from famews.fairness_check.report_generation.helper_medvars import (
    get_stat_test_table_medvars,
)
from famews.fairness_check.report_generation.helper_missingness import (
    get_stat_test_intensity_msrt,
    summarize_intensity_msrt,
    summarize_stat_test_perf,
)
from famews.fairness_check.report_generation.helper_model_performance import (
    generate_table_ratio_stat_test,
    get_PRC_figure,
    get_stat_test_table_performance,
    get_table_summarized_metrics_group,
)
from famews.fairness_check.report_generation.helper_timegap import (
    get_stat_test_table_timegap,
    get_summary_timegap_mean,
    get_table_summary_timegap_group,
)
from famews.fairness_check.report_generation.utils import (
    TOC_STYLE,
    MyDocTemplate,
    df2table,
    double_fig2image,
    fig2image,
    round_down,
    round_up,
)
from famews.fairness_check.timegap_alarm_event.build_table import get_worst_timegap
from famews.fairness_check.timegap_alarm_event.draw_graph import (
    plot_boxplot_timegap_pw,
)
from famews.pipeline import PipelineState, StatefulPipelineStage


def on_maintitle_page(canvas, doc, pagesize=A4):
    canvas.setFont(
        "Courier-Bold",
        30,
    )
    canvas.drawCentredString(pagesize[0] / 2, pagesize[1] / 2, "Fairness Analysis Report")


def on_page(canvas, doc, pagesize=A4):
    page_num = canvas.getPageNumber()
    canvas.drawCentredString(pagesize[0] - 30, 30, str(page_num))


@gin.configurable("CreatePDFReport", denylist=["state"])
class CreatePDFReport(StatefulPipelineStage):

    name = "Create PDF Report"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        display_stat_test_table: bool = True,
        colors: List[str] = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:pink",
            "tab:brown",
        ],
        figsize_cal_curve: Tuple[int, int] = (6, 4),
        figsize_curves: Tuple[int, int] = (4, 3.75),
        figsize_boxplot_metrics: Tuple[int, int] = (8, 3),
        figsize_barplot_intensity: Tuple[int, int] = (7, 4),
        k_feat_importance: int = 15,
        max_cols_topkfeat_table: int = 4,
        **kwargs,
    ):
        """Pipeline stage to create PDF report.

        Parameters
        ----------
        state : PipelineState
            Pipeline state
        num_workers : int, optional
            Number of workers, by default 1
        display_stat_test_table : bool, optional
            Flag stating whether we display the statistical test tables, by default True
        colors : List[str], optional
            List of colors, by default [ "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink", "tab:brown", ]
        figsize_cal_curve : Tuple[int, int], optional
            Size of calibration figure, by default (6, 4)
        figsize_curves : Tuple[int, int], optional
            Size of other curves figure , by default (4, 3.75)
        figsize_boxplot_metrics : Tuple[int, int], optional
            Size of boxplot metrics figure, by default (8, 3)
        figsize_barplot_intensity : Tuple[int, int], optional
            Size of barplot of intensity of measurements figure, by default (7, 4)
        k_feat_importance: int, optional
            Number of most important features to display (while running the feature importance analysis stage), by default 15
        max_cols_topkfeat_table: int, optional
            Maximum number of columns to display next to each other for the feature ranking table, by default 4
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.pdf_path = Path(self.state.fairness_log_dir) / "report.pdf"
        self.padding = dict(leftPadding=72, rightPadding=72, topPadding=36, bottomPadding=18)
        self.frame = Frame(0, 0, *A4, **self.padding)
        self.maintitle_template = PageTemplate(
            id="title", frames=self.frame, onPage=on_maintitle_page, pagesize=A4
        )
        self.template = PageTemplate(id="portrait", frames=self.frame, onPage=on_page, pagesize=A4)
        self.style = getSampleStyleSheet()
        self.display_stat_test_table = display_stat_test_table
        self.colors = colors
        self.figsize_cal_curve = figsize_cal_curve
        self.figsize_curves = figsize_curves
        self.figsize_boxplot_metrics = figsize_boxplot_metrics
        self.figsize_barplot_intensity = figsize_barplot_intensity
        self.k_feat_importance = k_feat_importance
        self.max_cols_topkfeat_table = max_cols_topkfeat_table
        self.list_metrics_performance = []
        self.list_metrics_missingness = []
        self.section_counter = 1
        self.style.add(
            ParagraphStyle(
                name="toc_title", parent=self.style["Normal"], fontName="Courier-Bold", fontSize=20
            )
        )
        self.style.add(
            ParagraphStyle(
                name="agg_title",
                parent=self.style["Normal"],
                fontName="Helvetica-Bold",
                fontSize=11,
                spaceAfter=6,
                spaceBefore=8,
            )
        )
        self.style.add(
            ParagraphStyle(
                name="sub_title",
                parent=self.style["Normal"],
                fontName="Helvetica-Bold",
                fontSize=13,
                spaceAfter=6,
                spaceBefore=12,
            )
        )
        self.style.add(
            ParagraphStyle(
                name="grouping_title",
                parent=self.style["Heading4"],
                spaceAfter=4,
                spaceBefore=4,
            )
        )
        self.style.add(ParagraphStyle(name="table_title", parent=self.style["Normal"], fontSize=8))

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        """Check if the current `PipelineState` has been run

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        return False

    def run(self):
        """
        Create PDF report.
        """
        doc = MyDocTemplate(
            str(self.pdf_path.absolute()), pageTemplates=[self.maintitle_template, self.template]
        )
        toc = TableOfContents()
        toc.levelStyles = TOC_STYLE
        story = [Paragraph(""), NextPageTemplate("portrait"), PageBreak()]
        story += [
            Paragraph("Table of Contents:", self.style["toc_title"]),
            Paragraph("<br />\n <br />"),
            toc,
            PageBreak(),
        ]
        story += self.create_report_patient_info()
        if hasattr(self.state, "metrics_group_df") and self.state.metrics_group_df is not None:
            # add description here
            story.append(PageBreak())
            story += self.create_report_model_performance()
        if hasattr(self.state, "timegap_group_df") and self.state.timegap_group_df is not None:
            story.append(PageBreak())
            story += self.create_report_timegap()
        if hasattr(self.state, "medvars_group_df") and self.state.medvars_group_df is not None:
            story.append(PageBreak())
            story += self.create_report_medvars()
        if hasattr(self.state, "feat_ranking_all") and self.state.feat_ranking_all is not None:
            story.append(PageBreak())
            story += self.create_report_feat_importance()
        if (
            hasattr(self.state, "missingness_performance_df")
            and self.state.missingness_performance_df is not None
        ):
            story.append(PageBreak())
            story += self.create_report_missingness()
        story += self.create_glossary()
        doc.multiBuild(story)

    def create_report_patient_info(self) -> list:
        """Create report for patient information.

        Returns
        -------
        list
            List of elements to append to the report
        """
        story = [
            Paragraph(
                f"{self.section_counter}. Information about test dataset", self.style["Heading1"]
            )
        ]
        counter_table = ord("a")
        for group_name in self.state.group_size:
            df_group = (
                pd.DataFrame(self.state.group_size[group_name])
                .T.reset_index()
                .rename(
                    columns={
                        "index": "Category",
                        "size": "Number of patients",
                        "count_w_event": "Number of patients with event",
                    }
                )
            )
            story.append(Paragraph(f"Grouping by {group_name}", self.style["Heading4"]))
            story.append(
                Paragraph(
                    f"<u>Table {self.section_counter}.{chr(counter_table)}</u>",
                    self.style["table_title"],
                )
            )
            counter_table += 1
            story.append(df2table(df_group))
            if group_name in self.state.removed_groups:
                string_cats = ", ".join(self.state.removed_groups[group_name])
                story.append(
                    Paragraph(
                        f"The following categories have been removed for the rest of the analysis because they didn't contain enough patients with event: {string_cats}"
                    )
                )
        self.section_counter += 1
        return story

    def create_report_model_performance(self) -> list:
        """Create report for model performance per cohort.

        Returns
        -------
        list
            List of elements to append to the report
        """
        self.state.metrics_group_df = self.state.metrics_group_df.dropna(
            axis=1, how="all"
        )  # remove metrics that are only NaNs
        self.list_metrics_performance = self.state.metrics_group_df.columns[:-3]
        metrics_xlim = {
            metric: (
                round_down(min(self.state.metrics_group_df[metric])),
                round_up(max(self.state.metrics_group_df[metric])),
            )
            for metric in self.list_metrics_performance
        }
        worst_cats_metrics = {metric: [] for metric in self.list_metrics_performance}
        dict_worst_cats = None
        stat_test_groups = {group_name: None for group_name in self.state.groups.keys()}
        story = [
            Paragraph(f"{self.section_counter}. Model Performance Analysis", self.style["Heading1"])
        ]
        subsection_counter = 1
        subsubsection_counter = 1
        story.append(
            Paragraph(
                "Goal: Comparing the model performance across cohorts of patients\n",
                self.style["Heading4"],
            )
        )
        story.append(
            Paragraph(
                f"Binary metrics computed with a threshold on score of {round(self.state.threshold, 3)}.\n"
            )
        )
        story.append(
            (
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}. Aggregated views",
                    self.style["Heading2"],
                )
            )
        )
        story.append(
            Paragraph(
                f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. Summarized performance metrics per grouping",
                self.style["sub_title"],
            )
        )
        counter_table = ord("a")
        for group_name, cats in self.state.groups.items():
            story.append(Paragraph(f"Grouping by {group_name}", self.style["Heading4"]))
            cats_size = {cat: dc["size"] for cat, dc in self.state.group_size[group_name].items()}
            metrics_df = self.state.metrics_group_df[
                (self.state.metrics_group_df["group"] == group_name)
                & (self.state.metrics_group_df["cat"].isin(cats))
            ]
            str_minority_cat, df_summary_metrics = get_table_summarized_metrics_group(
                metrics_df, self.list_metrics_performance, cats_size
            )
            story.append(Paragraph(str_minority_cat))
            story.append(Paragraph("<br/>"))
            story.append(
                Paragraph(
                    f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                    self.style["table_title"],
                )
            )
            counter_table += 1
            story.append(df2table(df_summary_metrics))
        subsubsection_counter += 1
        if self.state.do_stat_test:
            for group_name, cats in self.state.groups.items():
                df_stat_test, worst_cats_metrics = get_stat_test_table_performance(
                    self.state.metrics_group_df,
                    self.state.type_table_groups,
                    group_name,
                    cats,
                    self.list_metrics_performance,
                    worst_cats_metrics,
                    self.state.significance_level,
                    self.state.filter_delta,
                )
                stat_test_groups[group_name] = df_stat_test
            story.append(PageBreak())
            (
                tables_ratio_group,
                summary_strings_group,
                global_summary_string,
            ) = generate_table_ratio_stat_test(
                self.state.groups,
                stat_test_groups,
                self.state.type_table_groups,
                len(self.list_metrics_performance),
            )
            story.append(
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. Summary view based on the ratio of significantly worse metrics",
                    self.style["sub_title"],
                )
            )
            counter_table = ord("a")
            story.append(
                Paragraph("<b>We first show an overview of this analysis over all groupings.</b>")
            )
            story.append(Paragraph("<br />\n"))
            story += [Paragraph(global_summary_string[0]), Paragraph(global_summary_string[1])]
            story.append(Paragraph("<br />\n <br />"))
            story.append(
                Paragraph(
                    "In the following tables, we display the ratio of significantly worse metrics (over the total number of analysed performance metrics) for each category of patients."
                )
            )
            for group_name, cats in self.state.groups.items():
                story.append(Paragraph(f"Grouping by {group_name}", self.style["Heading4"]))
                story += [
                    Paragraph(summary_strings_group[group_name][0]),
                    Paragraph(summary_strings_group[group_name][1]),
                ]
                story.append(Paragraph("<br /><br />"))
                story.append(
                    Paragraph(
                        f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                        self.style["table_title"],
                    )
                )
                story.append(df2table(tables_ratio_group[group_name]))
                counter_table += 1
            table_worst_cats, dict_worst_cats = get_worst_metrics_performance(worst_cats_metrics)
            story.append(PageBreak())
            subsubsection_counter += 1
            story.append(
                Paragraph(
                    f" {self.section_counter}.{subsection_counter}.{subsubsection_counter}. Top 3 cohorts with the biggest performance metric discrepancies",
                    self.style["sub_title"],
                )
            )
            counter_table = ord("a")
            story.append(
                Paragraph(
                    "In the following table, we show for each performance metric the 3 cohorts with the biggest delta that are significantly worse off than the rest of the patients. If some cells are empty, this means that there are less than 3 cohorts, possibly none, that are significantly worse than the rest of the patients for this particular metric."
                )
            )
            story.append(Paragraph("<br />\n <br />"))
            story.append(
                Paragraph(
                    f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                    self.style["table_title"],
                )
            )
            story.append(df2table(table_worst_cats))
        story.append(PageBreak())
        subsection_counter += 1
        subsubsection_counter = 1
        story.append(
            (
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}. Grouping by ",
                    self.style["Heading2"],
                )
            )
        )
        if self.state.do_stat_test:
            story.append(
                Paragraph(
                    "For each grouping, we display box plots that show the performance metrics' distributions for the different categories of patients. For each metric, we emphasize with a black star the cohorts that are significantly worse off compared to the rest of the patients and with a red star the cohorts that appear in the table <b>Top 3 cohorts with the biggest performance metric discrepancies</b>."
                )
            )
            story.append(
                Paragraph(
                    f"For each grouping, we propose a table that presents the results of the statistical analysis: comparing the different performance metrics for a cohort against the rest of the patients. P-values are obtained by running the Mann-Whitney U test with Bonferroni correction. We display only metrics and cohorts with a significant p-value (smaller than {self.state.significance_level}/number of comparisons) and whose delta is bigger than {self.state.filter_delta}. For binary grouping, we display the category with the worst distribution for each metric. While for multicategorical grouping, we display whether the distribution for the category is better or worse than for the rest of patients\n"
                )
            )
        else:
            story.append(
                Paragraph(
                    "For each grouping, we display box plots that show the performance metrics' distributions for the different categories of patients."
                )
            )
        if hasattr(self.state, "curves_group") and self.state.curves_group is not None:
            story.append(
                Paragraph(
                    "We also display the calibration curve for each grouping's categories as well as the curves corresponding to each score-based metrics."
                )
            )
        for group_name, cats in self.state.groups.items():
            story.append(
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. ... {group_name}",
                    self.style["sub_title"],
                )
            )
            counter_table = ord("a")
            counter_figure = ord("a")
            metrics_df = self.state.metrics_group_df[
                self.state.metrics_group_df["group"] == group_name
            ].rename(columns={"cat": group_name})
            figs_metrics = plot_boxplot_metrics_pw(
                metrics_df,
                group_name,
                cats,
                self.list_metrics_performance,
                figsize=self.figsize_boxplot_metrics,
                do_stat_test=self.state.do_stat_test,
                type_table_groups=self.state.type_table_groups,
                df_stat_test=stat_test_groups[group_name],
                dict_worst_cats=dict_worst_cats,
                metrics_xlim=metrics_xlim,
                color_palette=self.colors,
            )
            story.append(
                Paragraph(
                    f"<u>Figure {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_figure)}</u>",
                    self.style["table_title"],
                )
            )
            story += [fig2image(fig) for fig in figs_metrics]
            counter_figure += 1
            if self.state.do_stat_test and self.display_stat_test_table:
                story.append(
                    Paragraph(
                        f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                        self.style["table_title"],
                    )
                )
                story.append(df2table(stat_test_groups[group_name]))
            if hasattr(self.state, "curves_group") and self.state.curves_group is not None:
                story.append(Paragraph("<br />\n <br />"))
                if "calibration_error" in self.list_metrics_performance:
                    fig_calibration = draw_calibration_group(
                        self.state.curves_group,
                        group_name,
                        cats,
                        self.colors,
                        self.figsize_cal_curve,
                        "Calibration curve",
                    )
                    story.append(
                        Paragraph(
                            f"<u>Figure {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_figure)}</u>",
                            self.style["table_title"],
                        )
                    )
                    story.append(fig2image(fig_calibration))
                    counter_figure += 1
                if "auroc" in self.list_metrics_performance:
                    fig_roc = draw_roc_group(
                        metrics_df,
                        self.state.curves_group,
                        group_name,
                        cats,
                        self.colors,
                        self.figsize_curves,
                        "ROC curve",
                    )
                    story.append(
                        Paragraph(
                            f"<u>Figure {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_figure)}</u>",
                            self.style["table_title"],
                        )
                    )
                    story.append(fig2image(fig_roc))
                    counter_figure += 1
                if (
                    "auprc" in self.list_metrics_performance
                    or "corrected_auprc" in self.list_metrics_performance
                    or "event_auprc" in self.list_metrics_performance
                    or "corrected_event_auprc" in self.list_metrics_performance
                ):
                    story.append(
                        Paragraph(
                            f"<u>Figure {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_figure)}</u>",
                            self.style["table_title"],
                        )
                    )
                    counter_figure += 1
                if (
                    "auprc" in self.list_metrics_performance
                    or "corrected_auprc" in self.list_metrics_performance
                ):
                    story.append(
                        get_PRC_figure(
                            metrics_df,
                            self.state.curves_group,
                            group_name,
                            cats,
                            self.list_metrics_performance,
                            event_based=False,
                            colors=self.colors,
                            figsize_curves=self.figsize_curves,
                        )
                    )
                    counter_figure += 1
                if (
                    "event_auprc" in self.list_metrics_performance
                    or "corrected_event_auprc" in self.list_metrics_performance
                ):
                    story.append(
                        get_PRC_figure(
                            metrics_df,
                            self.state.curves_group,
                            group_name,
                            cats,
                            self.list_metrics_performance,
                            event_based=True,
                            colors=self.colors,
                            figsize_curves=self.figsize_curves,
                        )
                    )

            story.append(PageBreak())
            subsubsection_counter += 1
        self.section_counter += 1
        return story

    def create_report_timegap(self) -> list:
        """Create report for time gap between alarm and event comparison per group.

        Returns
        -------
        list
            List of elements to append to the report
        """
        story = [Paragraph(f"{self.section_counter}. Time Gap Analysis", self.style["Heading1"])]
        subsection_counter = 1
        subsubsection_counter = 1
        story.append(
            Paragraph(
                "Goal: Checking whether the time gap between the first correct alarm and the start of the corresponding event are similar across cohorts of patients\n",
                self.style["Heading4"],
            )
        )
        worst_cats_timegap = {gp: [] for gp in self.state.list_group_start_event}
        dict_worst_cats = None
        stat_test_groups = {group_name: None for group_name in self.state.groups.keys()}
        story.append(
            (
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}. Aggregated views",
                    self.style["Heading2"],
                )
            )
        )
        story.append(
            Paragraph(
                f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. Summary statistics of median time gap per grouping",
                self.style["sub_title"],
            )
        )
        for gp in self.state.list_group_start_event:
            story.append(Paragraph(get_summary_timegap_mean(self.state.timegap_group_df, gp)))
        counter_table = ord("a")
        for group_name, cats in self.state.groups.items():
            story.append(Paragraph(f"Grouping by {group_name}", self.style["Heading4"]))
            cats_size = {cat: dc["size"] for cat, dc in self.state.group_size[group_name].items()}
            timegap_df = self.state.timegap_group_df[
                self.state.timegap_group_df["group"] == group_name
            ]
            df_summary_timegap = get_table_summary_timegap_group(
                timegap_df, self.state.list_group_start_event, cats_size
            )
            story.append(
                Paragraph(
                    f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                    self.style["table_title"],
                )
            )
            story.append(df2table(df_summary_timegap))
            counter_table += 1
        subsubsection_counter += 1
        if self.state.do_stat_test:
            for group_name, cats in self.state.groups.items():
                df_stat_test, worst_cats_timegap = get_stat_test_table_timegap(
                    self.state.timegap_group_df,
                    self.state.type_table_groups,
                    group_name,
                    cats,
                    self.state.list_group_start_event,
                    worst_cats_timegap,
                    self.state.significance_level,
                    self.state.filter_delta,
                )
                stat_test_groups[group_name] = df_stat_test
            story.append(Paragraph("<br />\n<br />"))

            table_worst_cats, dict_worst_cats = get_worst_timegap(worst_cats_timegap)
            story.append(
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. Top 3 cohorts with the biggest time gap discrepancies",
                    self.style["sub_title"],
                )
            )
            counter_table = ord("a")
            story.append(
                Paragraph(
                    "In the following table, we show for each start of the event window the 3 cohorts with the biggest delta that are significantly worse off than the rest of the patients. If some cells are empty, this means that there are fewer than 3 cohorts, possibly none, that are significantly worse than the rest of the patients for this particular start of the event window."
                )
            )
            story.append(Paragraph("<br />\n<br />"))
            story.append(
                Paragraph(
                    f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                    self.style["table_title"],
                )
            )
            story.append(df2table(table_worst_cats))
        story.append(PageBreak())
        subsection_counter += 1
        subsubsection_counter = 1
        story.append(
            (
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}. Grouping by",
                    self.style["Heading2"],
                )
            )
        )
        if self.state.do_stat_test:
            story.append(
                Paragraph(
                    "For each grouping, we display box plots that show the median time gap between alarm and event for the different categories of patients depending on the period of the stay when the event began. For each start of event window, we emphasize with a black star the cohorts that are significantly worse off compared to the rest of the patients and with a red star the cohorts that appear in the table <b>Top 3 cohorts with the biggest time gap discrepancies</b>."
                )
            )
            story.append(
                Paragraph(
                    f"For each grouping, we propose a table that presents the results of the statistical analysis: comparing the time gap from alarm to event for one cohort against the rest of the patients. P-values are obtained by running the Mann-Whitney U test with Bonferroni correction. We display only start of event windows and cohorts with a significant p-value (smaller than {self.state.significance_level}/number of comparisons) and whose delta is bigger than {self.state.filter_delta}. For binary grouping, we display the category with the worst time gap distribution for each start of event window. While for multicategorical grouping we display whether the distribution for the category is better or worse than for the rest of patients\n"
                )
            )
        else:
            story.append(
                Paragraph(
                    "For each grouping, we display box plots that show the median time gap between alarm and event for the different categories of patients depending on the period of the stay when the event began."
                )
            )
        for group_name, cats in self.state.groups.items():
            story.append(
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. ... {group_name}",
                    self.style["sub_title"],
                )
            )
            counter_table = ord("a")
            counter_figure = ord("a")
            timegap_df = self.state.timegap_group_df[
                self.state.timegap_group_df["group"] == group_name
            ].rename(columns={"cat": group_name})
            figs_metrics = plot_boxplot_timegap_pw(
                timegap_df,
                group_name,
                cats,
                self.state.list_group_start_event,
                figsize=self.figsize_boxplot_metrics,
                do_stat_test=self.state.do_stat_test,
                type_table_groups=self.state.type_table_groups,
                df_stat_test=stat_test_groups[group_name],
                dict_worst_cats=dict_worst_cats,
                timegap_xlim=self.state.timegap_xlim,
                color_palette=self.colors,
            )
            story.append(
                Paragraph(
                    f"<u>Figure {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_figure)}</u>",
                    self.style["table_title"],
                )
            )
            story += [fig2image(fig) for fig in figs_metrics]
            if self.state.do_stat_test:
                story.append(
                    Paragraph(
                        f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                        self.style["table_title"],
                    )
                )
                story.append(df2table(stat_test_groups[group_name]))
            story.append(PageBreak())
            subsubsection_counter += 1
        self.section_counter += 1
        return story

    def create_report_medvars(self) -> list:
        """Create report for analysis of median medical variable per cohort.

        Returns
        -------
        list
            List of elements to append to the report
        """
        medvars_xlim = {
            medvar: (
                round_down(
                    min(
                        [
                            min(self.state.medvars_group_df[medvar]),
                            min(self.state.medvars_group_df[medvar + "_not_inevent"]),
                            min(self.state.medvars_group_df[medvar + "_never_inevent"]),
                        ]
                    )
                ),
                round_up(
                    max(
                        [
                            max(self.state.medvars_group_df[medvar]),
                            max(self.state.medvars_group_df[medvar + "_not_inevent"]),
                            max(self.state.medvars_group_df[medvar + "_never_inevent"]),
                        ]
                    )
                ),
            )
            for medvar in self.state.list_medvars
        }
        story = [
            Paragraph(f"{self.section_counter}. Medical Variable Analysis", self.style["Heading1"])
        ]
        subsection_counter = 1
        subsubsection_counter = 1
        story.append(
            Paragraph(
                "Goal: Comparing the median value of relevant medical variables across cohorts\n",
                self.style["Heading4"],
            )
        )
        story.append(
            Paragraph(f"We check the following variables: {', '.join(self.state.list_medvars)}")
        )
        var_suffixes = ["", "_not_inevent", "_never_inevent"]
        worst_cats_medvars = {
            (var, suffix): [] for var in self.state.list_medvars for suffix in var_suffixes
        }
        dict_worst_cats = None
        stat_test_groups = {group_name: None for group_name in self.state.groups.keys()}
        if self.state.do_stat_test:
            for group_name, cats in self.state.groups.items():
                df_stat_test, worst_cats_medvars = get_stat_test_table_medvars(
                    self.state.medvars_group_df,
                    self.state.type_table_groups,
                    group_name,
                    cats,
                    self.state.list_medvars,
                    var_suffixes,
                    worst_cats_medvars,
                    self.state.significance_level,
                    self.state.filter_delta,
                )
                stat_test_groups[group_name] = df_stat_test
            story.append(
                (
                    Paragraph(
                        f"{self.section_counter}.{subsection_counter}. Aggregated views",
                        self.style["Heading2"],
                    )
                )
            )
            table_worst_cats, dict_worst_cats = get_worst_medvars_delta(
                worst_cats_medvars, medvars_units=self.state.medvars_units
            )
            story.append(
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. Top 3 cohorts with the biggest differences in the medical variables distributions",
                    self.style["sub_title"],
                )
            )
            counter_table = ord("a")
            story.append(
                Paragraph(
                    "In the following table, for each of the selected medical variables and median computation condition, we show the 3 cohorts with the biggest delta that are significantly different than the rest of the patients. If some cells are empty, that means that there are less than 3 cohorts (possibly none) that are significantly different than the rest of the patients for this particular medical variable and median computation condition."
                )
            )
            story.append(Paragraph("<br />\n<br />"))
            story.append(
                Paragraph(
                    f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                    self.style["table_title"],
                )
            )
            story.append(df2table(table_worst_cats))
            subsection_counter += 1
            subsubsection_counter = 1
        story.append(PageBreak())
        story.append(
            (
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}. Grouping by",
                    self.style["Heading2"],
                )
            )
        )
        if self.state.do_stat_test:
            story.append(
                Paragraph(
                    "For each grouping, we display box plots that show the median value of the selected medical variables for three conditions: all time points during the entire stay, time points while not in an event, and time points from patients not experiencing any event. For each variable and condition, we emphasize with a black star the cohorts that are significantly different compared to the rest of the patients and with a red star the cohorts that appear in the table <b>Top 3 cohorts with the biggest differences in the medical variables values</b>."
                )
            )
            story.append(
                Paragraph(
                    f"For each grouping, we propose a table that presents the results of the statistical analysis: comparing the medical variables' median value for one cohort against the rest of the patients. P-values are obtained by running the Mann-Whitney U test with Bonferroni correction. We display only medical variables and cohorts with a significant p-value (smaller than {self.state.significance_level}/number of comparisons) and whose delta is bigger than {self.state.filter_delta}. For binary grouping, we display the category with the greatest median value for each of the selected medical variables and median computation condition. While for multicategorical grouping we display whether the median value for the category is greater or less than for the rest of patients\n"
                )
            )
        else:
            story.append(
                Paragraph(
                    "For each grouping, we display box plots that show the median value of the selected medical variables for three conditions: all time points during the entire stay, time points while not in an event, and time points from patients not experiencing any event.\n"
                )
            )
        for group_name, cats in self.state.groups.items():
            story.append(
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. ... {group_name}",
                    self.style["sub_title"],
                )
            )
            counter_table = ord("a")
            counter_figure = ord("a")
            medvars_df = self.state.medvars_group_df[
                self.state.medvars_group_df["group"] == group_name
            ].rename(columns={"cat": group_name})
            if self.state.do_stat_test and self.state.type_table_groups["binary_group"]:
                is_binary_group = group_name in self.state.type_table_groups["binary_group"]
            else:
                is_binary_group = None
            figs_metrics = plot_boxplot_medvars_pw(
                medvars_df,
                group_name,
                cats,
                self.state.list_medvars,
                var_suffixes,
                figsize=self.figsize_boxplot_metrics,
                do_stat_test=self.state.do_stat_test,
                is_binary_group=is_binary_group,
                df_stat_test=stat_test_groups[group_name],
                dict_worst_cats=dict_worst_cats,
                medvars_xlim=medvars_xlim,
                color_palette=self.colors,
            )
            story.append(
                Paragraph(
                    f"<u>Figure {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_figure)}</u>",
                    self.style["table_title"],
                )
            )
            story += [fig2image(fig) for fig in figs_metrics]
            if self.state.do_stat_test:
                story.append(
                    Paragraph(
                        f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                        self.style["table_title"],
                    )
                )
                story.append(df2table(stat_test_groups[group_name]))
            story.append(PageBreak())
            subsubsection_counter += 1
        self.section_counter += 1
        return story

    def create_report_feat_importance(self) -> list:
        """Create report for analysis of feature importance per cohort.

        Returns
        -------
        list
            List of elements to append to the report
        """
        story = [
            Paragraph(
                f"{self.section_counter}. Feature importance Analysis", self.style["Heading1"]
            )
        ]
        subsection_counter = 1
        subsubsection_counter = 1
        story.append(
            Paragraph(
                f"Goal: Comparing the top {self.k_feat_importance} most important features across cohorts\n",
                self.style["Heading4"],
            )
        )
        if self.state.do_stat_test:
            rbo_table, to_color_rbo, critical_rbo_value = get_rbo_table(
                self.state.feat_ranking_all,
                self.state.feat_ranking_per_group,
                self.state.feat_ranking_random_group,
                self.k_feat_importance,
            )
            story.append(
                (
                    Paragraph(
                        f"{self.section_counter}.{subsection_counter}. Aggregated views",
                        self.style["Heading2"],
                    )
                )
            )
            story.append(
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}.{subsection_counter} Similarity of feature ranking per groupig",
                    self.style["sub_title"],
                )
            )
            counter_table = ord("a")
            story.append(
                Paragraph(
                    f"The following table displays the RBO (similarity measure) between the feature ranking for a patients' cohort and the general feature ranking. We consider the feature ranking for a specific cohort to be significantly different when its RBO is smaller than {round(critical_rbo_value, 3)} (colored in red in the table).\n"
                )
            )
            story.append(Paragraph("<br />\n <br />"))
            story.append(
                Paragraph(
                    f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                    self.style["table_title"],
                )
            )
            story.append(df2table_rbo(rbo_table, to_color_rbo))
            (
                topk_ranking_per_group,
                to_color_red_per_group,
                to_color_blue_per_group,
            ) = get_topk_ranking_critical_per_group(
                self.state.feat_ranking_all,
                self.state.feat_ranking_per_group,
                self.state.feat_ranking_random_group,
                self.k_feat_importance,
            )
            subsection_counter += 1
            subsubsection_counter = 1
        else:
            topk_ranking_per_group = get_topk_ranking_per_group(
                self.state.feat_ranking_all,
                self.state.feat_ranking_per_group,
                self.k_feat_importance,
            )
            to_color_red_per_group = None
            to_color_blue_per_group = None
        story.append(PageBreak())
        topk_ranking_all = self.state.feat_ranking_all[: self.k_feat_importance]
        story.append(
            (
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}. Grouping by",
                    self.style["Heading2"],
                )
            )
        )
        story.append(
            (
                Paragraph(
                    f"We will now display for each grouping, the top {self.k_feat_importance} most important features. When the feature's rank changes compared to the general ranking, we put the rank difference in parentheses.\n"
                )
            )
        )
        if self.state.do_stat_test:
            story.append(
                (
                    Paragraph(
                        f"We color in red the features that aren't in the general top {self.k_feat_importance} features and in blue the ones that change place within the top {self.k_feat_importance}, when their delta of inverse rank is significantly large.\n"
                    )
                )
            )
        for group_name, topk_ranking_group in topk_ranking_per_group.items():
            story.append(
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. ... {group_name}",
                    self.style["sub_title"],
                )
            )
            counter_table = ord("a")
            to_color_red_group = (
                to_color_red_per_group[group_name] if to_color_red_per_group is not None else None
            )
            to_color_blue_group = (
                to_color_blue_per_group[group_name] if to_color_blue_per_group is not None else None
            )
            story.append(
                Paragraph(
                    f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                    self.style["table_title"],
                )
            )
            story += generate_table_topk_feat(
                self.k_feat_importance,
                topk_ranking_all,
                topk_ranking_group,
                to_color_red_group,
                to_color_blue_group,
                self.max_cols_topkfeat_table,
            )
            subsubsection_counter += 1
        story.append(PageBreak())
        self.section_counter += 1
        return story

    def create_report_missingness(self) -> list:
        """Create report for missingness analysis per cohort.

        Returns
        -------
        list
            List of elements to append to the report
        """
        self.state.missingness_performance_df = self.state.missingness_performance_df.dropna(
            axis=1, how="all"
        )  # remove metrics that are only NaNs
        self.list_metrics_missingness = self.state.missingness_performance_df.columns[:-3]
        metrics_xlim = {
            metric: (
                round_down(min(self.state.missingness_performance_df[metric])),
                round_up(max(self.state.missingness_performance_df[metric])),
            )
            for metric in self.list_metrics_missingness
        }
        perf_stat_test_vars = {var: None for var in self.state.missingness_med_vars.keys()}
        intensity_stat_test_vars = {var: {} for var in self.state.missingness_med_vars.keys()}
        self.state.intensity_msrt_df = self.state.intensity_msrt_df.join(self.state.patients_df)
        story = [Paragraph(f"{self.section_counter}. Missingness Analysis", self.style["Heading1"])]
        subsection_counter = 1
        subsubsection_counter = 1
        story.append(
            Paragraph(
                "Goal: Comparing the intensity of measurements across cohorts of patients and its impact of performance\n",
                self.style["Heading4"],
            )
        )
        story.append(
            Paragraph(
                f"Binary metrics computed with a threshold on score of {round(self.state.threshold, 3)}.\n"
            )
        )
        if self.state.do_stat_test:
            story.append(
                (
                    Paragraph(
                        f"{self.section_counter}.{subsection_counter}. Aggregated views",
                        self.style["Heading2"],
                    )
                )
            )
            for var, (
                intensity_msrt_cats,
                performance_cats,
            ) in self.state.missingness_med_vars.items():
                story.append(
                    Paragraph(
                        f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. {var}",
                        self.style["sub_title"],
                    )
                )
                counter_table = ord("a")
                for group_name, cats in self.state.groups.items():
                    intensity_stat_test_vars[var][group_name] = get_stat_test_intensity_msrt(
                        self.state.intensity_msrt_df,
                        var,
                        group_name,
                        cats,
                        self.state.significance_level,
                    )
                story.append(
                    Paragraph(
                        "Groupings that are statistically dependent on the intensity of measurements:",
                        self.style["agg_title"],
                    )
                )
                summary_intensity_msrt = summarize_intensity_msrt(
                    self.state.intensity_msrt_df,
                    var,
                    intensity_stat_test_vars[var],
                    list(intensity_msrt_cats.keys())[1],
                )
                story.append(
                    Paragraph(
                        f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                        self.style["table_title"],
                    )
                )
                story.append(df2table(summary_intensity_msrt))
                metrics_df = self.state.missingness_performance_df[
                    self.state.missingness_performance_df["variable"] == var
                ]
                df_stat_test = generate_table_perf_missingness(
                    metrics_df,
                    self.list_metrics_missingness,
                    performance_cats[:-1],
                    self.state.significance_level,
                    self.state.filter_delta,
                )
                perf_stat_test_vars[var] = df_stat_test
                summaries_stat_test = summarize_stat_test_perf(
                    df_stat_test, performance_cats[:-1], len(self.list_metrics_missingness)
                )
                story.append(
                    Paragraph(
                        "Summary of the impact of missingness on performance.",
                        self.style["agg_title"],
                    ),
                )
                for perf_cat in performance_cats[:-1]:
                    story.append(Paragraph(summaries_stat_test[perf_cat]))
                subsubsection_counter += 1
            subsection_counter += 1
            subsubsection_counter = 1
        story.append(PageBreak())

        if self.state.do_stat_test:
            story.append(
                Paragraph(
                    f"For each grouping, we display a bar plot that shows the percentage of each intensity of measurement category within a cohort of patients. The dashed lines represent the percentage of each intensity of measurement category with respect to the entire patient population. We run the Chi-squared independence test (with significance level {self.state.significance_level}) to assess the depence between the intensity of measurement and the grouping."
                )
            )
            story.append(
                Paragraph(
                    "In the impact on performance subsection, we present box plots that show the metrics' distribution for each of the missingness categories. For each metric, we mark with a black star the missingness categories that are significantly worse compared to metrics computed on data points with present measurement."
                )
            )
            story.append(
                Paragraph(
                    f"We also propose tables presenting the results of the impact on performance statistical analysis, we display only metrics and missingness categories with a significant p-value (smaller than {self.state.significance_level}/number of comparisons) and whose delta is bigger than {self.state.filter_delta}. We compare the metrics for missingness categories <i>missing_msrt</i> and <i>no_msrt</i> (when relevant) against the <i>with_msrt</i> category. P-values are obtained by running the Mann-Whitney U test with Bonferroni correction.\n"
                )
            )
        else:
            story.append(
                Paragraph(
                    "For each grouping, we display a bar plot that shows the percentage of each intensity of measurement category within a cohort of patients. The dashed lines represent the percentage of each intensity of measurement category with respect to the entire patient population."
                )
            )
            story.append(
                Paragraph(
                    "In the impact on performance subsection, we present box plots that show the metrics' distribution for each of the missingness category."
                )
            )

        for var, (
            intensity_msrt_cats,
            performance_cats,
        ) in self.state.missingness_med_vars.items():
            story.append(
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}. Study of the variable {var}",
                    self.style["Heading2"],
                )
            )
            subsubsection_counter = 1
            # generate graphs for intensity of msrt analysis
            story.append(
                (
                    Paragraph(
                        f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. Intensity of measurement per grouping",
                        self.style["sub_title"],
                    )
                )
            )
            counter_figure = ord("a")
            for group_name, cats in self.state.groups.items():
                story.append(Paragraph(f"Grouping by {group_name}", self.style["Heading3"]))
                fig = draw_intensity_bar_plot(
                    var,
                    group_name,
                    cats,
                    self.state.intensity_msrt_df,
                    list(intensity_msrt_cats.keys()),
                    colors=self.colors,
                    figsize=self.figsize_barplot_intensity,
                )
                story.append(
                    Paragraph(
                        f"<u>Figure {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_figure)}</u>",
                        self.style["table_title"],
                    )
                )
                story.append(fig2image(fig))
                counter_figure += 1
                if self.state.do_stat_test:
                    story.append(Paragraph(intensity_stat_test_vars[var][group_name][1]))
            subsubsection_counter += 1
            story.append(
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}.{subsubsection_counter}. Impact on performance",
                    self.style["sub_title"],
                )
            )
            counter_figure = ord("a")
            counter_table = ord("a")
            metrics_df = self.state.missingness_performance_df[
                self.state.missingness_performance_df["variable"] == var
            ]
            figs_metrics = plot_boxplot_missingness_metrics_pw(
                metrics_df,
                performance_cats,
                self.list_metrics_missingness,
                figsize=self.figsize_boxplot_metrics,
                do_stat_test=self.state.do_stat_test,
                df_stat_test=perf_stat_test_vars[var],
                metrics_xlim=metrics_xlim,
                color_palette=self.colors,
            )
            story.append(
                Paragraph(
                    f"<u>Figure {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_figure)}</u>",
                    self.style["table_title"],
                )
            )
            story += [fig2image(fig) for fig in figs_metrics]
            if self.state.do_stat_test and self.display_stat_test_table:
                story.append(
                    Paragraph(
                        f"<u>Table {self.section_counter}.{subsection_counter}.{subsubsection_counter}.{chr(counter_table)}</u>",
                        self.style["table_title"],
                    )
                )
                story.append(df2table(perf_stat_test_vars[var]))
            story.append(PageBreak())
            subsection_counter += 1
        self.section_counter += 1
        return story

    def create_glossary(self) -> list:
        story = [Paragraph(f"{self.section_counter}. Glossary", self.style["Heading1"])]
        subsection_counter = 1
        story.append(
            (
                Paragraph(
                    f"{self.section_counter}.{subsection_counter}. General concepts",
                    self.style["Heading2"],
                )
            )
        )
        story += generate_glossary_general()
        subsection_counter += 1
        if hasattr(self.state, "metrics_group_df") and self.state.metrics_group_df is not None:
            # add description here
            story.append(
                (
                    Paragraph(
                        f"{self.section_counter}.{subsection_counter}. Model Performance Analysis concepts",
                        self.style["Heading2"],
                    )
                )
            )
            story += generate_glossary_model_performance(self.list_metrics_performance)
            subsection_counter += 1
        if hasattr(self.state, "timegap_group_df") and self.state.timegap_group_df is not None:
            story.append(
                (
                    Paragraph(
                        f"{self.section_counter}.{subsection_counter}. Time Gap Analysis concepts",
                        self.style["Heading2"],
                    )
                )
            )
            story += generate_glossary_timegap()
            subsection_counter += 1
        if hasattr(self.state, "medvars_group_df") and self.state.medvars_group_df is not None:
            story.append(
                (
                    Paragraph(
                        f"{self.section_counter}.{subsection_counter}. Medical Variable Analysis concepts",
                        self.style["Heading2"],
                    )
                )
            )
            story += generate_glossary_medvars()
            subsection_counter += 1
        if hasattr(self.state, "feat_ranking_all") and self.state.feat_ranking_all is not None:
            story.append(
                (
                    Paragraph(
                        f"{self.section_counter}.{subsection_counter}. Feature Importance Analysis concepts",
                        self.style["Heading2"],
                    )
                )
            )
            story += generate_glossary_feature_importance(self.k_feat_importance)
            subsection_counter += 1
        if (
            hasattr(self.state, "missingness_performance_df")
            and self.state.missingness_performance_df is not None
        ):
            story.append(
                (
                    Paragraph(
                        f"{self.section_counter}.{subsection_counter}. Missingness Analysis concepts",
                        self.style["Heading2"],
                    )
                )
            )
            intensity_msrt_cats = list(self.state.missingness_med_vars.values())[0][0]
            story += generate_glossary_missingness(
                intensity_msrt_cats, self.list_metrics_missingness, self.list_metrics_performance
            )
            subsection_counter += 1
        self.section_counter += 1
        return story
