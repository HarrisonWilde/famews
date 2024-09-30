import logging
import pickle
from pathlib import Path
from typing import Dict, List

import gin
import pandas as pd
from famews.fairness_check.model_performance.helper_analyse_performance import (
    compute_metrics,
    compute_prevalence_groups,
    map_predictions_group,
)
from famews.fairness_check.utils.get_pop_info import check_group_has_event
from famews.fairness_check.utils.helper_bootstrap import draw_bootstrap_sample
from famews.fairness_check.utils.helper_groups import (
    get_map_group_pid,
    get_map_not_group_pid,
)
from famews.pipeline import PipelineState, StatefulPipelineStage


@gin.configurable("AnalysePerformanceGroup", denylist=["state"])
class AnalysePerformanceGroup(StatefulPipelineStage):
    """
    Analyse Performance per Group Pipeline stage
    """

    name = "Analyse Performance Group"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        use_cache: int = True,
        overwrite: bool = False,
        store_curve: bool = False,
        nbins_calibration: int = 5,
        metrics_binary_to_check: List[str] = [
            "recall",
            "precision",
            "npv",
            "fpr",
            "corrected_precision",
            "corrected_npv",
            "event_recall",
        ],
        metrics_score_to_check: List[str] = [
            "positive_class_score",
            "negative_class_score",
            "auroc",
            "auprc",
            "corrected_auprc",
            "event_auprc",
            "corrected_event_auprc",
            "calibration_error",
        ],
        **kwargs,
    ):
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.metrics_binary_to_check = metrics_binary_to_check
        self.metrics_score_to_check = metrics_score_to_check
        self.do_bootstrap = self.state.do_stat_test
        self.store_curve = store_curve
        self.nbins_calibration = nbins_calibration
        self.name_suffix = f"_at_{self.state.name_threshold}"
        self.use_cache = use_cache
        self.overwrite = overwrite
        self.cache_path = Path(self.state.fairness_log_dir)
        self.metrics_path = self.cache_path / "metrics_group.csv"
        self.curves_path = self.cache_path / "curves_group.pkl"

    def runnable(self) -> bool:
        """
        Check whether the state contains predictions, patients dataframe and grouping definition
        """
        return (
            hasattr(self.state, "predictions")
            & hasattr(self.state, "patients_df")
            & hasattr(self.state, "event_bounds")
            & hasattr(self.state, "groups")
        )

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains cohort metrics and in case of caching on disk check if the file has been cached

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        # check if self.state contains predictions and in case of caching on disk check if the file has been cached
        if self.use_cache:
            return (
                hasattr(self.state, "metrics_group_df")
                and self.state.metrics_group_df is not None
                and self.metrics_path.exists()
                and (
                    not self.store_curve
                    or (
                        hasattr(self.state, "curves_group")
                        and self.state.curves_group is not None
                        and self.curves_path.exists()
                    )
                )
            )
        return (
            hasattr(self.state, "metrics_group_df")
            and self.state.metrics_group_df is not None
            and (
                not self.store_curve
                or (hasattr(self.state, "curves_group") and self.state.curves_group)
            )
        )

    def run(self):
        """
        Compute the model performance metrics for each cohort.
        """
        if (
            not self.overwrite
            and self.metrics_path.exists()
            and (not self.store_curve or self.curves_path.exists())
        ):
            # load previously cached metrics and curves
            self.state.metrics_group_df = pd.read_csv(self.metrics_path)
            if self.store_curve:
                with open(self.curves_path, "rb") as f:
                    self.state.curves_group = pickle.load(f)
            return
        if self.do_bootstrap:
            n_samples = 100
            bootstrap_sample_pids = draw_bootstrap_sample(
                list(self.state.predictions.keys()),
                sample_size=1,
                n_samples=10 * n_samples,
            )  # do we want to parametrise this?
        else:
            n_samples = 1
            bootstrap_sample_pids = [list(self.state.predictions.keys())]
        self.analysis_res = []
        if self.store_curve:
            self.analysis_res_curve = {
                group_name: {cat: [] for cat in cats}
                for group_name, cats in self.state.groups.items()
            }
        else:
            self.analysis_res_curve = None
        n_analysed_samples = 0
        for bootstrap_pids in bootstrap_sample_pids:
            patients = self.state.patients_df.loc[bootstrap_pids]
            if self.do_bootstrap:
                if n_analysed_samples >= n_samples:
                    break
                if not check_group_has_event(
                    patients,
                    self.state.groups,
                    self.state.event_bounds,
                    self.state.max_len,
                    self.state.horizon,
                ):
                    continue
                else:
                    n_analysed_samples += 1
            for group_name, cats in self.state.groups.items():
                logging.info(f"[{self.__class__.__name__}] Group: {group_name}")
                map_group_pid = get_map_group_pid(patients, group_name, cats)
                self.get_performance_group(
                    map_group_pid, group_name, cats, n_analysed_samples, False
                )
                if (
                    self.state.do_stat_test
                    and group_name in self.state.type_table_groups["multicat_group"]
                ):
                    map_group_pid = get_map_not_group_pid(patients, group_name, cats)
                    group_name = f"{group_name}_bool"
                    self.get_performance_group(
                        map_group_pid,
                        group_name,
                        self.state.table_groups[group_name],
                        n_analysed_samples,
                        True,
                    )
        if self.do_bootstrap and n_analysed_samples < n_samples:
            logging.warning(
                f"[{self.__class__.__name__}] Analysis done only with {n_analysed_samples} (over {n_samples}). Think about increasing GetGroupSize.min_event_group parameter to discard cohorts with too few events."
            )
        self.state.metrics_group_df = pd.DataFrame(self.analysis_res)
        self.state.curves_group = self.analysis_res_curve
        if self.use_cache:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.state.metrics_group_df.to_csv(self.metrics_path, index=False)
            if self.store_curve:
                with open(self.curves_path, "wb") as f:
                    pickle.dump(self.analysis_res_curve, f)

    def get_performance_group(
        self,
        map_group_pid: Dict[str, List[int]],
        group_name: str,
        cats: List[str],
        it: int,
        run_stat_test: bool = False,
    ):
        """Get model performance for each category of a grouping.

        Parameters
        ----------
        map_group_pid : Dict[str, List[int]]
            Dictionary mapping each category to corresponding list of patient ids {cat: [pids]}
        group_name : str
            Group name
        cats : List[str]
            List of categories
        it : int
            Iteration number
        run_stat_test : bool, optional
            Whether this function is run only to perform the statistical test, by default False
        """
        if run_stat_test:
            store_curve = False
        else:
            store_curve = self.store_curve
        (
            labels_array_group,
            preds_array_group,
        ) = map_predictions_group(self.state.predictions, map_group_pid)
        (
            dict_prevalence,
            prev_baseline_precision,
            prev_baseline_npv,
        ) = compute_prevalence_groups(labels_array_group)
        for cat in cats:
            metrics, curves = compute_metrics(
                self.state.predictions,
                map_group_pid[cat],
                labels_array_group[cat],
                preds_array_group[cat],
                self.state.threshold,
                self.state.event_bounds,
                prev_baseline_precision,
                prev_baseline_npv,
                dict_prevalence[cat],
                self.state.horizon,
                self.state.max_len,
                self.state.timestep,
                self.name_suffix,
                store_curve,
                self.nbins_calibration,
                self.metrics_binary_to_check,
                self.metrics_score_to_check,
            )
            metrics["group"] = group_name
            metrics["cat"] = cat
            metrics["run"] = it
            self.analysis_res.append(metrics)
            if store_curve:
                self.analysis_res_curve[group_name][cat].append(curves)
