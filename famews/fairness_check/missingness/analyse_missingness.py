import logging
from pathlib import Path
from typing import Dict, List, Tuple

import gin
import numpy as np
import pandas as pd

from famews.data.utils import MarkedDataset
from famews.fairness_check.missingness.helper_analyse_missingness import (
    compute_metrics_timepoint,
    compute_prevalence_timepoint_categories,
    get_predictions_msrt_ind,
    get_predictions_no_msrt,
    map_intensity_rate_category,
)
from famews.fairness_check.utils.helper_bootstrap import draw_bootstrap_sample
from famews.pipeline import PipelineState, StatefulPipelineStage


@gin.configurable("AnalyseMissingnessGroup", denylist=["state"])
class AnalyseMissingnessGroup(StatefulPipelineStage):
    """
    Analyse Missingness per Group stage
    """

    name = "Analyse Missingness Group"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        use_cache: int = True,
        overwrite: bool = False,
        path_common_stage_dir: Path = gin.REQUIRED,
        patients_source: str = "hirid",
        medical_vars: Dict[str, int] = gin.REQUIRED,
        vars_only_patient_msrt: List[str] = [],
        category_intensity: Dict[str, Tuple[float, float]] = {
            "insufficient": (0, 0.9),
            "enough": (0.9, 1),
        },
        metrics_binary_to_check: List[str] = [
            "recall",
            "precision",
            "npv",
            "fpr",
            "corrected_precision",
            "corrected_npv",
        ],
        metrics_score_to_check: List[str] = [
            "positive_class_score",
            "negative_class_score",
            "auroc",
            "auprc",
            "corrected_auprc",
        ],
        **kwargs,
    ):
        """Assess the dependence between intensity of measurements of selected variables and cohorts of patients, then measure its impact
        on model performance.

        Parameters
        ----------
        state : PipelineState
            Pipeline state
        num_workers : int, optional
            Number of workers, by default 1
        use_cache : int, optional
            flag to cache analysis results and use previously cached results, by default True
        overwrite : bool, optional
            flag to overwrite analysis results that were previously cached, by default False
        path_common_stage_dir : Path, optional
            Path to directory where common_stage preprocessed data is stored, by default gin.REQUIRED
        patients_source : str, optional
            Data source, by default "hirid"
        medical_vars : Dict[str, int], optional
            Dictionary of medical variables to study with their expected sampling intervals in minutes, by default gin.REQUIRED
        vars_only_patient_msrt : List[str], optional
            List of medical variables for which we want only to study predictions on patients with measurement (subset of medical_vars keys), by default []
        category_intensity : Dict[str, Tuple[float, float]], optional
            Dictionary defining the categories of intensity of measurement {intensity_category: (lower, upper]}, categories have to be
            in increasing order, by default { "insufficient": (0, 0.9), "enough": (0.9, 1), }
        metrics_binary_to_check : List[str], optional
            List of binary metrics to measure (no event-based metrics should be present), by default [ "recall", "precision", "npv", "fpr", "corrected_precision",  "corrected_npv"]
        metrics_score_to_check : List[str], optional
            List of score-based metrics to measure (no event-based metrics should be present), by default [ "positive_class_score", "negative_class_score", "auroc", "auprc", "corrected_auprc", ]
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.use_cache = use_cache
        self.overwrite = overwrite
        self.do_bootstrap = self.state.do_stat_test
        self.path_common_stage_dir = Path(path_common_stage_dir)
        self.patients_source = patients_source
        self.medical_vars = medical_vars
        self.vars_only_patient_msrt = vars_only_patient_msrt
        self.category_intensity = category_intensity
        self.name_suffix = f"_at_{self.state.name_threshold}"
        self.metrics_binary_to_check = metrics_binary_to_check
        self.metrics_score_to_check = metrics_score_to_check
        self.cache_path = Path(self.state.fairness_log_dir)
        self.missingness_perf_path = self.cache_path / "missingness_perf_metrics.csv"
        self.intensity_msrt_path = self.cache_path / "intensity_msrts.csv"

    def runnable(self) -> bool:
        """
        Check whether the state contains predictions, patients dataframe and grouping definition
        """
        return (
            hasattr(self.state, "predictions")
            & hasattr(self.state, "patients_df")
            & hasattr(self.state, "groups")
        )

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains intensity of measurements dataframe, performance metrics for missingness, studied medical variables
         and in case of caching on disk check if the file has been cached.

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        # check if self.state contains predictions and in case of caching on disk check if the file has been cached
        if self.use_cache:
            return (
                hasattr(self.state, "missingness_performance_df")
                and self.state.missingness_performance_df is not None
                and self.missingness_perf_path.exists()
                and hasattr(self.state, "intensity_msrt_df")
                and self.state.intensity_msrt_df is not None
                and self.intensity_msrt_path.exists()
                and hasattr(self.state, "missingness_med_vars")
                and self.state.missingness_med_vars is not None
            )
        return (
            hasattr(self.state, "missingness_performance_df")
            and self.state.missingness_performance_df is not None
            and hasattr(self.state, "intensity_msrt_df")
            and self.state.intensity_msrt_df is not None
            and hasattr(self.state, "missingness_med_vars")
            and self.state.missingness_med_vars is not None
        )

    def run(self):
        """
        Run the Missingness analysis.
        """
        self.state.missingness_med_vars = {}
        for var in self.medical_vars.keys():
            if var not in self.vars_only_patient_msrt:
                self.state.missingness_med_vars[var] = (
                    {**{"no_msrt": (0, 0)}, **self.category_intensity},
                    ["no_msrt", "missing_msrt", "with_msrt"],
                )
            else:
                self.state.missingness_med_vars[var] = (
                    {**{"no_msrt": (0, 0)}, **self.category_intensity},
                    ["missing_msrt", "with_msrt"],
                )
        if (
            not self.overwrite
            and self.missingness_perf_path.exists()
            and self.intensity_msrt_path.exists()
        ):
            # load previously cached metrics and curves
            self.state.missingness_performance_df = pd.read_csv(self.missingness_perf_path)
            self.state.intensity_msrt_df = pd.read_csv(self.intensity_msrt_path).set_index("pid")
            return

        self.test_pids = list(self.state.pid_split["test"])
        self.get_missingness_info_patients()
        self.performance_res = []
        if self.do_bootstrap:
            n_samples = 100
            bootstrap_sample_pids = draw_bootstrap_sample(
                list(self.state.predictions.keys()), sample_size=1, n_samples=n_samples
            )  # do we want to parametrise this?
        else:
            n_samples = 1
            bootstrap_sample_pids = [list(self.state.predictions.keys())]
        for it, bootstrap_pids in enumerate(bootstrap_sample_pids):
            # patients = self.state.patients_df.loc[bootstrap_pids]
            for var in self.medical_vars:
                logging.info(f"[{self.__class__.__name__}] [Performance metrics] Variable: {var}")
                self.get_performance_var(bootstrap_pids, var, it)
                # compute metrics
        self.state.missingness_performance_df = pd.DataFrame(self.performance_res)
        if self.use_cache:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.state.missingness_performance_df.to_csv(self.missingness_perf_path, index=False)
            self.state.intensity_msrt_df.to_csv(self.intensity_msrt_path)

    def get_performance_var(self, list_pids: List[int], var: str, it: int):
        """Compute the model performance on different missingness categories on
        the predictions made on a subset of patients.

        Parameters
        ----------
        list_pids : List[int]
            List of patient ids
        var : str
            Variable name
        it : int
            Bootstrap iteration
        """
        map_cat_labels_array = {}
        map_cat_preds_array = {}
        if var not in self.vars_only_patient_msrt:
            preds_no_msrt, labels_no_msrt = get_predictions_no_msrt(
                self.state.predictions, list_pids, self.dict_var_pid_no_msrt[var]
            )
            map_cat_labels_array["no_msrt"] = labels_no_msrt
            map_cat_preds_array["no_msrt"] = preds_no_msrt
        (
            preds_w_msrt,
            labels_w_msrt,
            preds_missing_msrt,
            labels_missing_msrt,
        ) = get_predictions_msrt_ind(self.state.predictions, list_pids, self.dict_cat_msrt_ind[var])
        map_cat_labels_array["with_msrt"] = labels_w_msrt
        map_cat_preds_array["with_msrt"] = preds_w_msrt
        map_cat_labels_array["missing_msrt"] = labels_missing_msrt
        map_cat_preds_array["missing_msrt"] = preds_missing_msrt
        (
            dict_prevalence,
            prev_baseline_precision,
            prev_baseline_npv,
        ) = compute_prevalence_timepoint_categories(map_cat_labels_array)
        for cat in map_cat_labels_array.keys():
            metrics = compute_metrics_timepoint(
                map_cat_labels_array[cat],
                map_cat_preds_array[cat],
                self.state.threshold,
                prev_baseline_precision,
                prev_baseline_npv,
                dict_prevalence[cat],
                self.name_suffix,
                self.metrics_binary_to_check,
                self.metrics_score_to_check,
            )
            metrics["variable"] = var
            metrics["cat"] = cat
            metrics["run"] = it
            self.performance_res.append(metrics)

    def get_missingness_info_patients(self):
        """Extract missingness information from common stage data.
        The following process in run for every selected medical variable:
            - for each patient with at least a measurement, determine the intensity of measurement
            and save indices of timepoints without valid measurement.
            - for patient without measurement, save its pid

        Raises
        ------
        ValueError
            Unsupported patients source
        """
        if self.patients_source == "hirid":
            from famews.data.hirid.constants import DATETIME, PART_PARQUET_PATTERN, PID
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported patients source: {self.patients_source}"
            )

        common_stage_parquet_file = next(self.path_common_stage_dir.rglob("*.parquet"))
        common_stage_directory = common_stage_parquet_file.parent
        common_stage_dataset = MarkedDataset(
            common_stage_directory, part_re=PART_PARQUET_PATTERN, force=False
        )
        parts_list = common_stage_dataset.list_parts()
        keep_columns = [PID, DATETIME] + list(self.medical_vars.keys())
        self.dict_var_pid_no_msrt = {var: [] for var in self.medical_vars.keys()}
        self.dict_intensity_msrt = {var: {} for var in self.medical_vars.keys()}
        self.dict_cat_msrt_ind = {var: {} for var in self.medical_vars.keys()}
        for part in parts_list:
            df_vars = pd.read_parquet(part, columns=keep_columns)
            grouped_df_vars = df_vars.groupby(PID)
            for pid, df in grouped_df_vars:
                df = df.sort_values(by=DATETIME)[: self.state.max_len]
                if not len(df):
                    continue
                if pid in self.test_pids:
                    for var, baseline_time in self.medical_vars.items():
                        los = df[DATETIME].max() * self.state.timestep
                        df_var = df[[DATETIME, var]].dropna()
                        if not len(df_var):
                            self.dict_var_pid_no_msrt[var].append(pid)
                            self.dict_intensity_msrt[var][pid] = "no_msrt"
                        else:
                            rate_msrt = 1 - self.get_rate_missing_var(
                                list(df_var[DATETIME]), los, baseline_time
                            )
                            self.dict_intensity_msrt[var][pid] = map_intensity_rate_category(
                                rate_msrt, self.category_intensity
                            )
                            isna_ = np.array(df[var].isna())
                            self.dict_cat_msrt_ind[var][pid] = self.get_indices_missing_var(
                                list(df_var[DATETIME]), isna_, baseline_time
                            )
        self.state.intensity_msrt_df = pd.DataFrame(self.dict_intensity_msrt)
        self.state.intensity_msrt_df.index.name = "pid"

    def get_rate_missing_var(self, list_datetime: list, los: int, baseline_time: int) -> float:
        """Compute the rate of missing measurement for a patient and a variable.
        Based on the list of datetimes of actual measurement and the baseline time, we are able to
        compute the number of missed measurement. The number of expected measurement is obtained by
        dividing the length of stay with the baseline time.

        Parameters
        ----------
        list_datetime : list
            List of datetime of measurement
        los : int
            Patient's length of stay
        baseline_time : int
            Expected interval of measurement (in minutes)

        Returns
        -------
        float
            Rate of missing measurement: nb_missed_msrt / nb_exp_msrt.
        """
        nb_missed_msrt = 0
        nb_exp_msrt = np.ceil(los / baseline_time)
        for i in range(0, len(list_datetime) - 1):
            interval_no_msrt = (list_datetime[i + 1] - list_datetime[i]) * self.state.timestep
            if interval_no_msrt > baseline_time:
                nb_missed_msrt += interval_no_msrt // baseline_time
        return nb_missed_msrt / nb_exp_msrt

    def get_indices_missing_var(
        self, list_datetime: list, isna_: np.array, baseline_time: int
    ) -> Dict[str, List[int]]:
        """Get the indices in the time series with measurement and with missing measurement for a patient and a variable.
        We propagate measurement forward based on the baseline time.

        Parameters
        ----------
        list_datetime : list
            List of datetime of measurement
        isna_ : np.array
            Array of boolean stating for each timestep whether a measuement has been done
        baseline_time : int
            Expected interval of measurement (in minutes)

        Returns
        -------
        Dict[str, List[int]]
            Map missingness category to a list of indices
        """
        ind_w_msrt = []
        ind_missing_msrt = []
        for dt_msrt in list_datetime:
            isna_[
                int(dt_msrt // self.state.timestep) : min(
                    len(isna_), int((dt_msrt + baseline_time) // self.state.timestep)
                )
            ] = False
        for ind, is_missing in enumerate(isna_):
            if is_missing:
                ind_missing_msrt.append(ind)
            else:
                ind_w_msrt.append(ind)
        return {"with_msrt": ind_w_msrt, "missing_msrt": ind_missing_msrt}
