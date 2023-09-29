import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import gin
import pandas as pd

from famews.data.utils import MarkedDataset
from famews.fairness_check.utils.helper_bootstrap import draw_bootstrap_sample
from famews.fairness_check.utils.helper_groups import (
    get_map_group_pid,
    get_map_not_group_pid,
)
from famews.pipeline import PipelineState, StatefulPipelineStage

GET_LAMBDA = lambda expr: lambda **kwargs: bool(eval(expr, kwargs))


@gin.configurable("AnalyseMedVarsGroup", denylist=["state"])
class AnalyseMedVarsGroup(StatefulPipelineStage):
    """
    Analyse Medical Variables per Group stage
    """

    name = "Analyse Medical Variables Group"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        use_cache: int = True,
        overwrite: bool = False,
        path_common_stage_dir: Path = gin.REQUIRED,
        patients_source: str = "hirid",
        classical_vars: List[str] = gin.REQUIRED,
        conditioned_vars: Dict[str, Tuple[str, bool]] = None,
        specified_conditions: Dict[str, Dict[str, Tuple[str, str]]] = None,
        vars_units: Dict[str, str] = {},
        **kwargs,
    ):
        """Compute the median values of specified medical variables for each cohort of patients, based on the training data.
        For each cohort, the median value is computed for 3 conditions: all patients across the entire stay, for all patients but only while not in event, and
        only for patients not experiencing any event.

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
        classical_vars : List[str], optional
            List of medical variables (without conditioning), by default gin.REQUIRED
        conditioned_vars : Dict[str, Tuple[str, bool]], optional
            Dictionary of medical variables to study when a specific condition is met {var: (condition_name, condition_boolean)}, by default None
        specified_conditions : Dict[str, Dict[str, Tuple[str, str]]], optional
            Dictionary defining the conditions {condition_name: {var: (condition_on, condition_off)}}, by default None
        vars_units: Dict[str, str], optional
            Dictionary storing the units of the medical variables (only for display purpose), by default {}
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.use_cache = use_cache
        self.overwrite = overwrite
        self.do_bootstrap = self.state.do_stat_test
        self.path_common_stage_dir = Path(path_common_stage_dir)
        self.patients_source = patients_source
        self.classical_vars = classical_vars
        self.conditioned_vars = conditioned_vars
        self.specified_conditions = specified_conditions
        self.state.medvars_units = vars_units
        self.cache_path = Path(self.state.fairness_log_dir)
        self.medvars_path = self.cache_path / "med_vars_group.csv"

    def runnable(self) -> bool:
        """
        Check whether the state contains pid_split, patients dataframe, event bounds and grouping definition
        """
        return (
            hasattr(self.state, "pid_split")
            & hasattr(self.state, "patients_df")
            & hasattr(self.state, "event_bounds")
            & hasattr(self.state, "groups")
        )

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains medical vars info per cohort and in case of caching on disk check if the file has been cached

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        return (
            hasattr(self.state, "medvars_group_df")
            and self.state.med_vars_group_df is not None
            and hasattr(self.state, "list_medvars")
            and self.state.list_medvars is not None
            and (self.med_vars_path.exists() or not self.use_cache)
        )

    def run(self):
        """
        Run the Med Vars analysis.
        """
        self.state.list_medvars = self.classical_vars + list(self.conditioned_vars.keys())
        if not self.overwrite and self.medvars_path.exists():
            # load previously cached results
            self.state.medvars_group_df = pd.read_csv(self.medvars_path)
            return
        self.train_pids = list(self.state.pid_split["train"])
        if self.do_bootstrap:
            bootstrap_sample_pids = draw_bootstrap_sample(
                self.train_pids, sample_size=1, n_samples=100
            )
        else:
            bootstrap_sample_pids = [self.train_pids]
        self.medvars_patients = self.get_medvars_patients()
        n_analysed_samples = 0
        self.state.medvars_group_df = pd.DataFrame()
        for bootstrap_pids in bootstrap_sample_pids:
            patients = self.state.patients_df.loc[bootstrap_pids]
            for group_name, cats in self.state.groups.items():
                logging.info(f"[{self.__class__.__name__}] Grouping: {group_name}")
                map_group_pid = get_map_group_pid(patients, group_name, cats)
                self.agg_medvars_group(map_group_pid, group_name, cats, n_analysed_samples)
                if (
                    self.state.do_stat_test
                    and group_name in self.state.type_table_groups["multicat_group"]
                ):
                    map_group_pid = get_map_not_group_pid(patients, group_name, cats)
                    group_name = f"{group_name}_bool"
                    self.agg_medvars_group(
                        map_group_pid,
                        group_name,
                        self.state.table_groups[group_name],
                        n_analysed_samples,
                    )
            n_analysed_samples += 1
        self.state.medvars_group_df[
            [
                var + suffix
                for var in self.state.list_medvars
                for suffix in ["", "_not_inevent", "_never_inevent"]
            ]
        ] = self.state.medvars_group_df[
            [
                var + suffix
                for var in self.state.list_medvars
                for suffix in ["", "_not_inevent", "_never_inevent"]
            ]
        ].astype(
            float
        )
        if self.use_cache:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.state.medvars_group_df.to_csv(self.medvars_path, index=False)

    def agg_medvars_group(
        self, map_group_pid: Dict[str, List[int]], group_name: str, cats: List[str], it: int
    ):
        """Compute the median time gap per category of patients.

        Parameters
        ----------
        map_group_pid : Dict[str, List[int]]
            Dictionary that maps category of patients to the corresponding list of patient ids
        group_name : str
            Group name
        cats : List[str]
            List of categories
        it : int
            Iteration number (useful while doing bootstrapping)
        """
        for cat in cats:
            medvars_cat = self.medvars_patients.loc[map_group_pid[cat]].median()
            medvars_cat["group"] = group_name
            medvars_cat["cat"] = cat
            medvars_cat["run"] = it
            self.state.medvars_group_df = pd.concat(
                (self.state.medvars_group_df, pd.DataFrame(medvars_cat).T), ignore_index=True
            )

    def get_medvars_patients(self) -> pd.DataFrame:
        """Iterate through the common_stage data to compute the median value of some medical variables for all patients.

        Returns
        -------
        pd.DataFrame
            store for each patient the median value of the medical variables

        Raises
        ------
        ValueError
            Unsupported source of data
        """
        if self.patients_source == "hirid":
            from famews.data.hirid.constants import DATETIME, PART_PARQUET_PATTERN, PID
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported patients sources: {self.patients_source}"
            )

        common_stage_parquet_file = next(self.path_common_stage_dir.rglob("*.parquet"))
        common_stage_directory = common_stage_parquet_file.parent
        common_stage_dataset = MarkedDataset(
            common_stage_directory, part_re=PART_PARQUET_PATTERN, force=False
        )
        parts_list = common_stage_dataset.list_parts()
        keep_columns = [PID, DATETIME] + self.state.list_medvars
        for cond_var in self.specified_conditions.values():
            keep_columns += list(cond_var.keys())
        res = {}
        for part in parts_list:
            df_medvars = pd.read_parquet(part, columns=keep_columns)
            grouped_df_medvars = df_medvars.groupby(PID)
            for pid, df in grouped_df_medvars:
                if pid in self.train_pids:
                    res[pid] = self.agg_medvars(pid, df.sort_values(by=DATETIME))
        return pd.DataFrame(res).T

    def agg_medvars(self, pid: int, df: pd.DataFrame) -> Dict[str, float]:
        """Compute the median of some medical variables for a specific patient.

        Parameters
        ----------
        pid : int
            Patient id
        df : pd.DataFrame
            dataframe with common stage data

        Returns
        -------
        Dict[str, float]
            Mapping a medical variable to its median value
        """
        res = {}
        df["in_event"] = False
        if pid in self.state.event_bounds:
            event_bounds = self.state.event_bounds[pid]
        else:
            event_bounds = []
        if len(event_bounds):
            for (start, stop) in event_bounds:
                df.loc[start:stop, "in_event"] = True
        for var in self.classical_vars:
            res[var] = df[var].dropna().median()
            res[f"{var}_not_inevent"] = df[~df["in_event"]][var].dropna().median()
            if not len(event_bounds):
                res[f"{var}_never_inevent"] = res[var]
        for var, (cond_name, cond_value) in self.conditioned_vars.items():
            df = self.build_condition(cond_name, df)
            df_cond = df[df[cond_name] == cond_value]
            res[var] = df_cond[var].dropna().median()
            res[f"{var}_not_inevent"] = df_cond[~df_cond["in_event"]][var].dropna().median()
            if not len(event_bounds):
                res[f"{var}_never_inevent"] = res[var]
        return res

    def build_condition(self, condition_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Check for which time points a condition is met. The condition is defined in specified_conditions as {var: (condition_on, condition_off)}.
        The condition is met when var satisfies condition_on.

        Parameters
        ----------
        condition_name : str
            Condition name
        df : pd.DataFrame
            Dataframe with common stage data

        Returns
        -------
        pd.DataFrame
            Dataframe with an extra column containing whether the condition is met or not for the different time points
        """
        start = None
        df[condition_name] = False
        for var, (str_check_on, str_check_off) in self.specified_conditions[condition_name].items():
            check_on = GET_LAMBDA(str_check_on)
            check_off = GET_LAMBDA(str_check_off)
            for j in df[var].dropna().index:
                val = df.loc[j, var]
                if check_on(val=val) and start is None:
                    start = j
                if check_off(val=val) and start is not None:
                    df.loc[start:j, condition_name] = True
        return df
