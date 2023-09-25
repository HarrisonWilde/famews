import logging
import pickle
from pathlib import Path
from typing import Dict, List

import gin
import numpy as np
import pandas as pd

from famews.fairness_check.utils.get_pop_info import check_group_has_event
from famews.fairness_check.utils.helper_bootstrap import draw_bootstrap_sample
from famews.fairness_check.utils.helper_groups import (
    get_map_group_pid,
    get_map_not_group_pid,
)
from famews.pipeline import PipelineState, StatefulPipelineStage


@gin.configurable("AnalyseTimeGapGroup", denylist=["state"])
class AnalyseTimeGapGroup(StatefulPipelineStage):
    """
    Analyse Time Gap per Group Pipeline stage
    """

    name = "Analyse Time Gap Group"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        use_cache: int = True,
        overwrite: bool = False,
        groups_start_event: List[int] = [0, 3, 6, 12],
        **kwargs,
    ):
        """Compare the median time gap between first correct alarm and corresponding event for each cohort.

        Parameters
        ----------
        state : PipelineState
            Pipeline state
        num_workers : int, optional
            number of workers, by default 1
        use_cache : int, optional
            flag to cache analysis results and use previously cached results, by default True
        overwrite : bool, optional
            flag to overwrite analysis results that were previously cached, by default False
        groups_start_event : List[int], optional
            List to define the bounds to group the events based on the duration between the last event and the start of the event,
            values in the list corresponds to the lower bound of the duration window in hours and should be in strictly increasing order (e.g. default value gives the following groups: [0-3h, 3-6h, 6-12h, >12h]), by default [0, 3, 6, 12]
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.use_cache = use_cache
        self.overwrite = overwrite
        self.do_bootstrap = self.state.do_stat_test
        self.groups_start_event = groups_start_event
        self.cache_path = Path(self.state.fairness_log_dir)
        self.timegap_path = self.cache_path / "timegap_group.csv"

    def runnable(self) -> bool:
        """
        Check whether the state contains predictions, patients dataframe, event bounds and grouping definition
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
        return (
            hasattr(self.state, "timegap_group_df")
            and self.state.timegap_group_df is not None
            and hasattr(self.state, "list_group_start_event")
            and self.state.list_group_start_event is not None
            and hasattr(self.state, "timegap_xlim")
            and self.state.timegap_xlim is not None
            and (self.timegap_path.exists() or not self.use_cache)
        )

    def run(self):
        """
        Run the Time Gap between alarm and event analysis.
        """
        # check that self.groups_start_event is in increasing order
        assert all(
            self.groups_start_event[i] < self.groups_start_event[i + 1]
            for i in range(len(self.groups_start_event) - 1)
        ), f"[{self.__class__.__name__}] values in group_start_event should be in increasing order without duplicate."
        self.state.list_group_start_event = self.get_list_group_start_event()
        self.state.timegap_xlim = {
            f"{self.groups_start_event[i]}-{self.groups_start_event[i+1]}h": (
                0,
                min(self.groups_start_event[i + 1], self.state.horizon) * 60,
            )
            for i in range(len(self.groups_start_event[:-1]))
        }
        self.state.timegap_xlim[f">{self.groups_start_event[-1]}h"] = (
            0,
            self.state.horizon * 60,
        )
        if not self.overwrite and self.timegap_path.exists():
            # load previously cached metrics and curves
            self.state.timegap_group_df = pd.read_csv(self.timegap_path)
            return
        if self.do_bootstrap:
            n_samples = 100
            bootstrap_sample_pids = draw_bootstrap_sample(
                list(self.state.predictions.keys()), sample_size=1, n_samples=10 * n_samples
            )
        else:
            n_samples = 1
            bootstrap_sample_pids = [list(self.state.predictions.keys())]
        self.timegap_patients = self.get_timegap_patients()
        n_analysed_samples = 0
        self.state.timegap_group_df = pd.DataFrame()
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
                logging.info(f"[{self.__class__.__name__}] Grouping: {group_name}")
                map_group_pid = get_map_group_pid(patients, group_name, cats)
                self.agg_timegap_group(map_group_pid, group_name, cats, n_analysed_samples)
                if (
                    self.state.do_stat_test
                    and group_name in self.state.type_table_groups["multicat_group"]
                ):
                    map_group_pid = get_map_not_group_pid(patients, group_name, cats)
                    group_name = f"{group_name}_bool"
                    self.agg_timegap_group(
                        map_group_pid,
                        group_name,
                        self.state.table_groups[group_name],
                        n_analysed_samples,
                    )
        if self.do_bootstrap and n_analysed_samples < n_samples:
            logging.warning(
                f"[{self.__class__.__name__}] Analysis done only with {n_analysed_samples} (over {n_samples}). Think about increasing GetGroupSize.min_event_group parameter to discard cohorts with too few events."
            )
        if self.use_cache:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.state.timegap_group_df.to_csv(self.timegap_path, index=False)

    def agg_timegap_group(
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
        res = {"group_start_event": [], "time_gap": [], "group": [], "cat": [], "run": []}
        for cat in cats:
            for group_start_event in self.state.list_group_start_event:
                timegap_window = self.timegap_patients[
                    self.timegap_patients["group_start_event"] == group_start_event
                ].set_index("pid")
                sublist_pids = [pid for pid in map_group_pid[cat] if pid in timegap_window.index]
                timegap_cat = timegap_window.loc[sublist_pids]
                median_timegap = timegap_cat["time_gap"].median()
                res["group_start_event"].append(group_start_event)
                res["time_gap"].append(median_timegap)
                res["group"].append(group_name)
                res["cat"].append(cat)
                res["run"].append(it)

        self.state.timegap_group_df = pd.concat((self.state.timegap_group_df, pd.DataFrame(res)))

    def get_timegap_patients(self) -> pd.DataFrame:
        """Get the time gap between alarm and event for all patients used in the analysis. Each event is also mapped
        to its corresponding start of the event group (depending on the duration between the last event or start of the stay and the start of the event).

        Returns
        -------
        pd.DataFrame
            Dataframe gathering the time gap between alarm and event for all patients used in the analysis
        """
        timegap_patients = {"pid": [], "time_gap": [], "group_start_event": [], "event_number": []}
        range_pred = int(self.state.horizon * 60 / self.state.timestep)
        for pid, (pred, label) in self.state.predictions.items():
            if pid in self.state.event_bounds:
                stop_last_event = 0
                event_number = 0
                for start, stop in self.state.event_bounds[pid]:
                    if start == 0:
                        stop_last_event = stop
                        continue
                    start_alarm_window = max(stop_last_event, start - range_pred)
                    if start_alarm_window > self.state.max_len:
                        break
                    group_start_event = self.get_group_start_event(
                        stop_last_event * self.state.timestep, start * self.state.timestep
                    )
                    pred_alarm = pred[start_alarm_window:start]
                    masked_pred_alarm = np.where(
                        np.isnan(label[start_alarm_window:start]), -1, pred_alarm
                    )
                    indices_alarm = np.argwhere(masked_pred_alarm >= self.state.threshold)
                    event_number += 1
                    if len(indices_alarm):
                        ind_first_alarm = indices_alarm[0][0]
                        time_gap = (len(pred_alarm) - ind_first_alarm) * self.state.timestep
                        timegap_patients["pid"].append(pid)
                        timegap_patients["time_gap"].append(time_gap)
                        timegap_patients["group_start_event"].append(group_start_event)
                        timegap_patients["event_number"].append(event_number)
                    stop_last_event = stop
        return pd.DataFrame(timegap_patients)

    def get_group_start_event(self, stop_last_event: int, start_event: int) -> str:
        """Map an event to its start of event group (depending on the duration between last event or start of stay and the start of event).

        Parameters
        ----------
        stop_last_event : int
            End of last event (in minutes)
        start_event : int
            Start of current event (in minutes)

        Returns
        -------
        str
            Start of event group
        """
        len_wo_event = start_event - stop_last_event
        for i in range((len(self.groups_start_event[:-1]))):
            if (
                len_wo_event > self.groups_start_event[i] * 60
                and len_wo_event <= self.groups_start_event[i + 1] * 60
            ):
                return f"{self.groups_start_event[i]}-{self.groups_start_event[i+1]}h"
        return f">{self.groups_start_event[-1]}h"

    def get_list_group_start_event(self) -> List[str]:
        """Get the list of start event groups, as used in the tables.

        Returns
        -------
        List[str]
            List of start event groups
        """
        list_group_start_event = []
        for i in range((len(self.groups_start_event[:-1]))):
            list_group_start_event.append(
                f"{self.groups_start_event[i]}-{self.groups_start_event[i+1]}h"
            )
        list_group_start_event.append(f">{self.groups_start_event[-1]}h")
        return list_group_start_event
