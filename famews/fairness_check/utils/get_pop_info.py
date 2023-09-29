import logging
from typing import Dict, List, Tuple

import gin
import pandas as pd

from famews.fairness_check.utils.helper_groups import construct_table_groups
from famews.pipeline import PipelineState, StatefulPipelineStage


@gin.configurable("GetGroupSize", denylist=["state"])
class GetGroupSize(StatefulPipelineStage):
    name = "Get Group Size"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        min_event_group: int = 1,
        **kwargs,
    ):
        """GetGroupSize initialisation

        Parameters
        ----------
        state : PipelineState
            pipeline state
        num_workers : int, optional
            number of workers, by default 1
        min_event_group : int, optional
            minimal number of event per cohort to run the analysis, by default 1
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.min_event_group = min_event_group

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains group size information

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        return False

    def run(self):
        """
        Get the size of each cohort. Remove cohorts with insufficient number of events.
        """
        range_pred = int(self.state.horizon * 60 / self.state.timestep)
        test_pids = self.state.predictions.keys()
        test_patients_df = self.state.patients_df.loc[test_pids]
        res = {
            group_name: {cat: {} for cat in cats} for group_name, cats in self.state.groups.items()
        }
        to_remove = {}
        for group_name, cats in self.state.groups.items():
            for cat in cats:
                cat_patients_df = test_patients_df[test_patients_df[group_name] == cat]
                res[group_name][cat]["size"] = len(cat_patients_df)
                count_event = 0
                for pid in cat_patients_df.index:
                    if pid in self.state.event_bounds and len(self.state.event_bounds[pid]) > 0:
                        event_bound_pid = self.state.event_bounds[pid]
                        for (start, stop) in event_bound_pid:
                            if start <= self.state.max_len + range_pred and start > 0:
                                count_event += 1
                                break
                res[group_name][cat]["count_w_event"] = count_event
                if count_event < self.min_event_group:
                    if group_name in to_remove:
                        to_remove[group_name].append(cat)
                    else:
                        to_remove[group_name] = [cat]
        self.state.group_size = res
        for group_name, cats_to_remove in to_remove.items():
            logging.warning(
                f"{cats_to_remove} in {group_name} have less than {self.min_event_group} events, so we remove them from the grouping!"
            )
            cats = self.state.groups[group_name]
            for cat in cats_to_remove:
                cats.remove(cat)
            if len(cats) < 2:
                del self.state.groups[group_name]
                logging.warning(
                    f"Removing {group_name} completly from grouping because it has {len(cats)} categories."
                )
            else:
                self.state.groups[group_name] = cats
        self.state.removed_groups = to_remove
        if self.state.do_stat_test:
            self.state.table_groups = construct_table_groups(
                self.state.groups, self.state.type_table_groups
            )


def check_group_has_event(
    patients_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    event_bounds: Dict[int, List[Tuple[int, int]]],
    max_len: int,
    horizon: int,
) -> bool:
    """Check whether all patient cohorts have events.

    Parameters
    ----------
    patients_df : pd.DataFrame
        Patient dataframe
    groups : Dict[str, List[str]]
        Groups {group_name: [categories]}
    event_bounds : Dict[int, List[Tuple[int, int]]]
        Event bounds {pid: [(start, stop)]}
    max_len : int
        Maximal length of ts considred for each patient
    horizon : int
        Alarm horizon in hours

    Returns
    -------
    bool
        Whether all patient cohorts have events
    """
    range_pred = int(horizon * 60 / 5)
    for group_name, cats in groups.items():
        for cat in cats:
            cat_patients_df = patients_df[patients_df[group_name] == cat]
            count_event = 0
            for pid in cat_patients_df.index:
                if pid in event_bounds and len(event_bounds[pid]) > 0:
                    event_bound_pid = event_bounds[pid]
                    for (start, stop) in event_bound_pid:
                        if start <= max_len + range_pred and start > 0:
                            count_event += 1
                            break
            if not count_event:
                logging.warning(
                    f"{cat} in {group_name} has no event in this bootstrap sample, go to the next!"
                )
                return False
    return True
