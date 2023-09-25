import logging
import pickle
from pathlib import Path
from typing import Callable, List

import gin
import pandas as pd
import tables
import yaml
from yaml.loader import BaseLoader

from famews.fairness_check.utils.compute_threshold_score import GetThresholdScore
from famews.fairness_check.utils.get_pop_info import GetGroupSize
from famews.fairness_check.utils.helper_endpoints import (
    get_event_bounds_patients,
    merge_event_bounds,
)
from famews.fairness_check.utils.hirid.extract_info_patients import (
    create_patients_df_hirid,
)
from famews.fairness_check.utils.mimic.extract_info_patients import (
    create_patients_df_mimic,
)
from famews.pipeline import PipelineBase, PipelineState, StatefulPipelineStage
from famews.train.utils import save_gin_config_file
from famews.utils.handle_predictions import (
    HandleMultiplePredictions,
    HandlePredictions,
)


@gin.configurable("FairnessAnalysisPipeline")
class FairnessAnalysisPipeline(PipelineBase):
    name = "Fairness Analysis Pipeline"

    def __init__(
        self,
        log_dir: Path,
        num_workers: int = 1,
        analysis_stages: list[Callable] = [],
        use_multiple_predictions: bool = True,
        threshold_score: float = None,
        name_threshold: str = "event_recall_0.9",
        horizon: int = 12,
        max_len: int = gin.REQUIRED,
        timestep: int = 5,
        do_stat_test: bool = False,
        significance_level: float = 0.001,
        filter_delta: float = 0,
    ):
        """Run fairness analysis pipeline.

        Parameters
        ----------
        log_dir : Path
            Log directory
        num_workers : int, optional
            Number of workers, by default 1
        analysis_stages : list[Callable], optional
            List of pipeline stages to run, by default []
        use_multiple_predictions : bool, optional
            Whether to use average of several models (True) or a unique model (False), by default True
        threshold_score : float, optional
            Threshold value, by default None
        name_threshold : str, optional
            Name of threshold in the format {metric_name}_{target_value}, by default "event_recall_0.9"
        horizon : int, optional
            Prediction horizon in hour, by default 12
        max_len : int, optional
           Max length of time series to consider per patient (on 5 min grid), by default gin.REQUIRED
        timestep: int, optional
            Timestep of time series in minutes, by default 5
        do_stat_test : bool, optional
            Whether to run statistical testing, if yes then the analysis will be run with bootstrapping, by default False
        significance_level : float, optional
            Significance level for the statistical testing, by default 0.001
        filter_delta : float, optional
            Filter on the delta, if greater than filter_delta the delta is considered as important, by default 0

        Raises
        ------
        ValueError
            Invalid Pipeline Stage passed to analysis stages.
        """
        super().__init__([], num_workers)
        self.state.log_dir = Path(log_dir)
        self.state.threshold = threshold_score
        self.state.name_threshold = name_threshold
        self.state.horizon = horizon
        self.state.max_len = max_len
        self.state.timestep = timestep
        self.state.do_stat_test = do_stat_test
        self.state.significance_level = significance_level
        self.state.filter_delta = filter_delta
        load_group_stage = SetUpGroupDef(self.state)
        load_patients_stage = LoadPatientsDf(self.state)
        load_event_bounds_stage = LoadEventBounds(self.state)
        if use_multiple_predictions:
            load_predictions_stage = HandleMultiplePredictions(self.state)
        else:
            load_predictions_stage = HandlePredictions(self.state)
        get_group_size_stage = GetGroupSize(self.state)
        load_pid_split_stage = LoadSplitPids(self.state)
        setup_stages = [
            load_predictions_stage,
            load_group_stage,
            load_patients_stage,
            load_event_bounds_stage,
            get_group_size_stage,
            load_pid_split_stage,
        ]
        if self.state.threshold is None:
            get_threshold_stage = GetThresholdScore(self.state)
            setup_stages.append(get_threshold_stage)
        else:
            self.state.name_threshold = f"threshold_{self.state.threshold}"
        # then look in gin config which analysis stage to run
        self.state.fairness_log_dir = self.state.log_dir / "fairness_check"
        if self.state.do_stat_test:
            self.state.fairness_log_dir = self.state.fairness_log_dir / "w_stat_test"
        else:
            self.state.fairness_log_dir = self.state.fairness_log_dir / "no_stat_test"
        self.state.fairness_log_dir = self.state.fairness_log_dir / self.state.name_threshold
        if self.state.merge_time_min:
            self.state.fairness_log_dir = (
                self.state.fairness_log_dir / f"merge_event_{self.state.merge_time_min}min"
            )

        additional_stages = []
        for stage_constructor in analysis_stages:
            stage = stage_constructor(self.state)
            if isinstance(stage, StatefulPipelineStage):
                additional_stages.append(stage)
            else:
                raise ValueError(f"Invalid stage: {stage} must be a `StatefulPipelineStage`")

        final_stages = setup_stages + additional_stages + [Cleanup(self.state)]
        self.add_stages(final_stages)

        logging.debug(f"[{self.name}] initialized with stages:")
        for s in self.stages:
            logging.debug(f"\t{s}")


@gin.configurable("SetUpGroupDef", denylist=["state"])
class SetUpGroupDef(StatefulPipelineStage):

    name = "Set up Group Definition"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        group_config_file: Path = gin.REQUIRED,
        **kwargs,
    ):
        """Define patient grouping through a YAML config file.

        Parameters
        ----------
        state : PipelineState
            Pipeline state
        num_workers : int, optional
            Number of workers, by default 1
        group_config_file : Path, optional
            Config file for grouping definition, by default gin.REQUIRED
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.group_config_file = group_config_file

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains grouping definition

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        return hasattr(self.state, "groups") and self.state.groups is not None

    def run(self):
        """
        Load grouping definition. If we run statistical testing, the config also define how to compare cohorts.
        """
        with open(self.group_config_file, "r") as f:
            try:
                group_config_dict = yaml.load(f, Loader=BaseLoader)
            except yaml.YAMLError as e:
                logging.error(
                    f"[{self.__class__.__name__}] YAML error while loading the grouping config: {e}"
                )

        self.state.groups = {
            **group_config_dict["group"],
        }
        if self.state.do_stat_test:
            self.state.type_table_groups = group_config_dict["type_table_group"]


@gin.configurable("LoadPatientsDf", denylist=["state"])
class LoadPatientsDf(StatefulPipelineStage):
    name = "Load Patients df"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        patients_df_file: Path = gin.REQUIRED,
        pid_name: str = "patientid",
        patients_source: str = "hirid",  # or "mimic"
        general_table_hirid_path: Path = None,
        extra_info_hirid_path: Path = None,
        extra_info_hirid_attributes: List[str] = [],
        root_mimic: Path = None,
        **kwargs,
    ):
        """Load patients demographics information. This is needed to build groupings.

        Parameters
        ----------
        state : PipelineState
            Pipeline state
        num_workers : int, optional
            Number of workers, by default 1
        patients_df_file : Path, optional
            Patients dataframe, by default gin.REQUIRED
        pid_name : str, optional
            Column name for patient id, by default "patientid"
        patients_source : str, optional
            Data source, by default "hirid"
        extra_info_hirid_path : Path, optional
            Path for dataframe containing extra information for HiRID, by default None
        extra_info_hirid_attributes : List[str], optional
            List of attributes to load from extra info table, by default []
        root_mimic : Path, optional
            (if patients_source is mimic) root directory of mimic tables, by default None
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.patients_df_file = Path(patients_df_file)
        self.pid_name = pid_name
        self.patients_source = patients_source
        self.general_table_hirid_path = general_table_hirid_path
        self.extra_info_hirid_path = extra_info_hirid_path
        self.extra_info_hirid_attributes = extra_info_hirid_attributes
        self.root_mimic = root_mimic

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains pateints' dataframe.

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        return hasattr(self.state, "patients_df") and self.state.patients_df is not None

    def run(self):
        """Load/create patients dataframe. Check that each grouping present in the self.state.groups object has a
        corresponding column in the patient dataframe.

        Raises
        ------
        ValueError
            Unsupported data source
        Exception
            A grouping is present in the grouping definition but not in the patients dataframe
        """
        if self.patients_df_file.exists():
            self.state.patients_df = pd.read_csv(self.patients_df_file).set_index(self.pid_name)
        else:
            if self.patients_source == "hirid":
                self.state.patients_df = create_patients_df_hirid(
                    self.general_table_hirid_path,
                    self.patients_df_file,
                    self.state.predictions,
                    self.extra_info_hirid_path,
                    self.extra_info_hirid_attributes,
                    self.state.timestep,
                )
            elif self.patients_source == "mimic":
                self.state.patients_df = create_patients_df_mimic(
                    Path(self.root_mimic),
                    self.patients_df_file,
                    self.state.predictions,
                    self.state.timestep,
                )
            else:
                raise ValueError(
                    f"[{self.__class__.__name__}] Unsupported patients sources: {self.patients_source}"
                )
        # check that all groupings are in patients_df columns
        for group_name in self.state.groups:
            if group_name not in self.state.patients_df.columns:
                raise Exception(
                    f"[{self.__class__.__name__}] Grouping not defined in patients dataframe: {group_name}"
                )


@gin.configurable("LoadEventBounds", denylist=["state"])
class LoadEventBounds(StatefulPipelineStage):
    # maybe at some point we will move this and actually do the computations or load the file depending on whether it has already been generated and stored on disk
    name = "Load Event Bounds"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        event_bounds_file: Path = gin.REQUIRED,
        patient_endpoint_data_dir: Path = None,
        event_column: str = "circ_failure_status",
        patients_source: str = "hirid",  # or "mimic"
        merge_time_min: int = 0,
        **kwargs,
    ):
        """Stage to load event bounds. If aren't already cached, we generate them from endpoints data.

        Parameters
        ----------
        state : PipelineState
            Pipeline state
        num_workers : int, optional
            Number of workers, by default 1
        event_bounds_file : Path, optional
            File containing event bounds, by default gin.REQUIRED
        patient_endpoint_data_dir : Path, optional
            Directory containing endpoints data, by default None
        event_column : str, optional
           Column name for event status, by default "circ_failure_status"
        patients_source : str, optional
            Source of data, by default "hirid"
        merge_time_min: int, optional
            Minimal time between event, otherwise they are merged, by default 0
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.event_bounds_file = Path(event_bounds_file)
        self.patient_endpoint_data_dir = patient_endpoint_data_dir
        self.event_column = event_column
        self.patients_source = patients_source
        self.merge_time_min = merge_time_min
        self.state.merge_time_min = merge_time_min

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains event bounds

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        return hasattr(self.state, "event_bounds") and self.state.event_bounds is not None

    def run(self):
        """
        Load event bounds. If merge_time_min isn't 0, then we merge event bounds.
        """
        if self.event_bounds_file.exists():
            with open(self.event_bounds_file, "rb") as f:
                self.state.event_bounds = pickle.load(f)
        else:
            self.state.event_bounds = get_event_bounds_patients(
                Path(self.patient_endpoint_data_dir),
                self.event_column,
                self.patients_source,
                self.state.timestep,
            )
            with open(self.event_bounds_file, "wb") as f:
                pickle.dump(self.state.event_bounds, f)
        if self.merge_time_min:
            self.state.event_bounds = merge_event_bounds(
                self.state.event_bounds, self.merge_time_min, self.state.timestep
            )


@gin.configurable("LoadSplitPids", denylist=["state"])
class LoadSplitPids(StatefulPipelineStage):
    name = "Load Split PIDs"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        split_file: Path = gin.REQUIRED,
        data_table_file: Path = None,
        **kwargs,
    ):
        """Stage to load pid split. If aren't already cached, we generate them from data table.

        Parameters
        ----------
        state : PipelineState
            Pipeline state
        num_workers : int, optional
            Number of workers, by default 1
        split_file : Path, optional
            File containing for each split the list of pids, by default gin.REQUIRED
        data_table_file : Path, optional
            Path containing table with split definition, by default None
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.split_file = Path(split_file)
        self.data_table_file = data_table_file

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains pid split.

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        return hasattr(self.state, "pid_split") and self.state.pid_split is not None

    def run(self):
        """
        Load pids split.
        """
        if self.split_file.exists():
            with open(self.split_file, "rb") as f:
                self.state.pid_split = pickle.load(f)
        else:
            self.state.pid_split = {}
            patient_windows = tables.open_file(self.data_table_file).root.patient_windows
            for split in ["train", "val", "test"]:
                self.state.pid_split[split] = patient_windows[split][:, 2]
            with open(self.split_file, "wb") as f:
                pickle.dump(self.state.pid_split, f)


class Cleanup(StatefulPipelineStage):
    name = "Clean up after Fairness Analysis"

    def __init__(self, state: PipelineState, num_workers: int = 1, **kwargs) -> None:
        """Clean the GIN state and log final config.

        Parameters
        ----------
        state : PipelineState
            Pipeline state
        num_workers : int, optional
            Number of workers, by default 1
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        self.done = False

    def runnable(self) -> bool:
        return True

    def is_done(self) -> bool:
        """Check if the stage has been completed

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        return self.done

    def run(self):
        gin_log_path = Path(self.state.fairness_log_dir)
        gin_log_path.mkdir(parents=True, exist_ok=True)
        save_gin_config_file(gin_log_path, filename="final_config.gin")
        gin.clear_config()
        self.done = True
