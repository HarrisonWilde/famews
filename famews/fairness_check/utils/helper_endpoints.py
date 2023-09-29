import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from famews.data.utils import MarkedDataset


def get_event_bounds_patients(
    patient_endpoint_data_dir: Path, event_column: str, patients_source: str, timestep: int = 5
) -> Dict[str, List[Tuple[int, int]]]:
    """_summary_

    Parameters
    ----------
    patient_endpoint_data_dir : Path
        Directory with endpoints dataframes
    event_column : str
       Name of event status column
    patients_source : str
        Source of data
    timestep: int, optional
        Timestep of time series in minutes, by default 5

    Returns
    -------
    Dict[str, List[Tuple[int, int]]]
        Event bounds

    Raises
    ------
    ValueError
        Unsupported data source
    """
    if patients_source == "hirid":
        from famews.data.hirid.constants import (
            BATCH_PARQUET_PATTERN,
            PID,
            REL_DATETIME,
        )
    # still need to implement it for mimic since for the moment we don't have the preprocessing for it (only what the output of fair_ml_in_icu)
    # elif patients_source == 'mimic':
    #    from famews.data.mimic.constants import BATCH_PARQUET_PATTERN, REL_DATETIME, PID
    else:
        raise ValueError(f"[Load Event Bounds] Unsupported patients sources: {patients_source}")
    patient_parquet_file = next(patient_endpoint_data_dir.rglob("*.parquet"))
    patient_parquet_directory = patient_parquet_file.parent

    patient_endpoint_dataset = MarkedDataset(
        patient_parquet_directory, part_re=BATCH_PARQUET_PATTERN, force=False
    )
    parts_list = patient_endpoint_dataset.list_parts()
    logging.info(f"[Load Event Bounds] Loading patient endpoint data: {patient_parquet_directory}")
    keep_columns = [event_column, REL_DATETIME, PID]
    event_bounds = {}
    for part in tqdm(parts_list):
        patient_df = pd.read_parquet(part, columns=keep_columns)

        patient_df["InEvent"] = patient_df[event_column] > 0
        patient_df = patient_df.drop(columns=[event_column])
        contains_nans = patient_df["InEvent"].isna().sum() > 0
        if contains_nans:
            patient_df["InEvent"].fillna(False, inplace=True)
        grouped_patient_df = patient_df.groupby(PID)
        for pid, df in grouped_patient_df:
            event_bounds[pid] = get_event_bounds(df, REL_DATETIME, timestep)
    return event_bounds


def get_event_bounds(
    df_endpt: pd.DataFrame, rel_datetime: str, timestep: int = 5
) -> List[Tuple[int, int]]:
    """_summary_

    Parameters
    ----------
    df_endpt : pd.DataFrame
        Dataframe containing endpoints of a patient
    rel_datetime : str
        Column name for relative datetime
    timestep: int, optional
        Timestep of time series in minutes, by default 5

    Returns
    -------
    List[Tuple[int, int]]
        List of event bounds for a patient
    """
    df_event = df_endpt.set_index(rel_datetime).sort_index()["InEvent"]
    list_events = []
    in_event = False
    start_event = None
    for dt, in_event_dt in df_event.items():
        if in_event_dt and not in_event:
            in_event = True
            start_event = int(dt / (timestep * 60))
        elif not in_event_dt and in_event:
            in_event = False
            list_events.append((start_event, int(dt / (timestep * 60))))
    if in_event:
        list_events.append((start_event, int(dt / (timestep * 60)) + 1))
    return list_events


def merge_event_bounds(
    event_bounds: Dict[str, List[Tuple[int, int]]], merge_time_min: int, timestep: int = 5
) -> Dict[str, List[Tuple[int, int]]]:
    """Merge event bounds. When the duration separating two events is smaller than merge_time_min then we merge them.

    Parameters
    ----------
    event_bounds : Dict[str, List[Tuple[int, int]]]
        Dictionary of event bounds: {pid: [(start_event, stop_event)]}
    merge_time_min : int
        Minimum duration separating two events after merging in minutes
    timestep: int, optional
        Timestep of time series in minutes, by default 5

    Returns
    -------
    Dict[str, List[Tuple[int, int]]]
        Merged event bounds
    """
    merge_step = merge_time_min // timestep
    merged_event_bounds = {}
    for pid, bounds in event_bounds.items():
        merged_bounds = []
        prev_stop = None
        for start, stop in bounds:
            if prev_stop is not None and start - prev_stop <= merge_step:
                (prev_start, _) = merged_bounds.pop()
                merged_bounds.append((prev_start, stop))
            else:
                merged_bounds.append((start, stop))
            prev_stop = stop
        merged_event_bounds[pid] = merged_bounds
    return merged_event_bounds
