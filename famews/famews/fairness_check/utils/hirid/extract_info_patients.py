from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from famews.fairness_check.utils.helper_groups import build_age_group, build_los_group
from famews.fairness_check.utils.hirid.map_apache_code import (
    CODE_SURGICAL_APACHE_II,
    CODE_SURGICAL_APACHE_IV,
    MAP_CODE_APACHE_II,
    MAP_CODE_APACHE_IV,
)


def process_apache(apache_ii_group, apache_iv_group) -> Tuple[str, str]:
    """Return apache name based on codes for APACHE II and APACHE IV.

    Parameters
    ----------
    apache_ii_group : int
        code for APACHE II
    apache_iv_group : int
        code for APACHE IV

    Returns
    -------
    Tuple[str, str]
        APACHE group, surgical status
    """
    if np.isfinite(apache_ii_group) and int(apache_ii_group) in MAP_CODE_APACHE_II.keys():
        apache_group = MAP_CODE_APACHE_II[int(apache_ii_group)]
        if int(apache_ii_group) in CODE_SURGICAL_APACHE_II:
            surgical_status = "Surgical"
        else:
            surgical_status = "Non-Surgical"
    elif np.isfinite(apache_iv_group) and int(apache_iv_group) in MAP_CODE_APACHE_IV.keys():
        apache_group = MAP_CODE_APACHE_IV[int(apache_iv_group)]
        if int(apache_iv_group) in CODE_SURGICAL_APACHE_IV:
            surgical_status = "Surgical"
        else:
            surgical_status = "Non-Surgical"
    else:
        apache_group = None
        surgical_status = None
    return apache_group, surgical_status


def add_extra_info(
    df_patients: pd.DataFrame, extra_info_path: Path, extra_info_attributes: List[str]
) -> pd.DataFrame:
    """Add extra grouping attributes to df_patients based on the attributes in extra_info_attributes.

    Parameters
    ----------
    df_patients : pd.DataFrame
        Patients dataframe
    extra_info_path : Path
        Path for HiRID extra info
    extra_info_attributes : List[str]
        List of extra attributes

    Returns
    -------
    pd.DataFrame
        Patients dataframe with extra info attributes
    """
    df_extra_info = pd.read_parquet(extra_info_path).set_index("patientid")
    for att in extra_info_attributes:
        df_patients[att] = df_extra_info[att]
    return df_patients


def create_patients_df_hirid(
    general_table_path: Path,
    output_path: Path,
    predictions: Dict[int, Tuple[np.array, np.array]],
    extra_info_path: Path = None,
    extra_info_attributes: List[str] = [],
    timestep: int = 5,
) -> pd.DataFrame:
    """Create patients table containing cohort belonging information for each patient.

    Parameters
    ----------
    general_table_path : Path
        Path to HiRID general table
    output_path : Path
        Path to store resulting table
    predictions : Dict[int, Tuple[np.array, np.array]]
        Predictions output by the model
    extra_info_path : Path, optional
        Path for HiRID extra info, by default None
    extra_info_attributes : List[str], optional
        List of extra attributes, by default []
    timestep: int, optional
        Timestep of time series in minutes, by default 5

    Returns
    -------
    pd.DataFrame
        Patients dataframe
    """
    df_patients = pd.read_parquet(general_table_path).set_index(
        "patientid"
    )  # assume sex, age and discharge_status in this table
    general_table_cols = df_patients.columns
    df_patients["age_group"] = df_patients["age"].apply(build_age_group)
    df_patients["los_group"] = df_patients.index.map(
        lambda pid: build_los_group(pid, predictions, timestep)
    )
    if "APACHE II Group" in general_table_cols and "APACHE IV Group" in general_table_cols:
        df_patients[["APACHE_group", "surgical_status"]] = df_patients[
            ["APACHE II Group", "APACHE IV Group"]
        ].apply(
            lambda s: process_apache(s["APACHE II Group"], s["APACHE IV Group"]),
            result_type="expand",
            axis=1,
        )
    if extra_info_path:
        df_patients = add_extra_info(df_patients, extra_info_path, extra_info_attributes)
    df_patients.to_csv(output_path)
    return df_patients
