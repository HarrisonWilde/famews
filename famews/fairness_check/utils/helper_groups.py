from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def get_map_group_pid(
    patient_df: pd.DataFrame, group_name: str, cats: List[str]
) -> Dict[str, List[int]]:
    """Map to each category in cats the corresping list of patient ids.

    Parameters
    ----------
    patient_df : pd.DataFrame
        dataframe indicating to which cohort each patient belongs, indexed by the pid
    group_name : str
       Group name_
    cats : List[str]
       List of categories for the grouping

    Returns
    -------
    Dict[str, List[int]]
        dictionary {cat: [pids]}
    """
    map_group_pid = {}
    for cat in cats:
        list_pids = patient_df[patient_df[group_name] == cat].index
        map_group_pid[cat] = list_pids
    return map_group_pid


def get_map_not_group_pid(
    patients_df: pd.DataFrame, group_name: str, cats: List[str]
) -> Dict[str, List[int]]:
    """For multicategories grouping, map to each not_cat the corresponding list of patient ids.

    Parameters
    ----------
    patients_df : pd.DataFrame
        Patients dataframe
    group_name : str
        Group name
    cats : List[str]
        List of categories

    Returns
    -------
    Dict[str, List[int]]
        Map to each not_cat  the corresponding list of patient ids
    """
    map_group_pid = {}
    for cat in cats:
        list_pids = patients_df[patients_df[group_name] != cat].index
        map_group_pid[f"Not {cat}"] = list_pids
    return map_group_pid


def construct_table_groups(
    groups: Dict[str, List[str]], type_table_groups: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Define grouping used for statistical tests.

    Parameters
    ----------
    groups : Dict[str, List[str]]
        Groupings {group_name: [categories]}
    type_table_groups : Dict[str, List[str]]
        Dictionary defining how to compare categories within a grouping for the statistical test {type_comparison: [group_name]}

    Returns
    -------
    Dict[str, List[str]]
        Grouping used for statistical test {group_name: [categories]}
    """
    table_groups = {}
    for group in type_table_groups["multicat_group"]:
        table_groups[f"{group}_bool"] = [f"Not {cat}" for cat in groups[group]]
    return table_groups


def build_age_group(age):
    """Given age, return corresponding age category.

    Parameters
    ----------
    age : int
        age patient

    Returns
    -------
    str
        age category
    """
    if age < 50:
        return "<50"
    if age < 65:
        return "50-65"
    if age < 75:
        return "65-75"
    if age < 85:
        return "75-85"
    return ">85"


def build_los_group(pid: int, predictions: Dict[int, Tuple[np.array, np.array]], timestep: int = 5):
    """Extract LOS category from length of predictions.

    Parameters
    ----------
    pid : int
        Patient id
    predictions: Dict[int, Tuple[np.array, np.array]]
        Predictions output by the model
    timestep: int, optional
        Timestep of time series in minutes, by default 5

    Returns
    -------
    str
        los category
    """
    if pid not in predictions:
        return None
    los = len(predictions[pid][0]) * timestep / (60 * 24)
    if los < 16 / 24:
        return "<16h"
    if los < 1:
        return "16-24h"
    if los < 2:
        return "24-48h"
    else:
        return ">48h"
