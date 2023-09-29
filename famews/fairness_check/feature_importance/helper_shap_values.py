import random
from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
import shap
import tables

# import torch


def average_shap_values_multiple_models(
    list_shap_values: List[Dict[int, np.array]]
) -> Dict[int, np.array]:
    """Average the matrices of SHAP values obtained from different models for each patient.

    Parameters
    ----------
    list_shap_values : List[Dict[int, np.array]]
        List of shap values dictionary mapping pid to matrix of shap values obtained from different models

    Returns
    -------
    Dict[int, np.array]
        List of shap values dictionary mapping pid to matrix of shap values that have been averaged from the ones obtained for the different models.
    """
    if len(list_shap_values) == 1:
        return list_shap_values[0]
    shap_values = {}
    for pid in list_shap_values[0].keys():
        shap_values[pid] = np.mean(
            [shap_values_model[pid] for shap_values_model in list_shap_values], axis=0
        )
    return shap_values


def compute_shap_values(
    model_type: str,
    model_path: Path,
    data_h5: tables.group.RootGroup,
    task_name: Union[str, int],
    split: str,
    max_len: int,
) -> Dict[int, np.array]:
    """Compute the matrix of SHAP values for the model predictions of each patient.

    Parameters
    ----------
    model_type : str
        Type of model, only classical_ml supportted
    model_path : Path
        Path to the model
    data_h5 : tables.group.RootGroup
        Table root from h5 file containing data
    task_name : Union[str, int]
        Name or index of the prediction task, useful to fetch the correct label in data_h5
    split : str
        Name of the split of data to used (either var or test)
    max_len : int
        Max length of time series to consider per patient (on timestep min grid)
    Returns
    -------
    Dict[int, np.array]
        Shap values dictionary mapping pid to matrix of shap values

    Raises
    ------
    ValueError
        Prediction task isn't in data_h5
    ValueError
        Model type is not supported
    """
    if isinstance(task_name, int):
        task_ind = task_name
    elif isinstance(task_name, str):
        tasks = np.array([name.decode("utf-8") for name in data_h5["labels"]["tasks"][:]])
        try:
            task_ind = np.where(tasks == task_name)[0][0]
        except IndexError as e:
            raise ValueError(f"Task: {task_name} not found in ML-stage data") from e

    if model_type == "classical_ml":
        model = joblib.load(model_path)
        explainer = shap.Explainer(model)
    else:
        raise ValueError("Model type is not supported")

    patient_windows = data_h5.patient_windows[split][:]
    shap_values = {}
    for start, stop, pid in patient_windows:
        label = data_h5.labels[split][start:stop, task_ind][:max_len]
        feature = data_h5.data[split][start:stop][:max_len][~np.isnan(label)]
        if len(feature):
            shap_values[pid] = explainer.shap_values(feature)[1]
    return shap_values


def average_shap_values_patient(shap_values: Dict[int, np.array]) -> Dict[int, np.array]:
    """Compute the absolute mean SHAP values for each feature and patient.

    Parameters
    ----------
    shap_values : Dict[int, np.array]
        Shap values dictionary mapping pid to matrix of shap values

    Returns
    -------
    Dict[int, np.array]
        Map from pid to the average SHAP values for each feature over the stay.
    """
    avg_shap_patients = {}
    for pid, shap_patient in shap_values.items():
        if len(shap_patient):
            avg_shap_patients[pid] = np.mean(np.absolute(shap_patient), axis=0)
    return avg_shap_patients


def average_shap_values_group(
    map_group_pid: Dict[str, List[int]], avg_shap_patients: Dict[int, np.array]
) -> Dict[str, np.array]:
    """Average the SHAP values over all patients of the cohorts.

    Parameters
    ----------
    map_group_pid : Dict[str, List[int]]
        map each category of patients to its list of pids
    avg_shap_patients : Dict[int, np.array]
        Map from pid to the average SHAP values for each feature over the stay

    Returns
    -------
    Dict[str, np.array]
        Map from the category to the average SHAP values for each feature over all patients of the category.
    """
    avg_shap_group = {}
    for cat, list_pids in map_group_pid.items():
        avg_shap_group[cat] = np.mean([avg_shap_patients[pid] for pid in list_pids], axis=0)
    return avg_shap_group


def get_feat_ranking(avg_shap_values: np.array, feature_names: List[str]) -> List[str]:
    """Compute the feature ranking from the SHAP values.

    Parameters
    ----------
    avg_shap_values : np.array
        Average shap value for each feature
    feature_names : List[str]
        Feature names

    Returns
    -------
    List[str]
        Feature ranking
    """
    ranked_inds = np.argsort(avg_shap_values)[::-1]
    return feature_names[ranked_inds]


def get_feat_ranking_group(
    avg_shap_values_group: Dict[str, np.array], feature_names: List[str]
) -> Dict[str, List[str]]:
    """Compute the feature ranking from the SHAP values for each cohort.

    Parameters
    ----------
    avg_shap_values_group : Dict[str, np.array]
        Map each cohort to the average shap value for each feature
    feature_names : List[str]
        Feature names

    Returns
    -------
    Dict[str, List[str]]
        Map each cohort to its feature ranking
    """
    feat_ranking_group = {}
    for cat, avg_shap_values in avg_shap_values_group.items():
        feat_ranking_group[cat] = get_feat_ranking(avg_shap_values, feature_names)
    return feat_ranking_group


def get_random_grouping(list_pids: List[int]) -> Dict[str, List[int]]:
    """Create a fake patient groupinds by splitting randomly the list of pids.

    Parameters
    ----------
    list_pids : List[int]
        List of pids

    Returns
    -------
    Dict[str, List[int]]
        Map from fake cohort to pids

    """
    n = np.random.randint(2, 7)
    random.shuffle(list_pids)
    map_group_pid = {}
    size_group = len(list_pids) // n
    for i in range(n - 1):
        map_group_pid[str(i)] = list_pids[i * size_group : (i + 1) * size_group]
    map_group_pid[str(n - 1)] = list_pids[(n - 1) * size_group :]
    return map_group_pid


def get_feat_ranking_random_grouping(
    avg_shap_patients: Dict[int, np.array], feature_names: List[str]
) -> Dict[str, Dict[str, List[str]]]:
    """Generate synthetic feature rankings from 100 random patients' groupings.

    Parameters
    ----------
    avg_shap_patients : Dict[int, np.array]
        Map each pid to its average SHAP value for each feature
    feature_names : List[str]
        Feature names

    Returns
    -------
    Dict[str, Dict[str, List[str]]]
        Map fake grouping to fake cohorts and their syntheic feature ranking.
    """
    N = 100
    feat_ranking_random_group = {}
    for i in range(N):
        map_group_pid = get_random_grouping(list(avg_shap_patients.keys()))
        avg_shap_group = average_shap_values_group(map_group_pid, avg_shap_patients)
        feat_ranking_random_group[str(i)] = get_feat_ranking_group(avg_shap_group, feature_names)
    return feat_ranking_random_group
