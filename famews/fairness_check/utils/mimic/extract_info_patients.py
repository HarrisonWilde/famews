from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from famews.fairness_check.utils.helper_groups import build_age_group, build_los_group

UNSPECIFIED_ETHN = [
    "UNKNOWN/NOT SPECIFIED",
    "PATIENT DECLINED TO ANSWER",
    "UNABLE TO OBTAIN",
]


def map_ethnicity(ethnicity: str) -> str:
    """Return ethnicity category.

    Parameters
    ----------
    ethnicity : str
        Ethnicity

    Returns
    -------
    str
        Ethinicity group (4 possible categories)
    """
    if "WHITE" in ethnicity or "PORTUGUESE" in ethnicity:
        return "WHITE"
    if "BLACK" in ethnicity or "CARIBBEAN" in ethnicity:
        return "BLACK"
    if ethnicity in UNSPECIFIED_ETHN:
        return "UNK"
    else:
        return "OTHER"


def create_patients_df_mimic(
    root_mimic: Path,
    output_path: Path,
    predictions: Dict[int, Tuple[np.array, np.array]],
    timestep: int = 5,
) -> pd.DataFrame:
    """Create patients table containing cohort belonging information for each patient.

    Parameters
    ----------
    root_mimic : Path
        Path to MIMIC root directory
    output_path : Path
        Path to store resulting table
    predictions : Dict[int, Tuple[np.array, np.array]]
        Predictions output by the model
    timestep: int, optional
        Timestep of time series in minutes, by default 5

    Returns
    -------
    pd.DataFrame
        Patients dataframe
    """
    res = {pid: {} for pid in predictions.keys()}
    patients_table = pd.read_csv(root_mimic / "PATIENTS.csv", parse_dates=["DOB"]).set_index(
        "SUBJECT_ID"
    )
    icustays_table = pd.read_csv(root_mimic / "ICUSTAYS.csv").set_index("ICUSTAY_ID")
    admissions_table = pd.read_csv(
        root_mimic / "ADMISSIONS.csv", parse_dates=["ADMITTIME"]
    ).set_index("HADM_ID")
    services_table = pd.read_csv(root_mimic / "SERVICES.csv")
    for pid in res:
        subject_id = icustays_table["SUBJECT_ID"].loc[pid]
        hadm_id = icustays_table["HADM_ID"].loc[pid]
        res[pid]["sex"] = patients_table["GENDER"].loc[subject_id]
        # age
        dob = patients_table["DOB"].loc[subject_id].date()
        admittime = admissions_table["ADMITTIME"].loc[hadm_id].date()
        age = int((admittime - dob) / timedelta(days=365))
        res[pid]["age_group"] = build_age_group(age)
        admission_type = admissions_table["ADMISSION_TYPE"].loc[hadm_id]
        if admission_type == "ELECTIVE":
            res[pid]["admission_type"] = "elective"
        else:
            res[pid]["admission_type"] = "emergency"
        # insurance type
        res[pid]["insurance_type"] = admissions_table["INSURANCE"].loc[hadm_id]
        # discharge status
        discharge_status = admissions_table["DISCHARGE_LOCATION"].loc[hadm_id]
        if discharge_status == "DEAD/EXPIRED":
            res[pid]["discharge_status"] = "dead"
        else:
            res[pid]["discharge_status"] = "alive"
        # ethnicity
        ethinicity = admissions_table["ETHNICITY"].loc[hadm_id]
        ethinicity_4 = map_ethnicity(ethinicity)
        res[pid]["ethnicity_4"] = ethinicity_4
        if ethinicity_4 == "WHITE":
            res[pid]["ethnicity_W"] = "WHITE"
        else:
            res[pid]["ethnicity_W"] = "NON-WHITE"
        # surgical status
        try:
            curr_service = services_table[
                (services_table["SUBJECT_ID"] == subject_id)
                & (services_table["HADM_ID"] == hadm_id)
            ].iloc[0]["CURR_SERVICE"]
            if "SURG" in curr_service:
                surgical_status = "Surgical"
            else:
                surgical_status = "Non-surgical"
        except:
            surgical_status = None
        res[pid]["surgical_status"] = surgical_status
    df_patients = pd.DataFrame(res).T
    df_patients.index.name = "PatientID"
    df_patients["los_group"] = df_patients.index.map(
        lambda pid: build_los_group(pid, predictions, timestep)
    )
    df_patients.to_csv(output_path)
    return df_patients
