# ========================================
#
# Merging and preprocessing HiRID tables
# Ref: https://github.com/ratschlab/HIRID-ICU-Benchmark/blob/master/icu_benchmarks/common/lookups.py
#
# ========================================
import logging
from pathlib import Path

import numpy as np
import pandas
import pandas as pd

from famews.data.hirid.constants import ADMTIME, PID, VALUE, VARID

STEPS_PER_HOURS = 60


def read_general_table(general_table_path: Path, set_index: str = None) -> pd.DataFrame:
    """
    Read in the HiRID general table

    Parameter
    ---------
    general_table_path: Path
        file path to the general table
    """
    logging.info(f"[General Table] Reading general table from: {general_table_path}")

    if general_table_path.name.endswith(".csv"):
        df = pd.read_csv(general_table_path)
        df[ADMTIME] = pd.to_datetime(df[ADMTIME])
    else:
        df = pd.read_parquet(general_table_path, engine="pyarrow")

    if set_index is not None:
        df = df.set_index(set_index)

    return df


def generate_extended_general_table(
    observation_tables_path: Path, general_table_path: Path, output_path: Path
):
    """
    Generate extended general table and write to `output_path`

    Parameter
    ---------
    observation_tables_path: Path
        path to the HiRID observation tables
    general_table_path: Path
        path to the HiRID general table
    output_path: Path
        path to write the result to
    """

    df_general_table = read_general_table(Path(general_table_path))

    df_obs_tables = pd.read_parquet(
        observation_tables_path,
        engine="pyarrow",
        columns=[PID, VARID, VALUE],
        filters=[(VARID, "in", [10000450, 9990002, 9990004])],
    )

    df_dropped = df_obs_tables.drop_duplicates([PID, VARID])

    df_additional_cols = pd.pivot_table(df_dropped, index=PID, columns=VARID, values=VALUE).rename(
        columns={10000450: "height", 9990002: "APACHE II Group", 9990004: "APACHE IV Group"}
    )

    df_out = df_general_table.merge(
        df_additional_cols, how="left", left_on=PID, right_index=True
    ).reset_index(drop=True)
    df_out[PID] = df_out[PID].astype("int32")
    df_out.to_parquet(output_path)


def read_extended_general_table(path: Path) -> pd.DataFrame:
    """
    Read static information file

    Parameter
    ---------
    static_path: Path
        file path to static patient information table
    """

    df_static = pd.read_parquet(path).reset_index()
    assert ADMTIME in df_static.columns
    return df_static[[PID, ADMTIME, "age", "sex", "height"]]


def read_var_ref_table(var_ref_path: Path) -> pd.DataFrame:
    """
    Read in a var ref table, expects a csv
    using '\t' as a separator.

    Parameter
    ---------
    var_ref_path: str
        path to read file from
    """
    return pd.read_csv(var_ref_path, sep="\t")


def read_stat_ref_table(var_ref_path: Path) -> pd.DataFrame:
    """
    Parameter
    ---------
    stat_ref_path: str
        path to read file from
    """
    return pd.read_parquet(var_ref_path)


def read_reference_table(varref_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read variableid-metavariableid mapping table (HiRID) for the merge step

    Parameter
    ---------
    varref_path: Path
        path to the varref table

    Return
    ------
    varref: pd.DataFrame
        loaded varref dataframe
    pharmaref: pd.DataFrame
        loaded pharmaref dataframe
    """

    varref = pd.read_csv(varref_path, sep="\t", encoding="cp1252")

    pharmaref = varref[varref["type"] == "pharma"].rename(columns={"variableid": "pharmaid"})
    enum_ref = {
        "very short": int(STEPS_PER_HOURS / 12),
        "short": 1 * STEPS_PER_HOURS,
        "4h": 4 * STEPS_PER_HOURS,
        "6h": 6 * STEPS_PER_HOURS,
        "12h": 12 * STEPS_PER_HOURS,
        "24h": 24 * STEPS_PER_HOURS,
        "3d": 72 * STEPS_PER_HOURS,
    }
    pharmaref.loc[:, "pharmaactingperiod_min"] = pharmaref.pharmaactingperiod.apply(
        lambda x: enum_ref[x] if type(x) == str else np.nan
    )
    check_func = (
        lambda x: float(x)
        if type(x) == float or "/" not in x
        else float(x.split("/")[0]) / float(x.split("/")[1])
    )

    # pharmaref.loc[:, "unitconversionfactor"] = pharmaref.unitconversionfactor.apply(check_func)
    pharmaref["unitconversionfactor"] = pharmaref.unitconversionfactor.apply(check_func)

    varref = varref[varref["type"] != "pharma"].copy()
    varref.drop(varref.index[varref.variableid.isnull()], inplace=True)
    # varref.loc[:, "variableid"] = varref.variableid.astype(int)
    varref["variableid"] = varref.variableid.astype(int)
    varref.set_index("variableid", inplace=True)
    return varref, pharmaref
