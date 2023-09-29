#!/usr/bin/env python
# coding: utf-8
# ===========================================================
#
# Run HiRID Preprocessing
# https://github.com/ratschlab/HIRID-ICU-Benchmark/blob/master/icu_benchmarks/synthetic_data/generate_simple_fake_data.py
#
# ===========================================================
import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from famews.data.hirid import constants


def get_date_in_range(start: datetime.datetime, end: datetime.datetime, n: int) -> pd.DataFrame:
    """
    Get a random date within a range

    Parameter
    ---------
    start: datetime.datetime
        start date of range
    end: datetime.datetime
        end date of range
    n: int
    """
    random_date = np.random.randint(int(start.timestamp()), int(end.timestamp()), n)
    return pd.to_datetime(random_date, unit="s")


def generate_summaries_dict(
    summaries_df: pd.DataFrame, varref_df: pd.DataFrame
) -> dict[int, tuple[float, ...]]:
    """
    Generate a statistics summaries dictionary

    Parameter
    ---------
    summaries_df: pd.DataFrame
        df with summary information
    varref_df: pd.DataFrame
        variable reference dataframe
    """
    merged = summaries_df.merge(
        varref_df, how="left", on=constants.VARID, suffixes=("_calc", "_ref")
    )

    # take data from varref if available, otherwise take stats as observed in the entire dataset
    merged["mean_ref"] = merged["mean_ref"].fillna(merged["mean_calc"])
    merged["standard_deviation_ref"] = merged["standard_deviation_ref"].fillna(
        merged["standard_deviation_calc"]
    )
    merged["lowerbound_ref"] = merged["lowerbound"].fillna(merged["min"])
    merged["upperbound_ref"] = merged["upperbound"].fillna(merged["max"])

    return {
        r[constants.VARID]: (
            r["mean_ref"],
            r["standard_deviation_ref"],
            r["lowerbound_ref"],
            r["upperbound_ref"],
            r["rounding_remainders"] < 0.5,
        )  # proxy for knowing whether the variable has integer values
        for _, r in merged.iterrows()
    }


def get_timestamps_diffs(duration: int, dists_df: pd.DataFrame, max_length: int) -> list[int]:
    """
    TODO
    """
    cur_time = 0

    diffs: list[int] = []

    weights = dists_df["count"] / dists_df["count"].sum()

    while cur_time < duration and len(diffs) < max_length:
        d = np.random.choice(dists_df["time_diff"], p=weights)
        duration += d

        diffs.append(d)

    return diffs


def get_varids(n: int, summaries_df: pd.DataFrame) -> np.ndarray:
    """
    Get random variable id(s)

    Parameter
    ---------
    n: int
        number of samples
    summaries_df: pd.DataFrame
        df with summary statistics
    """
    weights = summaries_df["count"] / summaries_df["count"].sum()
    return np.random.choice(summaries_df[constants.VARID], p=weights, size=n)


def sample_values(varid: int, summaries: dict[int, tuple[float, ...]]) -> np.ndarray:
    """
    Sample values for a specific variable

    Parameter
    ---------
    varid: int
        id of variable to sample from
    summaries: dict[str, tuple[float]]
        summary statistics for the variables
    """
    m, std, min_val, max_val, is_integer_val = summaries[varid]

    digits = 0 if is_integer_val else 2

    vals = np.round(np.random.normal(loc=m, scale=std), digits)
    return np.clip(vals, a_min=min_val, a_max=max_val)


def generate_fake_general_data(nr_patients: int) -> pd.DataFrame:
    """
    Generate synthetic static patient data

    Parameter
    ---------
    nr_patients: int
        number of patients to generate data for
    """
    pids = pd.Series(range(0, nr_patients)) + 1

    admission_times = get_date_in_range(
        datetime.datetime(2100, 1, 1), datetime.datetime(2199, 1, 1), nr_patients
    )

    return pd.DataFrame(
        {
            constants.PID: pids,
            "admissiontime": admission_times,
            "sex": np.random.choice(["M", "F"], size=nr_patients),
            "age": np.random.randint(4, 18, size=nr_patients) * 5,  # ages 20-90 in 5y
            "discharge_status": np.random.choice(["alive", "dead"], size=nr_patients),
        }
    )


def get_fake_obs_data(
    pid: int,
    duration: int,
    admission_time: datetime.datetime,
    summaries_df: pd.DataFrame,
    time_diff_dists: pd.DataFrame,
    varref_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get synthetic observation data

    Parameter
    ---------
    pid: int
        patient id
    duration: int
        duration of stay
    admission_time: datetime.datetime
        -
    summaries_df: pd.DataFrame
        dataset summary statistics
    time_diff_dists: pd.DataFrame
        time difference between events statistics
    varref_df: pd.DataFrame
        variable reference dataframe
    """
    ts = get_timestamps_diffs(duration, time_diff_dists, 10000)

    length = len(ts)
    # datetimes = pd.Series(
    #     pd.to_datetime(np.cumsum(ts) + admission_time.timestamp(), unit="s")
    # ).dt.floor("s")
    datetimes = pd.Series(pd.to_timedelta(np.cumsum(ts), unit="s")) + admission_time
    var_ids = get_varids(length, summaries_df)
    summaries_dict = generate_summaries_dict(summaries_df, varref_df)

    values = [sample_values(vid, summaries_dict) for vid in var_ids]

    return pd.DataFrame(
        {
            "datetime": datetimes,
            "entertime": datetimes,  # set equal to 'datetime' for simplicity
            constants.PID: pid,
            "status": 8,
            "stringvalue": None,
            "type": None,
            "value": values,
            constants.VARID: var_ids,
        }
    )


def get_fake_pharma_data(
    pid: int,
    duration: int,
    admission_time: datetime.datetime,
    summaries_df: pd.DataFrame,
    time_diff_dists: pd.DataFrame,
    varref_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create synthetic (fake) pharma data

    Parameter
    ---------
    pid: int
        patient id
    duration: int
        duration of stay
    admission_time: datetime.datetime
        -
    summaries_df: pd.DataFrame
        dataset summary statistics
    time_diff_dists: pd.DataFrame
        time difference between events statistics
    varref_df: pd.DataFrame
        variable reference dataframe
    """
    ts = get_timestamps_diffs(duration, time_diff_dists, 500)

    length = len(ts)
    # datetimes = pd.Series(
    #     pd.to_datetime(np.cumsum(ts) + admission_time.timestamp(), unit="s")
    # ).dt.floor("s")
    datetimes = pd.Series(pd.to_timedelta(np.cumsum(ts), unit="s")) + admission_time

    var_ids = get_varids(length, summaries_df)
    summaries_dict = generate_summaries_dict(summaries_df, varref_df)

    values = [sample_values(vid, summaries_dict) for vid in var_ids]

    return pd.DataFrame(
        {
            constants.PID: pid,
            "pharmaid": var_ids,
            "givenat": datetimes,
            "enteredentryat": datetimes,  # set equal to 'datetime' for simplicity
            "givendose": values,
            "cumulativedose": np.nan,
            "fluidamount_calc": np.nan,
            "cumulfluidamount_calc": np.nan,
            "doseunit": np.nan,
            "recordstatus": 8,
            "infusionid": np.random.randint(0, 10000, size=length),
            "typeid": 1,
        }
    )


def _write_part(df: pd.DataFrame, path: Path, part: int = 0):
    """
    Write df to path as parquet file

    Parameter
    ---------
    df: pd.DataFrame
        df to write
    path: Path
        path to write to (directory)
    part: int
        part number of file (default: 0)
    """
    path.mkdir(exist_ok=True, parents=True)
    df.to_parquet(path / f"part-{part}.parquet", index=False)


def get_parser(argv=None):
    """Parse arguments"""

    parser = argparse.ArgumentParser(description="Generate fake data")

    parser.add_argument("stats_dir", help="output dir of collect_stats.py", type=Path)
    parser.add_argument("output_dir", help="output dir of collect_stats.py", type=Path)
    parser.add_argument(
        "--var-ref-path", help="Path to load the variable references from", type=Path
    )
    parser.add_argument("--seed", help="random seed", type=int, default=40510)
    parser.add_argument(
        "--nr-patients", type=int, default=10, help="number of patients to generate per part file"
    )
    parser.add_argument("--nr-parts", type=int, default=1, help="Number of part files to generate")

    return parser.parse_args(argv)


def main(argv=None):
    """
    Generate synthetic (fake) HiRID data
    """
    args = get_parser(argv)

    stats_dir = args.stats_dir
    output_dir = args.output_dir

    var_ref_path = args.var_ref_path
    nr_patients = args.nr_patients

    np.random.seed(args.seed)

    file_extension = None
    pandas_reader = None
    if (stats_dir / "observation_tables_stats.parquet").exists():
        file_extension = "parquet"
        pandas_reader = pd.read_parquet
    elif (stats_dir / "observation_tables_stats.csv").exists():
        file_extension = "csv"
        pandas_reader = pd.read_csv
    else:
        logging.error(f"Could not find observation_tables_stats.[csv,parquet] in {stats_dir}")
        raise FileNotFoundError()

    logging.info(f"[MAIN] Reading stats data from {stats_dir}; detected {file_extension} extension")
    logging.info(f"[MAIN] generate {args.nr_parts} part files of {args.nr_patients} patients each")

    total_nr_patients = nr_patients * args.nr_parts
    df_general_fake = generate_fake_general_data(total_nr_patients)
    _write_part(df_general_fake, output_dir / "general_table")

    for i in range(args.nr_parts):

        logging.info(f"[MAIN] generate part: {i}")

        length_of_stay = np.random.randint(
            4 * 3600, 10 * 3600, nr_patients
        )  # between 4 and 10 hours

        time_diff_dists_obs = pandas_reader(stats_dir / f"time_diff_stats_obs.{file_extension}")
        time_diff_dists_pharma = pandas_reader(
            stats_dir / f"time_diff_stats_pharma.{file_extension}"
        )

        obs_summaries_df = pandas_reader(stats_dir / f"observation_tables_stats.{file_extension}")

        pharma_summaries_df = pandas_reader(stats_dir / f"pharma_records_stats.{file_extension}")

        varref_df = pd.read_csv(var_ref_path, sep="\t")

        lower_ids_bound = i * nr_patients
        upper_ids_bound = (i + 1) * nr_patients
        patient_ids = df_general_fake[constants.PID][lower_ids_bound:upper_ids_bound].tolist()

        dfs = []
        for pid in tqdm(patient_ids):
            offset_pid = pid - (i * nr_patients)
            df_fake = get_fake_obs_data(
                pid,
                length_of_stay[offset_pid - 1],
                df_general_fake.query(f"patientid == {pid}")["admissiontime"].iloc[0],
                obs_summaries_df,
                time_diff_dists_obs,
                varref_df,
            )

            dfs.append(df_fake)

        df_obs_fake = pd.concat(dfs)
        _write_part(df_obs_fake, output_dir / "observation_tables", part=i)

        dfs = []
        for pid in tqdm(patient_ids):
            offset_pid = pid - (i * nr_patients)
            df_fake = get_fake_pharma_data(
                pid,
                length_of_stay[offset_pid - 1],
                df_general_fake.query(f"patientid == {pid}")["admissiontime"].iloc[0],
                pharma_summaries_df,
                time_diff_dists_pharma,
                varref_df,
            )

            dfs.append(df_fake)

        df_pharma_fake = pd.concat(dfs)
        _write_part(df_pharma_fake, output_dir / "pharma_records", part=i)

    logging.info(40 * "=")
    logging.info("Finished")
    logging.info(40 * "=")


if __name__ == "__main__":

    # setup logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s | %(message)s")

    main()
