# ========================================
#
# Utilities around data handling
#
# ========================================
import gc
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence, Union

import gin
import numpy as np
import pandas as pd
import pathos
import tqdm

from famews.data.hirid import constants


# ========================================
#
# Organization / Storage
#
# ========================================
def delete_if_exist(path: str):
    """Deletes a `path` if it exists on the file-system"""
    if os.path.exists(path):
        os.remove(path)


class MarkedDataset:
    """
    A `MarkedDataset` can consist of a directory of part files following
    a regex or a single file representing a single part
    A `MarkedDataset` constisting of parts is marked as complete/done
    using an empty file called "_SUCCESS"

    Class Attributes
    ----------------
    success_file_name: str (immutable)
        file name for the empty marker file
    """

    _success_file_name: str = "_SUCCESS"

    def __init__(
        self, path: Path, part_re: re.Pattern = re.compile("part-([0-9]+).*"), force: bool = True
    ):
        """
        Constructor for `MarkedDataset`

        Parameter
        ---------
        path: str
            directory for this `MarkedDataset`
        path_re: re.Pattern
            regex pattern refering to the considered files
        force: bool
            whether we force clear the directory upon setting it up
        """
        self.path = Path(path)
        self.part_re = part_re
        self.force = force

    @property
    def success_file_name(self) -> str:
        """Getter for `success_file_name`"""
        return type(self)._success_file_name

    def mark_done(self):
        """Create the marker file"""
        if self.path.is_dir():
            (self.path / self.success_file_name).touch()

    def is_done(self) -> bool:
        """Check if the data location is marked as processed"""
        if self.path.is_dir():
            return (self.path / self.success_file_name).exists()
        return self.path.exists()

    def prepare(self, single_part: bool = False):
        """
        Prepare the data location

        Parameter
        ---------
        single_part: bool
            if True we operate on parts of a path only
        """
        if self.force and self.path.exists():
            if single_part:
                self.path.unlink()
            else:
                shutil.rmtree(self.path)

        if single_part:
            self.path.parent.mkdir(exist_ok=True, parents=True)
        else:
            self.path.mkdir(parents=True)

    def list_parts(self) -> list[Path]:
        """Return the list files belonging to this `MarkedDataset`"""

        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist")

        if not self.path.is_dir():
            return [self.path]

        parts = [f for f in self.path.iterdir() if f.is_file() and self.part_re.match(f.name)]

        def sorter(f):
            regex_match = self.part_re.match(f.name)
            if regex_match is None:
                raise ValueError(f"Regex {f.name} did not match.")
            else:
                return int(regex_match.groups()[0])

        parts_sorted = sorted(parts, key=lambda f: sorter(f))
        return parts_sorted


# ================================================================================
#
# Numpy / Pandas
#
# ================================================================================
def value_empty(
    size: Union[int, tuple[int, ...]], default_val: float, dtype: np.dtype = None
) -> np.ndarray:
    """
    Returns a vector filled with elements of a specific value

    Parameter
    ---------
    size: Union[int, tuple[int, ...]]
        size of the vector
    default_val: float
        the value to fill the vector with
    dtype: np.dtype
        numpy datatype to use for the vector
    """
    if dtype is not None:
        tmp_arr = np.empty(size, dtype=dtype)
    else:
        tmp_arr = np.empty(size)

    tmp_arr[:] = default_val
    return tmp_arr


def empty_nan(sz: Union[int, tuple[int, ...]]) -> np.ndarray:
    """
    Returns an empty NAN vector of specified size

    Parameter
    ---------
    sz: Union[int, tuple[int, ...]]
        size/shape of the returned vector
    """
    arr = np.empty(sz)
    arr[:] = np.nan
    return arr


def is_df_sorted(df: pd.DataFrame, colname: str) -> bool:
    """
    Check whether the dataframe is sorted w.r.t. to `colname`

    Parameter
    ---------
    df: pd.DataFrame
        dataframe to check
    colname: str
        column to check for sorting
    """
    return bool((np.array(df[colname].diff().dropna(), dtype=np.float64) >= 0).all())


# ========================================
#
# Processing
#
# ========================================
def map_and_combine_patient_dfs(
    map_pat_fns: Sequence[Callable],
    part_lists: list[list[Path]],
    reading_fns: Sequence[Callable],
    combine_fn: Callable,
    writing_fn: Callable,
    workers: int,
) -> None:
    """
    Parallel processing to map and combine patient files

    Parameter
    ---------
    map_pat_fns: Sequence[Callable]
        mapping functions to apply to the data
    part_lists: list[str]
        list of individual files to process
    reading_fns: Sequence[Callable]
        reading functions to read from the files
    combine_fn: Callable
        combine function after processing
    writing_fn: Callable
        write function for results
    workers: int
        number of parallel workers launched
    """

    def _process_parts(paths_same_part: Sequence[Path]):
        assert len(paths_same_part) == len(reading_fns) == len(map_pat_fns)

        part_name = paths_same_part[0].name
        dfs_mapped = []
        for (path, read_fn, map_fn) in zip(paths_same_part, reading_fns, map_pat_fns):
            df = read_fn(path)

            df_mapped = {pid: map_fn(df_pat) for (pid, df_pat) in df.groupby(constants.PID)}
            dfs_mapped.append(df_mapped)

        df_ret = combine_fn(dfs_mapped)
        writing_fn(df_ret, part_name)

    all_paths_same_part = zip(*part_lists)
    exec_parallel_on_parts(_process_parts, all_paths_same_part, workers)


def map_patient_df(
    map_pat_fn: Callable,
    part_list: list[Path],
    reading_fn: Callable,
    writing_fn: Callable,
    workers: int,
) -> None:
    """
    Map a patient dataframe: read->map->write

    Parameter
    ---------
    map_pat_fn: Callable
        mapping function (transformation)
    part_list: list[str]
        list of files to apply to
    reading_fn: Callable
        input I/O function
    writing_fn: Callable
        output I/O function
    workers: int
        parallel workers
    """
    map_df(
        lambda df: df.groupby(constants.PID).apply(map_pat_fn),
        part_list,
        reading_fn,
        writing_fn,
        workers,
    )


def map_df(
    map_fn: Callable,
    part_list: list[Path],
    reading_fn: Callable,
    writing_fn: Callable,
    workers: int,
    verbose: bool = False,
) -> None:
    """
    Map a dataframe in parallel: read->map->write

    Parameter
    ---------
    map_pat_fn: Callable
        mapping function (transformation)
    part_list: list[str]
        list of files to apply to
    reading_fn: Callable
        input I/O function
    writing_fn: Callable
        output I/O function
    workers: int
        parallel workers
    verbose: bool
        verbose output
    """

    def _process_part(path: Path):
        # we are doing the file I/O in the subprocess, not in the main process in order to avoid having to transfer
        # the data frame in memory from the main process to the subprocess
        df = reading_fn(path)
        if verbose:
            logging.info(f"[MAP] read {path}")

        # transform the data
        df_ret = map_fn(df)
        if verbose:
            logging.info(f"[MAP] processed {path}")

        # write out results
        part = path.name
        writing_fn(df_ret, part)
        if verbose:
            logging.info(f"[MAP] wrote {path}")

        gc.collect()

    exec_parallel_on_parts(_process_part, part_list, workers)


def map_reduce_patient_df(
    map_pat_fn: Callable,
    part_list: list[str],
    reading_fn: Callable,
    reduce_fn: Callable,
    workers: int,
):
    """
    Read data from files, transform, apply a reduction function

    Parameter
    ---------
    map_pat_fn: Callable
        transformation function
    part_list: list[str]
        list of files to read and process
    reading_fn: Callable
        input I/O function
    reduce_fn: Callable
        reduction function to apply
    workers: int
        number of parallel workers
    """

    def _process_part(path: Path):
        df = reading_fn(path)
        return reduce_fn(map_pat_fn(df_pat) for (_, df_pat) in df.groupby(constants.PID))

    return exec_parallel_on_parts(_process_part, part_list, workers)


def exec_parallel_on_parts(fnc: Callable, part_list: Iterable, workers: int) -> list:
    """
    Apply a function in parallel across a list of input files

    Parameter
    ---------
    fnc: Callable
        function to be applied to the input files (in parallel)
    part_list: Sequence
        list of input files (paths)
    workers: int
        number of parallel worker processes to launch

    Return
    ------
    list
        list of processed files
    """
    if workers > 1:
        # using pathos as it uses a more robust serialization than the default multiprocessing
        with pathos.multiprocessing.Pool(workers) as pool:
            return list(tqdm.tqdm(pool.imap(fnc, part_list)))
    else:
        return [fnc(part) for part in tqdm.tqdm(part_list)]


@gin.configurable()
def q_exp_param(dt, h_true, h_min, h_max, delta_h=1, gamma=0.01):
    """Exponential smoothing function.

    Args:
        dt: (int) distance steps to next event (+inf if no event).
        h_true: (int) true horizon of prediction in steps. (not used)
        h_min: (int) minimal horizon of smoothing in steps.
        h_max: (int) maximum horizon of smoothing in steps.
        delta_h: (float) number of step per hour.
        gamma: (float) positive smoothing strength parameter.

    Returns:
        q^exp(1|t) as a float
    """
    if dt <= h_min:
        return 1
    elif dt > h_max:
        return 0
    else:
        h_min_scaled = h_min / delta_h
        h_max_scaled = h_max / delta_h
        dt_scaled = dt / delta_h

        d = -(1 / gamma) * np.log(np.exp(-gamma * (h_min_scaled)) - np.exp(-gamma * (h_max_scaled)))
        A = -np.exp(-gamma * (h_max_scaled - d))
        return np.exp(-gamma * (dt_scaled - d)) + A


@gin.configurable()
def q_kro(dt, h_true, h_min, h_max, delta_h=12, gamma=0.1):
    """Simple step function for reproduction of non-smoothed case."""
    if dt <= h_true:
        return 1
    else:
        return 0


@gin.configurable()
def q_sigmoid_param(dt, h_true, h_min, h_max, delta_h=1, gamma=0.1):
    """Sigmoidal smoothing function.

    Args:
        dt: (int) distance steps to next event (+inf if no event).
        h_true: (int) true horizon of prediction in steps.
        h_min: (int) minimal horizon of smoothing in steps.
        h_max: (int) maximum horizon of smoothing in steps.
        delta_h: (float) number of step per hour.
        gamma: (float) positive smoothing strength parameter.

    Returns:
        q^sigmoid(1|t) as a float
    """
    if dt <= h_min:
        return 1
    elif dt > h_max:
        return 0
    else:
        beta = 1 / gamma
        h_min_scaled = h_min / delta_h
        h_max_scaled = h_max / delta_h
        h_true_scaled = h_true / delta_h

        dt_scaled = dt / delta_h

        norm = np.exp(beta * (h_max_scaled))
        if norm == np.inf:  # Overflow
            return q_kro(dt, h_true, h_min, h_max)

        # In the general case d != h and depends on h_min and h_max in the following way.
        D_1 = np.exp(beta * (h_min_scaled)) - np.exp(beta * (h_max_scaled))
        D_2 = np.exp(beta * (h_true_scaled)) - np.exp(beta * (h_max_scaled))
        D_1 /= np.exp(beta * (h_max_scaled))  # Scale for overflow
        D_2 /= np.exp(beta * (h_max_scaled))  # Scale for overflow
        n = -D_1 / 2 + D_2
        m = D_1 * np.exp(beta * (h_true_scaled)) / 2 - D_2 * np.exp(beta * (h_min_scaled))
        Q = n / m
        d = -(1 / beta) * np.log(Q)

        # Lower asymptote
        A = (np.exp(beta * (h_min_scaled - d)) + 1) / (
            (np.exp(beta * (h_min_scaled - d)) + 1) - (np.exp(beta * (h_max_scaled - d)) + 1)
        )

        # Capacity
        K = -A * np.exp(beta * (h_max_scaled - d))

        return (K - A) / (1 + np.exp(beta * (dt_scaled - d))) + A


# TODO add test for the wrapper
@gin.configurable("get_smoothed_labels")
def get_smoothed_labels(
    label,
    event,
    smoothing_fn=gin.REQUIRED,
    h_true=gin.REQUIRED,
    h_min=gin.REQUIRED,
    h_max=gin.REQUIRED,
    delta_h=12,
    gamma=0.1,
):
    """

    Label smoothing wrapper over a patient stay.

    Parameter
    ---------
    label: np.array
        array with non-smoothed EEP labels.
    event: np.array
        array with the event labels.
    smoothing_fn: function
        The smoothing function to apply to the labels
    h_true: int
        The true horizon at which the event is happening in steps.
    h_min: int
        The minimum horizon at which we apply the smoothing function in steps.
    h_max: integer
        The maximum horizon at which we apply the smoothing function in steps.
    delta_h: int
        the number of steps per hour.
    gamma: float
        The strength of the smoothing.
    """

    diffs = np.concatenate(
        [event[:1] == 1, (event[1:] == 1) & (event[:-1] != 1)], axis=-1
    )  # any change to an event
    pos_event_change_full = np.where(diffs)[0]
    label_for_event = label
    h_for_event = h_true

    last_know_idx = np.where(label_for_event != -1)[0][-1]
    last_know_label = label_for_event[last_know_idx]

    # Need to handle the case where the ts was truncatenated and there was an event at the end
    if ((last_know_label == 1) and (len(pos_event_change_full) == 0)) or (
        (last_know_label == 1) and (last_know_idx >= pos_event_change_full[-1])
    ):
        last_know_event = 0
        if len(pos_event_change_full) > 0:
            last_know_event = pos_event_change_full[-1]

        last_known_stable = 0
        known_stable = np.where(label_for_event == 0)[0]
        if len(known_stable) > 0:
            last_known_stable = known_stable[-1]

        diffs_label = np.concatenate(
            [np.zeros(1), label_for_event[1:] - label_for_event[:-1]], axis=-1
        )
        pos_change = np.where((diffs_label >= 1) & (label_for_event == 1))[0]
        last_pos_change = pos_change[
            np.where(pos_change > max(last_know_event, last_known_stable))
        ][0]
        pos_event_change_full = np.concatenate(
            [pos_event_change_full, [last_pos_change + h_for_event]]
        )

    # No event case
    if len(pos_event_change_full) == 0:
        pos_event_change_full = np.array([np.inf])

    time_array = np.arange(len(label))
    dist = pos_event_change_full.reshape(-1, 1) - time_array
    dte = np.where(dist > 0, dist, np.inf).min(axis=0)
    return np.array(
        list(
            map(
                lambda x: smoothing_fn(
                    x, h_true=h_true, h_min=h_min, h_max=h_max, delta_h=delta_h, gamma=gamma
                ),
                dte,
            )
        )
    )
