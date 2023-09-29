from typing import List

import numpy as np


def draw_bootstrap_sample(pid_list: List[int], sample_size=0.5, n_samples=10) -> List[List[int]]:
    """Draw n_samples bootstrap samples of size sample_size*len(pid_list) from pid_list.

    Parameters
    ----------
    pid_list : List[int]
        list of patient ids
    sample_size : float, optional
        size of sample to draw (as percentage of len(pid_list)), by default 0.5
    n_samples : int, optional
        number of samples to draw, by default 10

    Returns
    -------
    List[List[int]]
        Bootstrap samples of pids
    """
    np.random.seed(42)
    return np.random.choice(pid_list, (n_samples, int(sample_size * len(pid_list))), replace=True)
