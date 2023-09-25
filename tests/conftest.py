# ===========================================================
#
# Preprocessing Fixtures
#
# ===========================================================
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from pytest import TempPathFactory

from famews.data.datasets import ICUVariableLengthDataset
from famews.data.hirid import reading
from famews.pipeline import PipelineState
from famews.scripts.synthetic.generate_hirid_syn_data import main as syn_main
from famews.train.sequence import SequenceWrapper
from famews.utils.handle_predictions import HandlePredictions


# ===========================================================
# HiRID
# ===========================================================
@pytest.fixture(scope="session")
def preprocessing_resource_path() -> Path:
    test_root = Path(__file__).parent
    preprocessing_res = test_root.parent / "data" / "resources"
    return preprocessing_res


@pytest.fixture(scope="session")
def ci_resource_path() -> Path:
    test_root = Path(__file__).parent
    ci_res = test_root.parent / "ci" / "resources"
    return ci_res


@pytest.fixture(scope="session")
def df_var_ref(preprocessing_resource_path: Path) -> Path:
    varref = reading.read_var_ref_table(preprocessing_resource_path / "varref.tsv")
    return varref


@pytest.fixture(scope="session")
def df_stat_ref(preprocessing_resource_path: Path) -> Path:
    statref = reading.read_stat_ref_table(preprocessing_resource_path / "stats_cews.parquet")
    return statref


@pytest.fixture(scope="session")
def varref_processed(preprocessing_resource_path: Path) -> pd.DataFrame:
    varref, _ = reading.read_reference_table(preprocessing_resource_path / "varref.tsv")
    return varref


# ---------------------
# Dataset Statistics
# ---------------------
@pytest.fixture(scope="session")
def dataset_stats_path(ci_resource_path: Path) -> Path:
    return ci_resource_path / "synthetic"


@pytest.fixture(scope="session")
def synthetic_obs_stats_path(ci_resource_path: Path) -> Path:
    return ci_resource_path / "synthetic/observation_tables_stats.csv"


@pytest.fixture(scope="session")
def synthetic_obs_time_stats_path(ci_resource_path: Path) -> Path:
    return ci_resource_path / "synthetic/time_diff_stats_obs.csv"


@pytest.fixture(scope="session")
def synthetic_pharma_stats_path(ci_resource_path: Path) -> Path:
    return ci_resource_path / "synthetic/pharma_records_stats.csv"


@pytest.fixture(scope="session")
def synthetic_pharma_time_stats_path(ci_resource_path: Path) -> Path:
    return ci_resource_path / "synthetic/time_diff_stats_pharma.csv"


@pytest.fixture(scope="session")
def generate_synthetic_data(
    tmp_path_factory: TempPathFactory,
    preprocessing_resource_path: Path,
    dataset_stats_path: Path,
) -> Path:

    num_patients = 10
    num_parts = 2

    synthetic_path = tmp_path_factory.mktemp("synthetic")

    cmd_arguments = [
        str(dataset_stats_path),
        str(synthetic_path / "data"),
        "--var-ref-path",
        str(preprocessing_resource_path / "varref.tsv"),
        "--nr-patients",
        str(num_patients),
        "--nr-parts",
        str(num_parts),
    ]

    syn_main(cmd_arguments)

    return synthetic_path


# ---------------------
# Configurations
# ---------------------
@pytest.fixture(scope="session")
def config_examples_path(ci_resource_path: Path) -> Path:
    configs_path = ci_resource_path / "config"
    return configs_path


# ---------------------
# Configurations
# ---------------------
@pytest.fixture(scope="session")
def ftt_resource_path(preprocessing_resource_path) -> Path:
    ftt_res = preprocessing_resource_path / "ftt"
    return ftt_res


@pytest.fixture(scope="session")
def tmp_log_dir(tmp_path_factory: TempPathFactory):
    log_dir = tmp_path_factory.mktemp("log_test")
    return log_dir


@pytest.fixture(scope="session")
def get_predictions(get_hirid_preprocessing_data: Path, tmp_log_dir: Path):
    horizon = 12  # set by get_hirid_preprocessing_data fixture
    data_path = get_hirid_preprocessing_data / "processed"

    # ML Stage
    data_path = (
        data_path
        / f"ml_stage/ml_stage_sampletime_ffill_only_hr_no-offset-grid_non-causal-lactate_ffill-drugs_hr-at_grid-label-left_grid-value-median_no_ambiguous_full_window_mean-tsh_{horizon}h.h5"
    )
    dataset_class = ICUVariableLengthDataset

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return torch.ones(x.shape[0], x.shape[1], 1)

    model_wrapper = SequenceWrapper(
        model=DummyNet(),
        learning_rate=0.01,
        loss=nn.functional.binary_cross_entropy_with_logits,
    )
    state = PipelineState()
    state.data_path = data_path
    state.dataset_class = dataset_class
    state.model_wrapper = model_wrapper
    state.log_dir = tmp_log_dir
    handle_pred_stage = HandlePredictions(state)
    handle_pred_stage.run()
    return Path(state.log_dir) / "predictions" / f"predictions_split_test.pkl"
