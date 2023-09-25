import pickle
from pathlib import Path

import pytest

from famews.fairness_check.analysis_pipeline import LoadEventBounds, LoadPatientsDf
from famews.pipeline import PipelineState
