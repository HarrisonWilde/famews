import pickle
from pathlib import Path
from typing import Dict, List, Union

import gin
import numpy as np
import tables

from famews.fairness_check.feature_importance.helper_shap_values import (
    average_shap_values_group,
    average_shap_values_multiple_models,
    average_shap_values_patient,
    compute_shap_values,
    get_feat_ranking,
    get_feat_ranking_group,
    get_feat_ranking_random_grouping,
)
from famews.fairness_check.utils.helper_groups import get_map_group_pid
from famews.pipeline import PipelineState, StatefulPipelineStage


@gin.configurable("AnalyseFeatImportanceGroup", denylist=["state"])
class AnalyseFeatImportanceGroup(StatefulPipelineStage):
    """
    Analyse Feature Importance per Group Pipeline stage
    """

    name = "Analyse Feature Importance Group"

    def __init__(
        self,
        state: PipelineState,
        num_workers: int = 1,
        ml_stage_data_path: Path = None,
        split: str = "test",
        root_model_path: Path = None,
        model_seeds: List[str] = [],
        model_type: str = "classical_ml",
        model_save_name: str = None,
        task_name: Union[int, str] = None,
        feature_names_path: Path = None,
        root_shap_values_path: Path = gin.REQUIRED,
        use_cache: int = True,
        overwrite: bool = False,
        **kwargs,
    ):
        """Compare the feature ranking (importance-based) across cohorts.

        Parameters
        ----------
        state : PipelineState
            Pipeline state
        num_workers : int, optional
            number of workers, by default 1
        ml_stage_data_path: Path, optional
            Path for ML-stage data (expecting a h5 file), necessary if need to compute SHAP values and to infer the feature names, by default None
        split: str, optional
            Split name for which we want to compute the SHAP values (usually "val" or "test"), by default test
        root_model_path: Path, optional
            Path to the directory containing the pretrained model, only necessary if SHAP values aren't pre-computed and thus can't be loaded, by default None
        model_seeds: List[str], optional
            List of seeds in case we want to study the average of multiple models, in this case the model directories are of the form {root_model_path}/seed_{model_seeds[i]} and the shap values directories {root_shap_values_path}/seed_{model_seeds[i]}, by default []
        model_type: str, optional
            Model type (supported 'classical_ml'), only necessary if SHAP values aren't pre-computed and thus can't be loaded, by default 'classical_ml'
        model_save_name: str, optional
            Name of the file where the model has been saved, expected a string of the form *.joblib, by default None
        task_name: Union[int, str], optional
            Task name (or index if int), only necessary if SHAP values aren't pre-computed and thus can't be loaded, by default None
        feature_names_path: Path, optional
            Path to pickle file containing the least of feature names (will be infered through the ml_stage data if not provided), by default None
        root_shap_values_path: Path, optional
            Path to the directory containing the pre-computed SHAP values (or path to store them if not yet computed), by default None
        use_cache : int, optional
            flag to cache analysis results and use previously cached results, by default True
        overwrite : bool, optional
            flag to overwrite analysis results that were previously cached, by default False
        """
        super().__init__(state, num_workers=num_workers, **kwargs)
        if root_model_path is not None:
            if len(model_seeds):
                self.list_model_paths = [
                    Path(root_model_path) / f"seed_{s}" / model_save_name for s in model_seeds
                ]
            else:
                self.list_model_paths = [Path(root_model_path) / model_save_name]
        else:
            self.list_model_paths = []
        self.ml_stage_data_path = (
            Path(ml_stage_data_path) if ml_stage_data_path is not None else ml_stage_data_path
        )
        self.split = split
        self.model_type = model_type
        self.feature_names_path = (
            Path(feature_names_path) if feature_names_path is not None else feature_names_path
        )

        if len(model_seeds):
            self.list_shap_values_paths = [
                Path(root_shap_values_path) / f"seed_{s}" / f"{self.split}_shap_values.pkl"
                for s in model_seeds
            ]
        else:
            self.list_shap_values_paths = [
                Path(root_shap_values_path) / f"{self.split}_shap_values.pkl"
            ]
        self.task_name = task_name
        self.use_cache = use_cache
        self.overwrite = overwrite
        self.do_bootstrap = self.state.do_stat_test
        self.cache_path = Path(self.state.fairness_log_dir)
        self.feat_ranking_path = self.cache_path / "feat_ranking.pkl"

    def runnable(self) -> bool:
        """
        Check whether the state contains patients dataframe and grouping definition
        """
        return hasattr(self.state, "patients_df") & hasattr(self.state, "groups")

    def is_done(self) -> bool:
        """Check if the current `PipelineState` contains feature rankings and in case of caching on disk check if the file has been cached

        Returns
        -------
        bool
            Whether the stage has been completed.
        """
        # check if self.state contains predictions and in case of caching on disk check if the file has been cached
        if self.use_cache:
            return (
                self.feat_ranking_path.exists()
                and hasattr(self.state, "feat_ranking_per_group")
                and self.state.feat_ranking_per_group is not None
                and hasattr(self.state, "feat_ranking_all")
                and self.state.feat_ranking_all is not None
                and (
                    not self.state.do_stat_test
                    or (
                        hasattr(self.state, "feat_ranking_random_group")
                        and self.state.feat_ranking_random_group is not None
                    )
                )
            )
        return (
            hasattr(self.state, "feat_ranking_per_group")
            and self.state.feat_ranking_per_group is not None
            and hasattr(self.state, "feat_ranking_all")
            and self.state.feat_ranking_all is not None
            and (
                not self.state_do_stat_test
                or (
                    hasattr(self.state, "feat_ranking_random_group")
                    and self.state.feat_ranking_random_group is not None
                )
            )
        )

    def run(self):
        """
        Generate the feature ranking for each cohort.
        """
        if not self.overwrite and self.feat_ranking_path.exists():
            # load previously cached feature rankings
            with open(self.feat_ranking_path, "rb") as f:
                feat_ranking_cached = pickle.load(f)
                self.state.feat_ranking_per_group = feat_ranking_cached["group"]
                self.state.feat_ranking_all = feat_ranking_cached["all"]
                if self.state.do_stat_test:
                    self.state.feat_ranking_random_group = feat_ranking_cached["random"]
            return
        if self.ml_stage_data_path and self.ml_stage_data_path.exists():
            self.data_h5 = tables.open_file(self.ml_stage_data_path, "r").root
            feature_names = np.array([f.decode("utf-8") for f in self.data_h5.data.columns[:]])
        elif self.feature_names_path and self.feature_names_path.exists():
            self.data_h5 = None
            with open(self.feature_names_path, "rb") as f:
                feature_names = np.array(pickle.load(f))
        else:
            raise ValueError(
                "Feature names can't be loaded or infered as ML-stage can't be loaded either."
            )
        shap_values = self.load_shap_values()
        avg_shap_value_patients = average_shap_values_patient(shap_values)
        shap_pids = avg_shap_value_patients.keys()
        self.state.feat_ranking_all = get_feat_ranking(
            np.mean([avg_shap_value_patients[pid] for pid in shap_pids], axis=0), feature_names
        )
        patients = self.state.patients_df.loc[shap_pids]
        self.state.feat_ranking_per_group = {}
        for group_name, cats in self.state.groups.items():
            map_group_pid = get_map_group_pid(patients, group_name, cats)
            avg_shap_group = average_shap_values_group(map_group_pid, avg_shap_value_patients)
            self.state.feat_ranking_per_group[group_name] = get_feat_ranking_group(
                avg_shap_group, feature_names
            )
        if self.state.do_stat_test:
            self.state.feat_ranking_random_group = get_feat_ranking_random_grouping(
                avg_shap_value_patients, feature_names
            )
        if self.use_cache:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            feat_ranking_to_cache = {
                "all": self.state.feat_ranking_all,
                "group": self.state.feat_ranking_per_group,
            }
            if self.state.do_stat_test:
                feat_ranking_to_cache["random"] = self.state.feat_ranking_random_group
            with open(self.feat_ranking_path, "wb") as f:
                pickle.dump(feat_ranking_to_cache, f)

    def load_shap_values(self) -> Dict[int, np.array]:
        """Load the shap values. If they are already pre-computed, just load them from disk,
        otherwise compute them

        Returns
        -------
        Dict[int, np.array]
            Shap values dictionary mapping pid to matrix of shap values

        Raises
        ------
        Exception
            The shap values haven't been pre-computed and not enough information are provided to compute them
        """
        list_shap_values = []
        for i in range(len(self.list_shap_values_paths)):
            shap_values_path = self.list_shap_values_paths[i]
            if len(self.list_model_paths) > i:
                model_path = self.list_model_paths[i]
            else:
                model_path = None
            if shap_values_path and shap_values_path.exists():
                with open(shap_values_path, "rb") as f:
                    shap_values = pickle.load(f)
            elif model_path and model_path.exists() and self.data_h5 is not None:
                shap_values = compute_shap_values(
                    self.model_type,
                    model_path,
                    self.data_h5,
                    self.task_name,
                    self.split,
                    self.state.max_len,
                )
                with open(shap_values_path, "wb") as f:
                    pickle.dump(shap_values, f)
            else:
                raise Exception(
                    "SHAP values can't be loaded and can't be computed (the model path not provided)"
                )
            list_shap_values.append(shap_values)
        return average_shap_values_multiple_models(list_shap_values)
