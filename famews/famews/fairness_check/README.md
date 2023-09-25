# Fairness Analysis

## `FairnessAnalysisPipeline`

The pipeline can be run with the following command:

```
python -m famews.scripts.run_fairness_analysis \
       -g ./config/lgbm_base_fairness.gin \
       -l ./logs/lgbm_base \
       --seed 1111 \
```

### Concepts used
- Group name: refer to the attribute used to form the subcohorts of patients, for example sex or admission type
- Category: refer to the value taken by this attribute, it characterizes a specific subcohort, for example F (female) or elective
Please note that in the documentation we can used the word "cohort", "subcohort" and "category" equally, to refer to a set of patients sharing a common attribute value.
The word "grouping" is commonly used to refer to a way of categorizing patients, i.e. to the fact of conditioning on a specific attribute.

### Prerequisites to run the pipeline

- Store predictions on the test set by running the training pipeline with as additional stage `HandlePredictions` (an example GIN config is given in `config/lgbm_base_pred.gin`)
- Define which groupings and cohorts of patients we want to study in a YAML file (an example is given for **HiRID** in `config/group_hirid_complete.yaml` and for **MIMIC** in `config/group_mimic_complete.yaml`)
- Generate table with patient's demographics (this can be done directly in the `LoadPatientsDf` stage)
- Generate event bounds from endpoints (this can be done directly in the `LoadEventBounds` stage)
- Generate mapping from split name (train, val or test) to the corresponding list of patient ids (this can be done directly in the `LoadSplitPids` stage - one can pass a path to a pickle file containing the splits under the format `{split_name: np.array(list_pids)}` with `split_name` taking values in `train`, `val`, `test` and `list_pids` the list of patients ids.)
- in GIN config:
    - `use_multiple_predictions` flag to run the analysis of average of predictions by models trained with different random seeds. The root directory where the predictions are stored and the list of seeds need to be passed in GIN config as attribute of the `HandleMultiplePredictions` stage.
    - `threshold_score` threshold value on score **or** `name_threshold` to define a threshold from a target metric value (e.g. `event_recall_0.8` means that the threshold on score will be defined such that we have a global event-based recall at 0.8)
    - `do_stat_test` flag to run statistical tests for the different analysis
    - `significance_level` value of significance level for statistical test (before correction)
    - `horizon` prediction horizon in hour (by default 12 hours)
    - `max_len` maximal length of time series to consider per patient (on {timestep} min grid)
    - `timestep` Timestep of time series in minutes (by default 5)
- To run a particular analysis stage, it has to be added in the `FairnessAnalysisPipeline.analysis_stages` list in the GIN config
- To run the analysis stage on merged events, set `LoadEventBounds.merge_time_min` to the desired value in minutes.

### Analysis stages

The results of the analysis will be stored in `./logs/lgbm_base/fairness_check/{str_stat_test}/{name_threshold}/{merge_str}` with:
- `str_stat_test` being `no_stat_test` if the pipeline is run with `do_stat_test=False` or `w_stat_test` if the pipeline is run with `do_stat_test=True`
- `name_threshold` the string given in the config if any, otherwise `threshold_{threshold_score}`
- if `merge_time_min` is given in the config then `merge_str` is `merge_event_{merge_time_min}min`, else the path ends at the `name_threshold` folder

- `AnalysePerformanceGroup` computes model metrics for each cohort of patients

    In GIN config: 
    - `use_cache` flag to store analysis output
    - `overwrite` flag to overwrite analysis output that have already been stored
    - `store_curve` flag to enable curve generation (e.g. calibration) 
    - `nbins_calibration` number of bins to draw the calibration curve
    - `metrics_binary_to_check` List of binary metrics to measure, it has to be a subset of `["recall", "precision", "npv", "fpr", "corrected_precision", "corrected_npv", "event_recall"]`
    - `metrics_score_to_check` List of score-based metrics to measure, it has to be a subset of `["positive_class_score", "negative_class_score", "auroc", "auprc", "corrected_auprc", "event_auprc", "corrected_event_auprc", "calibration_error"]`

- `AnalyseTimeGapGroup` computes the median time gap between the first correct alarm and the start of the corresponding event, compares the difference of medians across cohorts of patients 

    In GIN config:
    - `use_cache` flag to store analysis output
    - `overwrite` flag to overwrite analysis output that have already been stored
    - `groups_start_event` list to define bounds of event groups based on the length of available prediction horizon. 
    For instance, to specify the following groups [0-3h, 3-6h, 6-12h, >12h], provide a list [0, 3, 6, 12]. Values in a list should be given in hours and sorted in increasing order without duplicate.
    We need to group events based on the duration between the end of the last event (or the start of stay) and the start of the event. This duration affects the range of possible values for the time gap between the alarm and the event. Some events have less available time in front of them than others, for instance, we can not predict 8 hours in advance for the event that happened 2 hours after the admission.

- `AnalyseMedVarsGroup` computes the median values of specified medical variables for each cohort of patients, based on the training data. 
    For each cohort, the median value is computed for three conditions: all patients across the entire stay, for all patients but only while not in an event, and only for patients not experiencing any event.

    In GIN config:
    - `use_cache` flag to store analysis output
    - `overwrite` flag to overwrite analysis output that have already been stored
    - `path_common_stage_dir` path to the directory that contains the `common_stage` preprocessed data (resampled on {timestep}-time grid before imputation, see more details in the HiRID benchmark repo: https://github.com/ratschlab/HIRID-ICU-Benchmark) 
    - `patients_source` string that specifies the data source (for the moment only working for `hirid` - we expect that `path_common_stage_dir` is a directory containing `.parquet` files containing time series data for batch of patients.)
    - `classical_vars` list of medical variables to study (same naming as in `common_stage` tables) 
    - `conditioned_vars` list of medical variables to study but only when they respect a certain condition on other medical variables 
    - `specified_conditions` object specifying conditions for the variables in `conditioned_vars`
    - `vars_units` dictionary mapping each medical variable to its unit, this is optional and will only be used for display purpose

    **Example of how to declare `conditioned_vars`**: We want to study the Mean Arterial Pressure when patients aren't under vasopressors (as they influence the MAP)
    `conditioned_vars={'ABPm': ('has_vasop', False)}` means that we compute the median ABPm when the condition `has_vasop` is False. Then the condition `has_vasop` needs to be defined in `specified_conditions`.

    `specified_conditions = {'has_vasop': {'vasop1': ('val>0', 'val==0'), 'vasop2': ('val>0', 'val==0')}}` means that `has_vasop` is True for all time points where the value of `vasop1` or the value of `vasop2` is greater than 0.

- `AnalyseFeatImportanceGroup` compares the feature ranking (per importance according to SHAP values) for each cohort of patients.

    In GIN config:
    - `use_cache` flag to store analysis output
    - `overwrite` flag to overwrite analysis output that have already been stored
    - `ml_stage_data_path` path to the ML-stage data (.h5 file) (contains input to the model)
    - `split` name of the data split for which we want to compute the SHAP values (either `test` or `val`)
    - `task_name` name (or index) of the task in the ML-stage dataset
    - `model_seeds` list of seeds in case of multiple models (it corresponds to the suffix of the model folder of the form `seed_{model_seeds[i]}`)
    - `root_model_path` path to the directory containing the trained models, in case of multiple models in contained the `seed_{model_seeds[i]}` folders
    - `model_type` string representing the type of model (for the moment only `classical_ml`is supported, by `classical_ml` we mean all non-Deep Learning models that are supported by `shap.Explainer`: https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html)
    - `model_save_name` name under which the model is saved (for `classical_ml` model type, expected name finishing with `.jobllib`)
    - `feature_names_path` path to a pickle file containing the list of feature names (in case the `ml_stage_data_path` hasn't been provided and thus the feature names cannot be infered)
    - `root_shap_values_path` path to the directory containing the SHAP values (or where to save them), in case of multiple models in contained the `seed_{model_seeds[i]}` folders

    It is possible to use this stage with pre-computed SHAP values and without access to the ML-stage data. In this case, the `root_shap_values_path` and `feature_names_path` need to be passed, and in case of multiple models the `model_seeds` as well.
    If the SHAP values have not yet been computed then the `ml_stage_data_path`, the `task_name` as well as all the information related to the trained models have to be passed. 
    **Please note that the computation of SHAP values within the stage isn't supported for DL model. If you want to use the stage for a DL model you will need to pass the pre-computed SHAP values directly as specified above.**

- `AnalyseMissingnessGroup` compares the intensity of measurements for each cohort of patients and assess the impact of missingness in model performance.

    In GIN config:
    - `use_cache` flag to store analysis output
    - `overwrite` flag to overwrite analysis output that have already been stored
    - `path_common_stage_dir` path to the directory that contains the `common_stage` preprocessed data (resampled on {timestep}-time grid before imputation, see more details in the HiRID benchmark repo: https://github.com/ratschlab/HIRID-ICU-Benchmark) 
    - `patients_source` string that specifies the data source (for the moment only working for `hirid` - we expect that `path_common_stage_dir` is a directory containing `.parquet` files containing time series data for batch of patients.)
    - `medical_vars` dictionary of medical variables to study (same naming as in `common_stage` tables) with their corresponding expected sampling intervals (in minutes)
    - `vars_only_patient_msrt` list of medical variables for which we want only to study predictions on patients with measurement (subset of medical_vars keys), if a variable is used in the label definition it needs to be appended to this list.
    - `category_intensity` Dictionary defining the categories of intensity of measurement of the following format `{intensity_category: (lower, upper)}` (with lower excluded and upper included). Categories have to be in increasing order.
    - `metrics_binary_to_check` List of binary metrics to measure (no event-based metrics should be present), it has to be a subset of `["recall", "precision", "npv", "fpr", "corrected_precision",  "corrected_npv"]`
    - `metrics_score_to_check` List of score-based metrics to measure (no event-based metrics should be present), it has to be a subset of `["positive_class_score", "negative_class_score", "auroc", "auprc", "corrected_auprc"]`

- `CreatePDFReport` generates PDF report with the results of each of the previous analysis

    In GIN config:
    - `colors` list of colors for the figure creation
    - `display_stat_test_table` whether to display statistical test tables in report
    - `figsize_*` size for the different figures to display in report
    - `k_feat_importance` number of feature to display for the `AnalyseFeatImportanceGroup` stage report
    - `max_cols_topkfeat_table`number of adjacent tables to render for the `AnalyseFeatImportanceGroup` stage report
