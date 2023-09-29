
# FAMEWS: a Fairness Auditing tool for Medical Early-Warning Systems

![FAMEWS Workflow](./data/figures/summary_tool_paper.png)

**FAMEWS** has primarily been designed to run on the HiRID dataset. However, it is possible to give already processed input to some stages in order to run it on other datasets that differ in their format.  
We also encourage users to add functionalities to the tool in order to expand the range of compatible datasets.  
This tool has been created to audit Early-Warning Systems in the medical domain. As such we consider a set of patients with a time-series of input features and a time-series of labels.   
As we focus on early warning, we expect a label for a current time step to be positive when a targeted event occurs a certain amount (called the prediction horizon) of time in the future. While the patient is undergoing an event, we expect the label to be NaN.  

For additional explanations on the tool, please refer to our paper: *FAMEWS: a Fairness Auditing tool for Medical Early-Warning Systems*.  
We provide a sample fairness audit report (`sample_fairness_report.pdf`) that can be produced with FAMEWS. The instructions to reproduce the report are given in the section **Pipeline Overview - How to run FAMEWS on HiRID?** of this README below the header **[TO RUN TO REPRODUCE SAMPLE REPORT]** (there are three steps: HiRID preprocessing, model inference and fairness analysis).

After explaining how to set up FAMEWS, we will first describe how to run it on the HiRID dataset and then we will give a more detailed documentation that extends its range of applications.

## Setup

This repository depends on the work done by [Yèche et al. HiRID Benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark)
to preprocess the HiRID dataset and get it ready for model training, as well as inference and fairness analysis.

The [HiRID Benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark) repository with the preprocessing is included as a submodule in this repository. To clone the repository with the submodule, run:

```bash
git submodule init
git submodule update

# follow instructions in the `HiRID Benchmark` repository to download and preprocess the dataset
# the subsequent steps rely on the different stage outputs defined by Yèche et al.
```

Then please follow the instructions of the HiRID Benchmark repository to obtain preprocessed data in a suitable format.

### Conda Environment

A conda environment configuration is provided: `environment_linux.yml`. You can create 
the environment with:
```
conda env create -f environment_linux.yml
conda activate famews
```

### Code Package

The `famews` package contains the relevant code components
for the pipeline. Install the package into your environment
with:
```
pip install -e ./famews
```

### Configurations

We use [Gin Configurations](https://github.com/google/gin-config/tags) to configure the
machine learning pipelines, preprocessing, and evaluation pipelines. Example configurations are in `./config`.  
**Please note that some paths need to be completed in these configs based on where the preprocessing outputs have been saved.
To facilitate this step, they are all gathered under `# Paths preprocessed data` or `# Data parameter`.**

## Pipeline Overview - How to run FAMEWS on HiRID?

Any task (preprocessing, training, evaluation, fairness analysis) is to be run with a script located in
`famews/scripts`. Ideally, these scripts invoke a `Pipeline` object, which consists of different
`PipelineStage` objects.

### Preprocessing
 
#### HiRID
>**[TO RUN TO REPRODUCE SAMPLE REPORT]**  
>This repository depends on the work done by [Yèche et al. HiRID Benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark)
>to preprocess the HiRID dataset and get it ready for model training, as well as inference and fairness analysis.

### ML Training

To facilitate experimentation, we provide model weights in `./data/models`.

#### LGBM model
To train an LGBM model, an example GIN config is available at `./config/lgbm_base_train.gin`.
Training can be performed with the following command:
```
python -m famews.scripts.train_tabular_model \
    -g ./config/lgbm_base_train.gin \
    -l ./logs/lgbm_base \
    --seed 1111
```

Pre-trained weights are available at `./data/models/lgbm` and can be used with the following command:
```
python -m famews.scripts.train_tabular_model \
    -g ./config/lgbm_base_pred.gin \
    -l ./logs/lgbm_base \
    --seed 1111
```
Note that these runs will also store in the log directory the predictions obtained on the test set.

You can launch several training with the `submit_wrapper.py` script. We encourage to do so to obtain model predictions from different random seeds (see config at `./config/lgbm_10seeds.yaml`).
The following command can be run:
```
python -m famews.scripts.submit_wrapper \
       --config ./config/lgbm_10seeds_train.yaml \
       -d ./logs/lgbm_10seeds
```

>**[TO RUN TO REPRODUCE SAMPLE REPORT]**  
>We also provide pre-trained weights for the LGBM models trained with 10 different random seeds in `./data/models/lgbm_10seeds`.
>To generate the predictions from each of these models, one can launch the `submit_wrapper_pred_models.py` script with the following command:
>```
>python -m famews.scripts.submit_wrapper_pred_models \
>       --config ./config/lgbm_10seeds_pred.yaml \
>       -d ./logs/lgbm_10seeds
>```

#### LSTM model
To train an LSTM model, an example GIN config is available at `./config/lstm_base_train.gin`.
Training can be performed with the following command:
```
python -m famews.scripts.train_sequence_model \
    -g ./config/lstm_base_train.gin \
    -l ./logs/lstm_base \
    --seed 1111
```

Pre-trained weights are available at `./data/models/lstm` and can be used with the following command:
```
python -m famews.scripts.train_sequence_model \
    -g ./config/lstm_base_pred.gin \
    -l ./logs/lstm_base \
    --seed 1111
```
Note that these runs will also store in the log directory the predictions obtained on the test set.

### Fairness analysis
To audit the fairness of a model, we first need to obtain its predictions on the test set (see above commands) and to obtain certain preprocessed data (see Preprocessing section).  
The following commands can be used to run a basic configuration of the fairness analysis on the HiRID dataset based on our example models. 
We give more details afterwards on how to construct such configurations for different use-cases.  
#### LGBM model
To audit an LGBM model, an example GIN config is available at `./config/lgbm_base_fairness.gin` and the following command can be run:
```
python -m famews.scripts.run_fairness_analysis \
    -g ./config/lgbm_base_fairness.gin \
    -l ./logs/lgbm_base/seed_1111 \
    --seed 1111
```
>**[TO RUN TO REPRODUCE SAMPLE REPORT]**  
>We encourage users to audit an averaged model obtained from models trained on different random seeds, an example GIN config is available at `./config/lgbm_10seeds_fairness.gin` and the following command can be run:
>```
>python -m famews.scripts.run_fairness_analysis \
>    -g ./config/lgbm_10seeds_fairness.gin \
>    -l ./logs/lgbm_10seeds \
>    --seed 1111
>```

#### LSTM model
To audit an LSTM model, an example GIN config is available at `./config/lstm_base_fairness.gin` and the following command can be run:
```
python -m famews.scripts.run_fairness_analysis \
    -g ./config/lstm_base_fairness.gin \
    -l ./logs/lstm_base/seed_1111 \
    --seed 1111
```
Please note that for this audit we don't run the `AnalyseFeatImportanceGroup` stage as it requires computing the SHAP values and this isn't supported for the DL learning model.  
However, if you still want to run this stage you can directly provide the SHAP values as input to the pipeline (see `./famews/famews/fairness_check/README.md` for more details).

## Detailed documentation on the `FairnessAnalysisPipeline`

### Concepts used
- Group name: refers to the attribute used to form the subcohorts of patients, for example, sex or admission reason
- Category: refers to the value taken by this attribute, it characterizes a specific subcohort, for example, F (female) or Cardiovascular
Please note that in the documentation we can use the word "cohort" and "category" equally, to refer to a set of patients sharing a common attribute value.
The word "grouping" is commonly used to refer to a way of categorizing patients, i.e. to the fact of conditioning on a specific attribute.

### Prerequisites to run the pipeline
These are the general preliminary steps that are required before running the `FairnessAnalysisPipeline`.
#### Obtain predictions on the test set
*On HiRID:*
    Store predictions on the test set by running the training pipeline with an additional stage `HandlePredictions` (example GIN configs are given in `./config`)
*On other data source:*
    Save your predictions in a pickle file as a dictionary: `{patient_id: (predictions, labels)}`. 
#### Define your groupings
*For all data sources*
    Define which groupings and cohorts of patients you want to study in a YAML file (an example is given for **HiRID** in `config/group_hirid_complete.yaml` and for **MIMIC** in `config/group_mimic_complete.yaml`)
#### Generate a dataframe with patient's demographics
*On HiRID:*
    This is done directly in the `LoadPatientsDf` stage (this is already run by the `FairnessAnalysisPipeline` no need for a preliminary step)
*On MIMIC III:*
    This is done directly in the `LoadPatientsDf` stage for the grouping's attributes listed in `config/group_mimic_complete.yaml` (this is already run by the `FairnessAnalysisPipeline` no need for a preliminary step)
*On other data source:*
    Construct a .csv indexed by patient id (row) and having as columns the different attributes used in your groupings.
#### Generate a dictionary containing the target event boundaries
*On HiRID:*
    This can be done directly in the `LoadEventBounds` stage (this is already run by the `FairnessAnalysisPipeline` no need for a preliminary step)
*On other data source:*
    For our primary use-case, the events to predict have a certain length. For each patient, we want to know the period of the stay for which he/she was undergoing an event. This needs to be saved in a pickle file  as a dictionary `{patient_id: [start_event, stop_event]}`. `start_event` and `stop_event` are indexed on an evenly spaced time grid (for HiRID it is a 5-minute time grid).
#### Obtain patients split (train, validation, test)
*On HiRID:*
    This can be done directly in the `LoadSplitPids` stage (this is already run by the `FairnessAnalysisPipeline` no need for a preliminary step)
*On other data source:*
    This needs to be saved in a pickle file  as a dictionary `{split_name: np.array(list_pids)}` with `split_name` taking values in `train`, `val`, `test` and `list_pids` the corresponding list of patient ids.

### Construct the GIN config
In the GIN config, as attributes of `FairnessAnalysisPipeline`:
- `use_multiple_predictions` flag to run the analysis of the average of predictions by models trained with different random seeds. The root directory where the predictions are stored and the list of seeds need to be passed in GIN config as an attribute of the `HandleMultiplePredictions` stage.
    If `use_multiple_predictions = False` then the pickle file containing the predictions needs to be stored in the log directory as `predictions/predictions_split_{split_name}.pkl`. 
- `threshold_score` threshold value on score **or** `name_threshold` to define a threshold from a target metric value (e.g. `event_recall_0.8` means that the threshold on score will be defined such that we have a global event-based recall at 0.8)
- `do_stat_test` flag to run statistical tests for the different analysis
- `significance_level` value of significance level for statistical test (before correction)
- `horizon` prediction horizon in hours (by default 12 hours)
- `max_len` maximal length of time series to consider per patient (on {timestep} min grid)
- `timestep` timestep of time series in minutes (by default 5)
- To run a particular analysis stage, it has to be added to the `FairnessAnalysisPipeline.analysis_stages` list in the GIN config

### Set-up stages

- `SetUpGroupDef` loads the grouping definitions
    In GIN config:
        - `group_config_file` path to the YAML file containing the grouping definition

- `LoadPatientsDf` loads or generates the dataframe with the patient's demographics
    If it has been pre-computed, you just need to give in the gin config the path at `LoadPatientsDf.patients_df_file`
    Otherwise, in the GIN config:
        - `patients_df_file`: path to store the generated dataframe
        - `patients_source`: data source, currently supported `hirid` or `mimic`
        - `general_table_hirid_path`: if the data source is `hirid`, path to the extended general table (generated by HiRID preprocessing)
        - `root_mimic`: if the data source is `mimic`, the path to the directory containing all MIMIC-III dataframes
        - `pid_name` string referring to the patient id, for HiRID it is `patientid` and for MIMIC it is `PatientID`

- `LoadEventBounds` load or generate the target event boundaries
    If it has been pre-computed, you just need to give in the gin config the path at `LoadEventBounds.event_bounds_file`
    Otherwise, in the GIN config:
        - `patients_df_file`: path to store the  target event boundaries
        - `patients_source`: data source, currently supported `hirid` 
        - `patient_endpoint_data_dir`: directory containing endpoints data (generated by HiRID preprocessing)
        - `event_column`: string referring to the event column name in the endpoint dataframes
    In both cases, one can merge events that are relatively close (less than `merge_time_min`) by setting `LoadEventBounds.merge_time_min` to the desired value in minutes.

- `LoadSplitPids` loads or generates patients splits (train, validation, test) dictionary
    If it has been pre-computed, you just need to give in the gin config the path at `LoadSplitPids.split_file`
    Otherwise, in the GIN config:
        - `split_file`: path to store the patients splits dictionary
        - `data_table_file`: path of h5 file containing the split definition, in the same format as the ML-stage preprocessed data of HiRID

- `HandlePredictions` loads the predictions that are stored at `{log_dir}/predictions/predictions_split_{split_name}.pkl`
    One can choose the split to load passing its name to `HandlePredictions.split` in the GIN config. We advise for the test set.
    This stage is run if `FairnessAnalysisPipeline.use_multiple_predictions=False`
    
- `HandleMultiplePredictions` loads multiple predictions and averages them. The subsequent analysis will be run on the averaged predictions.
    One can choose the split to load passing its name to `HandleMultiplePredictions.split` in the GIN config. We advise for the test set.
    Then, in the GIN config:
    - `root_predictions_path`: a path of the root directory containing the different predictions
    - `list_seeds`: list of seeds to construct the different prediction paths 
    Each set of predictions has to be stored in a pickle file whose path is of the form `root_predictions_path/seed_{list_seeds[i]}/predictions/predictions_split_{split_name}.pkl`.
    This stage is run if `FairnessAnalysisPipeline.use_multiple_predictions=True`.

### Analysis stages
We will now explain how to configure each of the analysis stages

- `AnalysePerformanceGroup` computes model metrics for each cohort of patients  
    In GIN config: 
    - `use_cache` flag to store analysis output
    - `overwrite` flag to overwrite analysis output that has already been stored
    - `store_curve` flag to enable curve generation (e.g. calibration) 
    - `nbins_calibration` number of bins to draw the calibration curve
    - `metrics_binary_to_check` List of binary metrics to measure, it has to be a subset of `["recall", "precision", "npv", "fpr", "corrected_precision", "corrected_npv", "event_recall"]`
    - `metrics_score_to_check` List of score-based metrics to measure, it has to be a subset of `["positive_class_score", "negative_class_score", "auroc", "auprc", "corrected_auprc", "event_auprc", "corrected_event_auprc", "calibration_error"]`

- `AnalyseTimeGapGroup` computes the median time gap between the first correct alarm and the start of the corresponding event, compares the difference of medians across cohorts of patients  
    In GIN config:
    - `use_cache` flag to store analysis output
    - `overwrite` flag to overwrite analysis output that has already been stored
    - `groups_start_event` list to define the bounds of event groups based on the length of the available prediction horizon. 
    For instance, to specify the following groups [0-3h, 3-6h, 6-12h, >12h], provide a list [0, 3, 6, 12]. Values in a list should be given in hours and sorted in increasing order without duplicates.
    We need to group events based on the duration between the end of the last event (or the start of the stay) and the start of the event. This duration affects the range of possible values for the time gap between the alarm and the event. Some events have less available time in front of them than others, for instance, we can not predict 8 hours in advance for the event that happened 2 hours after the admission.

- `AnalyseMedVarsGroup` computes the median values of specified medical variables for each cohort of patients, based on the training data.  
    For each cohort, the median value is computed for three conditions: all patients across the entire stay, for all patients but only while not in an event, and only for patients not experiencing any event.  
    Please note that to run this stage, you need preprocessed data that are of a similar format as the common stage preprocessed data of HiRID.  
    In GIN config:
    - `use_cache` flag to store analysis output
    - `overwrite` flag to overwrite analysis output that has already been stored
    - `path_common_stage_dir` path to the directory that contains the `common_stage` preprocessed data (resampled on {timestep}-time grid before imputation, see more details in the HiRID benchmark repo: https://github.com/ratschlab/HIRID-ICU-Benchmark) 
    - `patients_source` string that specifies the data source (for the moment only working for `hirid` - we expect that `path_common_stage_dir` is a directory containing `.parquet` files containing time series data for a batch of patients.)
    - `classical_vars` list of medical variables to study (same naming as in `common_stage` tables) 
    - `conditioned_vars` list of medical variables to study but only when they respect a certain condition on other medical variables 
    - `specified_conditions` object specifying conditions for the variables in `conditioned_vars`
    - `vars_units` dictionary mapping each medical variable to its unit, this is optional and will only be used for display purpose

    **Example of how to declare `conditioned_vars`**: We want to study the Mean Arterial Pressure when patients aren't under vasopressors (as they influence the MAP)
    `conditioned_vars={'ABPm': ('has_vasop', False)}` means that we compute the median ABPm when the condition `has_vasop` is False. Then the condition `has_vasop` needs to be defined in `specified_conditions`.  
    `specified_conditions = {'has_vasop': {'vasop1': ('val>0', 'val==0'), 'vasop2': ('val>0', 'val==0')}}` means that `has_vasop` is True for all time points where the value of `vasop1` or the value of `vasop2` is greater than 0.

- `AnalyseFeatImportanceGroup` compares the feature ranking (per importance according to SHAP values) for each cohort of patients.  
    It is possible to use this stage with pre-computed SHAP values and without access to the ML-stage data (same formatting as for HiRID). In this case, the `root_shap_values_path` and `feature_names_path` need to be passed, and in the case of multiple models the `model_seeds` as well.
    We expect the pre-computed SHAP values for a model to be stored in a pickle file containing a dictionary of the format `{patient_id: matrix_shap_values}`, `matrix_shap_values` being a matrix of SHAP values obtained for each feature, for each time step (it is of size `number_timesteps x number_features`).  
    If the SHAP values have not yet been computed then the `ml_stage_data_path`, the `task_name` as well as all the information related to the trained models have to be passed.  
    **Please note that the computation of SHAP values within the stage isn't supported for the DL model. If you want to use the stage for a DL model you will need to pass the pre-computed SHAP values directly as specified above.**  
    In GIN config:
    - `use_cache` flag to store analysis output
    - `overwrite` flag to overwrite analysis output that has already been stored
    - `ml_stage_data_path` path to the ML-stage data (.h5 file) (contains the input to the model, same format as preprocessed HiRID)
    - `split` name of the data split for which we want to compute the SHAP values (either `test` or `val`)
    - `task_name` name (or index) of the task in the ML-stage dataset
    - `model_seeds` list of seeds in case of multiple models (it corresponds to the suffix of the model folder of the form `seed_{model_seeds[i]}`)
    - `root_model_path` path to the directory containing the trained models, in case of multiple models in contained the `seed_{model_seeds[i]}` folders
    - `model_type` string representing the type of model (for the moment only `classical_ml`is supported, by `classical_ml` we mean all non-Deep Learning models that are supported by `shap.Explainer`: https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html)
    - `model_save_name` name under which the model is saved (for `classical_ml` model type, expected name finishing with `.jobllib`)
    - `feature_names_path` path to a pickle file containing the list of feature names (in case the `ml_stage_data_path` hasn't been provided and thus the feature names cannot be inferred)
    - `root_shap_values_path` path to the directory containing the SHAP values (or where to save them), in case of multiple models, it contains the `seed_{model_seeds[i]}` folders

- `AnalyseMissingnessGroup` compares the intensity of measurements for each cohort of patients and assesses the impact of missingness in model performance.  
    Please note that to run this stage, you need preprocessed data that are of a similar format as the common stage preprocessed data of HiRID.  
    In GIN config:
    - `use_cache` flag to store analysis output
    - `overwrite` flag to overwrite analysis output that has already been stored
    - `path_common_stage_dir` path to the directory that contains the `common_stage` preprocessed data (resampled on {timestep}-time grid before imputation, see more details in the HiRID benchmark repo: https://github.com/ratschlab/HIRID-ICU-Benchmark) 
    - `patients_source` string that specifies the data source (for the moment only working for `hirid` - we expect that `path_common_stage_dir` is a directory containing `.parquet` files containing time series data for a batch of patients.)
    - `medical_vars` dictionary of medical variables to study (same naming as in `common_stage` tables) with their corresponding expected sampling intervals (in minutes)
    - `vars_only_patient_msrt` list of medical variables for which we want only to study predictions on patients with measurement (a subset of medical_vars keys), if a variable is used in the label definition it needs to be appended to this list.
    - `category_intensity` Dictionary defining the categories of intensity of measurement of the following format `{intensity_category: (lower, upper)}` (with lower excluded and upper included). Categories have to be in increasing order.
    - `metrics_binary_to_check` List of binary metrics to measure (no event-based metrics should be present), it has to be a subset of `["recall", "precision", "npv", "fpr", "corrected_precision",  "corrected_npv"]`
    - `metrics_score_to_check` List of score-based metrics to measure (no event-based metrics should be present), it has to be a subset of `["positive_class_score", "negative_class_score", "auroc", "auprc", "corrected_auprc"]`

- `CreatePDFReport` generates a PDF report with the results of each of the previous analysis  
    In GIN config:
    - `colors` list of colors for the figure creation
    - `display_stat_test_table` whether to display statistical test tables in the report
    - `figsize_*` size for the different figures to display in the report
    - `k_feat_importance` number of features to display for the `AnalyseFeatImportanceGroup` stage report
    - `max_cols_topkfeat_table`number of adjacent tables to render for the `AnalyseFeatImportanceGroup` stage report


The results of the analysis will be stored in `{log_directory}/fairness_check/{str_stat_test}/{name_threshold}/{merge_str}` with:
- `str_stat_test` being `no_stat_test` if the pipeline is run with `do_stat_test=False` or `w_stat_test` if the pipeline is run with `do_stat_test=True`
- `name_threshold` the string given in the config if any, otherwise `threshold_{threshold_score}`
- if `merge_time_min` is given in the config then `merge_str` is `merge_event_{merge_time_min}min`, else the path ends at the `name_threshold` folder

### Special use-cases

#### Timepoint events
In case the target event has no duration i.e. it is a single timestep, it is possible to run the analysis but the event boundaries should be constructed accordingly `{patient_id: [(timepoint_event, timepoint_event+1)]}`.

#### No early warning
In case the alarm system is meant to raise the alarm when the event starts, the `horizon` has to be set at 0.
The stage `AnalyseTimeGapGroup` cannot be run. Note also that event-based metrics are ill-defined in this case, we thus advise to not use them in the audit and to choose a threshold target that does not depend on them.

#### Classical binary classification
For the following use-cases, `AnalyseTimeGapGroup` and `AnalyseMissingnessGroup`  aren't supported as well as all event-based metrics. 
*Single output for each patient*
Consider a use-case where the input data is still time-series for each patient but the output is a single label.
Predictions have to be stored in the following format `{patient_id: ([prediction], [label])}`. If the label is positive then the event boundary for the patient is `[(0, len(input))]` else it is an empty list.
`AnalyseFeatImportanceGroup` can be run if the SHAP values are pre-computed as explained above.

*Single output and no time-series input for each patient*
Consider a use-case where the input data for a patient is a single value for each feature and the output is a single label. 
Predictions have to be stored in the following format `{patient_id: ([prediction], [label])}`. If the label is positive then the event boundary for the patient is `[(0, 1)]` else it is an empty list. 
`AnalyseMedVarsGroup` can't be run.
`AnalyseFeatImportanceGroup` can be run if the SHAP values are pre-computed as explained above.
