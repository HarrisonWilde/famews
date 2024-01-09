
# FAMEWS: a Fairness Auditing tool for Medical Early-Warning Systems

![FAMEWS Workflow](./data/figures/summary_tool_paper.png)

**FAMEWS** has primarily been designed to run on the HiRID dataset. However, it is possible to give already processed input to some stages in order to run it on other datasets that differ in their format.  
We also encourage users to add functionalities to the tool in order to expand the range of compatible datasets.  
This tool has been created to audit Early-Warning Systems in the medical domain. As such we consider a set of patients with a time-series of input features and a time-series of labels.   
As we focus on early warning, we expect a label for a current time step to be positive when a targeted event occurs a certain amount (called the prediction horizon) of time in the future. While the patient is undergoing an event, we expect the label to be NaN.  

For additional explanations on the tool, please refer to our paper: *FAMEWS: a Fairness Auditing tool for Medical Early-Warning Systems*.  
We provide a sample fairness audit report (`sample_fairness_report.pdf`) that can be produced with FAMEWS. The instructions to reproduce the report are given in the section **Pipeline Overview - How to run FAMEWS on HiRID?** of this README below the header **[TO RUN TO REPRODUCE SAMPLE REPORT]** (there are three steps: HiRID preprocessing, model inference and fairness analysis).

After explaining how to set up FAMEWS, we will describe how to run it on the HiRID dataset and how to obtain the sample report. 
A more [detailed documentation](documentation/DETAILED_DOCS.md) on the extended range of applications is also available.

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
pip install -e ./
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
