
# FAMEWS: a Fairness Auditing tool for Medical Early-Warning Systems

![FAMEWS Workflow](./data/figures/summary_tool_paper.png)

## Setup

This repository depends on the work done by [Yèche et al. HiRID Benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark)
to preprocess the HiRID dataset and get it ready for model training, as well as inference and fairness analysis.

The [HiRID Benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark) repository with the preprocessing is included as a submodule in this repository. To clone the repository with the submodule, run:

```bash
git submodule init
git submodule update

# follow instructions in `HiRID Benchmark` repository to download and preprocess the dataset
# the subsequent steps rely on the different stage outputs defined by Yèche et al.
```

Then please follow the instructions of the HiRID Benchmark repository to obtain preprocessed data in a suitable format.

### Conda Environment

A conda environment configuration is provided: `environment_linux.yml`. You can create 
the environment with:
```
conda env create -f environment_linux.yml
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
machine learning pipelines, preprocessing, and evaluation pipelines. Example configurations are in `./config/example`.
If implement a configurable component, please make sure an example config of that component is to be found
in any of the example configurations.

## Pipeline Overview

Any task (preprocessing, training, evaluation) is to be run with a script located in
`famews/scripts`. Ideally these scripts invoke a `Pipeline` object, which conists of different
`PipelineStage` objects.

### Preprocessing
 
#### HiRID

TODO: refer to HiRID submodule

### ML Training

#### `DLTrainPipeline`

A training run is divided into a set of training stages (implemented as `PipelineStage`)
and configured using gin configuration files.

An individual run can be started similar to the following command:
```
python -m famews.scripts.train_sequence_model \
    -g ./config/example/lstm_test.gin \
    -l ./logs/lstm_test \
    --seed 1111 \
    --wandb_project test
```






