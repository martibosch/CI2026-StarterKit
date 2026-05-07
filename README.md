# Climate informatics 26 Hackathon Starter Kit

[![Launch on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-3CtrQ1MUnbnFvqugOrn3to4rqRi)
[![Submit Solution](https://img.shields.io/badge/Submit%20Solution-0A66C2?style=for-the-badge&logo=gradio&logoColor=white)](https://submission-7rre8iitk.brevlab.com/)
[![Live Leaderboard](https://img.shields.io/badge/Live%20Leaderboard-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)](https://tobifinn-ci2026-hackathon-leaderboard.hf.space/)
[![Dataset](https://img.shields.io/badge/Dataset-CI2026Hackathon-FF9D00?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/tobifinn/CI2026Hackathon)

This is the official starter kit for the Climate Informatics 26 hackathon. It
contains prepared code and scripts to help you get started with your hackathon
project. The kit includes:

- PyTorch-ready data loader
- Training and forecasting scripts supporting your own implementation
- Script to submit model predictions to the leaderboard
- Three baseline solutions
- Flexible configuration with Hydra
- Best practices in implementing PyTorch models for geoscience from multi-year experience

## The task

The goal is to predict **total cloud cover** — a single fraction in [0, 1]
representing the proportion of a grid cell covered by clouds — for every
location on a global 64×64 grid at 1.5° horizontal resolution.

Inputs are drawn from **ERA5**, the ECMWF atmospheric reanalysis or from
**AIMIP** (AI Model Intercomparison Project, for validation and testing only).
Each sample contains daily-averaged fields:

- **Pressure-level fields** (`input_level`): temperature, specific humidity,
  and horizontal wind components (u, v) at 7 pressure levels (1000, 850, 700,
  500, 250, 100, and 50 hPa).
- **Auxiliary static fields** (`input_auxiliary`): land-sea mask, orography,
  land-cover type, longitude, and latitude.

For all inputs, the **target** is the daily-averaged total cloud cover from
ERA5.

The training set consists of ERA5 data from 1979-01-01 to 2018-12-31 for a
single region (region 1). The validation and test sets consist of ERA5 and
AIMIP data for two regions. Consequently, there are four different evaluation configurations: ERA5 for region 1, ERA5 for region 2, AIMIP for region 1, and
AIMIP for region 2. Two different regions are tested for generalization to
unseen geographies, and the AIMIP configurations are tested for generalization
to AI weather models.

The ERA5 submissions are scored with the mean absolute error (MAE); AIMIP
submissions are scored with the continuous ranked probability score (CRPS),
which additionally rewards well-calibrated uncertainty by asking for 3
ensemble members instead of a single prediction.

The composite **skill score** is the average of the four individual skill
scores (one per region and dataset), where each individual skill score is
defined as `1 − score / baseline_score`. A skill score of 0 matches the
baseline; a higher score is better.

## Workflow overview

The typical workflow from setup to leaderboard entry follows these five steps:

1. **Set up** — create the Pixi environment, install dependencies, and
   download the data (see `Get started`\_).
1. **Train** — run `scripts/train.py` to train a model; checkpoints are
   saved under `data/models/<exp_name>/`.
1. **Forecast** — run `scripts/forecast.py +suite=val` to produce
   regional netCDF forecast files for all four evaluation configurations.
1. **Evaluate** — run `scripts/evaluate.py` to score your forecasts against
   the validation targets and check your skill score locally.
1. **Submit** — run `scripts/submit.py` (or upload manually via the portal)
   to send your test-set forecasts to the leaderboard.

## Directories

The repository contains the following summarized directories:

| Directory                | Description                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configs/`               | Hydra configuration files for all scripts. Sub-folders cover `data` (dataset paths and splits), `experiments` (full experiment presets combining model + hyperparameters), `model` (architecture hyperparameters), `suite` (multi-region forecast suites), and `test_data` (test-set paths). Root-level YAMLs (`train.yaml`, `forecast.yaml`, `submit.yaml`) are the default configs for each script. |
| `configs/experiments/`   | Ready-to-use experiment presets for the three baselines (`baseline_mlp`, `baseline_parametric`, `baseline_sundquist`). Pass one with `+experiments=<name>` to reproduce a baseline run end-to-end.                                                                                                                                                                                                    |
| `configs/model/`         | Per-architecture hyperparameter files (`mlp.yaml`, `parametric.yaml`, `sundquist.yaml`). Create a new file here when you implement your own model.                                                                                                                                                                                                                                                    |
| `configs/suite/`         | Suite configs (`val.yaml`, `test.yaml`) that instruct `forecast.py` to produce all four regional forecast files in one go.                                                                                                                                                                                                                                                                            |
| `data/`                  | Runtime data directory (not committed). Sub-folders are `train_data/` (downloaded zarr archives and validation/test targets), `models/` (saved checkpoints and training logs, keyed by `exp_name`), and `forecasts/` (forecast netCDF files, keyed by `exp_name`).                                                                                                                                    |
| `notebooks/`             | Place for your own Jupyter notebooks. Drop exploration and analysis notebooks here.                                                                                                                                                                                                                                                                                                                   |
| `scripts/`               | Entry-point scripts driven by Hydra. `train.py` trains a model, `forecast.py` produces regional netCDF forecasts, `evaluate.py` scores forecasts against validation targets, and `submit.py` runs the forecast suite and POSTs predictions to the submission portal.                                                                                                                                  |
| `starter_kit/`           | Installable Python package, installed editable by Pixi. Contains the core library code: `data.py` (PyTorch datasets), `layers.py` (input normalization), `model.py` (abstract `BaseModel` trainer), and the `baselines/` sub-package.                                                                                                                                                                 |
| `starter_kit/baselines/` | Three concrete baseline implementations: `mlp.py` (multi-layer perceptron), `parametric.py` (parametric cloud-cover scheme), and `sundquist.py` (Sundqvist diagnostic scheme). Use these as templates for your own model.                                                                                                                                                                             |

## Get started

**Prerequisites**: `Pixi <https://pixi.sh>`\_ must be installed. Python is
managed by Pixi. A GPU is optional but speeds up training significantly; the
scripts run on CPU by default. For GPU training, use the Nvidia Brev launchable,
which installs the CUDA Pixi environment automatically.

To get started you can either use the provided Nvidia Brev launchable, which
sets up the environment for you and automatically downloads the data, or you
can set up the environment manually as follows:

0. (Best practices) Fork the repository with GitHub and track your changes with
   git in the forked repository

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/tobifinn/CI2026-StarterKit.git
   cd CI2026-StarterKit
   ```

   or if you have forked the repository

   ```bash
   git clone https://github.com/<your_github_name>/<your_repo_name>.git
   cd <your_repo_name>
   ```

1. Install the Pixi environment:

   ```bash
   pixi install
   ```

1. Enter the Pixi environment:

   ```bash
   pixi shell
   ```

1. Download the data from HuggingFace to `data/train_data` and unzip the zarr archives:

   ```bash
   pixi run snakemake --cores 1 download_data
   ```

Now you are ready for the next steps: training of your own model, producing
forecasts, and submitting solutions to the leaderboard.

## Training

To train a model, you can use the provided `train.py` script. Based on Hydra,
the script supports flexible configuration and allows you to easily switch
between different implemented models.

**New to Hydra?** Hydra is a configuration framework used by all scripts here.
The short version: every configuration key can be overridden directly on the
command line as `key=value` (e.g. `batch_size=16`), and preset configuration
groups can be loaded with `+group=name` (e.g. `+experiments=baseline_mlp`).
You do not need to edit any YAML file to run the provided examples.

The recommended experiment workflow uses Snakemake. For example, to train,
forecast the four validation regions, and evaluate the baseline MLP:

```bash
pixi run snakemake --cores 1 --config runs=baseline_mlp
```

The trained model is saved to `data/models/baseline_mlp/`. The experiment name
controls this path and links training to later steps. Forecasts are saved under
`data/forecasts/baseline_mlp/`, and validation scores under
`scores/baseline_mlp.json`.

Any key in the configuration can be overridden directly on the command line via
the `extra` config arg, which is passed verbatim to Hydra. For example, to
increase the number of epochs and adjust the learning rate:

```bash
pixi run snakemake --cores 1 --config runs=baseline_mlp extra="n_epochs=50 learning_rate=5e-4"
```

You can see all train config options by running:

```bash
pixi run python scripts/train.py --help
```

To switch to a different baseline, use the `+experiment` flag with one of the
presets under `configs/experiments/`:

```bash
pixi run snakemake --cores 1 --config runs=baseline_parametric
pixi run snakemake --cores 1 --config runs=baseline_sundquist
```

## Forecasting

The Snakemake workflow runs validation forecasts automatically after training.
To generate test forecasts for a trained experiment, run:

```bash
pixi run snakemake --cores 1 test_forecasts --config runs=baseline_mlp
```

Forecast files are saved under `data/forecasts/${exp_name}/`.

Once again you can overwrite any key in the configuration on the command line.
For example, you can set a specific model checkpoint and the batch size by
setting:

````
```bash
pixi run snakemake --cores 1 test_forecasts --config runs=baseline_mlp extra="batch_size=8"
```
````

Usually, the forecast script should support all experiment and model config
options from the training script. Hence, to change the experiment to the
Sundqvist baseline implementation, you can use:

````
```bash
pixi run snakemake --cores 1 test_forecasts --config runs=baseline_sundquist
```
````

The checkpoint path can be also set to `None` to skip the checkpoint loading
and to run forecasts with an untrained model. This is helpful for debugging
purposes. The following command would run the forecast for the Sundqvist
baseline without any checkpoint loading and store the forecast to
`data/forecasts/baseline_sundquist_untrained/forecast.nc`:

````
```bash
python scripts/forecast.py \
    +experiments=baseline_sundquist \
    ckpt_path=null \
    exp_name=baseline_sundquist_untrained
```
````

## Evaluation

The starter kit also contains the evaluation script `evaluate.py` used by the
submission portal. While the submission portal uses the test set, you can
*only* use the validation set to automatically evaluate solutions.

The evaluation is performed over four different configurations:

- ERA5 (2019-01-01 to 2019-12-31) for region 1
- ERA5 (2019-01-01 to 2019-12-31) for region 2
- AIMIP (*masked*) for region 1
- AIMIP (*masked*) for region 2

As a consequence, the evaluation script needs for different forecasts over the
validation files. The forecast script supports a `suite` parameter to predict
with your model for all four configurations. The following command would run
the sundquist model for the validation suite:

````
```bash
python scripts/forecast.py +experiments=baseline_sundquist +suite=val
```
````

The files would be stored as:

- ERA5 region 1: `data/forecasts/baseline_sundquist/val_era5_region1.nc`
- ERA5 region 2: `data/forecasts/baseline_sundquist/val_era5_region2.nc`
- AIMIP region 1: `data/forecasts/baseline_sundquist/val_aimip_region1.nc`
- AIMIP region 2: `data/forecasts/baseline_sundquist/val_aimip_region2.nc`

While you can create the files on your own, we recommend to run the suite as it
produces correctly named files.

After producing all four forecasting files with the suite, you can run the
evaluation. For the Sundqvist experiment it would read:

````
```bash
python scripts/evaluate.py \
    --prediction_dir data/forecasts/baseline_sundquist
```
````

The evaluation script prints the scores. The scores can be stored additionally
to a `json` file (here to `scores/baseline_sundquist.json`) by:

````
```bash
python scripts/evaluate.py \
    --prediction_dir data/forecasts/baseline_sundquist \
    --to_json \
    --output_path scores/baseline_sundquist.json
```
````

The output will contain the mean absolute error (MAE) for the two ERA5
configurations and the continuous ranked probability score (CRPS) for the two
AIMIP configurations, as well as the combined skill `score` (here, the higher, the better) and a dummy team_name. The json for a trained Sundqvist experiment
in the validation suite should look like:

````
```json
{
    "mae_era5_region1": 0.1610385481594196,
    "mae_era5_region2": 0.15878042971186562,
    "crps_aimip_region1": 0.14800346999624828,
    "crps_aimip_region2": 0.15465294870905463,
    "score": -0.014118525220073369,
    "team_name": "my_team"
}
```
````

For the Sundqvist experiment, it is expected that the skill score is around 0., as it is normalized by test scores of a trained Sundqvist model.

The storage of `json` files for different configurations allows you a
programmatic evaluation of your model.

While you can run the forecast suite for the test set by `+suite=test`, you
cannot run the `evaluate.py` script over the test set as the target files are
missing.

## Submission

There are two different ways to submit a solution to the leaderboard: using the
`submit.py` script or manual submission. For a submission, you need a
whitelisted email address and the four needed forecasting files.

The email address is used to check if you are registered for the hackathon, to
check your submission rate limits (**3 submissions/address/hour**), and
to link your submission to a team name on the leaderboard. If you don't have a
whitelisted email address, please contact the hackathon organizers.

1. `submit.py` script: run the script with your email address and your forecast
   configuration (here once again for the Sundqvist baseline):

   ```bash
   python scripts/submit.py \
       --email <your email address> \
       --experiments=baseline_sundquist
   ```

   The submission script will automatically run the forecast suite for the
   test set, store them under `data/forecasts/<exp_name>/`, and submit them to
   the leaderboard via API. When you want to submit forecasts that you already
   produced, you can skip the forecast suite
   by passing `skip_forecast=true`:

   ```bash
   python scripts/submit.py \
       --email <your email address> \
       --experiments=baseline_sundquist \
       --skip_forecast true
   ```

   For the forecast configuration, the submit script reuses all keywords also
   available for the forecasting script.

1. Manual submission: when you have already produced forecasts you can also
   manually submit them either via the
   [portal](https://submission-7rre8iitk.brevlab.com/) or API. To submit via
   API you can use (once again for the Sundqvist baseline):

   ```bash
   curl -s -X POST https://submission-7rre8iitk.brevlab.com/api/v1/submissions \
       -F "email=<your email address>" \
       -F "file_era5_region1=@data/forecasts/baseline_sundquist/test_era5_region1.nc" \
       -F "file_era5_region2=@data/forecasts/baseline_sundquist/test_era5_region2.nc" \
       -F "file_aimip_region1=@data/forecasts/baseline_sundquist/test_aimip_region1.nc" \
       -F "file_aimip_region2=@data/forecasts/baseline_sundquist/test_aimip_region2.nc" \
       | python -m json.tool
   ```

In both cases, the submission portal compares your email address to the
whitelist and spawns a job to evaluate your submission. Your job is queued and
will be executed when previous jobs are finished. Hence, it might happen that you have to wait until your submission is fully evaluated, especially during
high time.

To check your submission status, you can use the API with the command

````
```bash
curl -s https://submission-7rre8iitk.brevlab.com/api/v1/submissions/<unique_idx> \
    | python -m json.tool
```
````

You can also use the submissions [portal](https://submission-7rre8iitk.brevlab.com/) to check your submission:

- In **Status**, you can type your submission ID to check the status and
  the scores of your submission.
- In **Leaderboard**, a time-ordered leaderboard is shown, your submission will
  be likely one of the last. You can click on the submission ID to check the scores of your submission.

When your submission was successful, it will be added to previously mentioned
time-ordered leaderboard and to the hourly-updated official
[leaderboard](https://tobifinn-ci2026-hackathon-leaderboard.hf.space/).

## Baseline models

Three baselines are provided, each representing a different modeling
approach:

- **MLP** (`baseline_mlp`): a fully-connected neural network that takes all
  pressure-level and auxiliary fields as input and learns a direct mapping to
  cloud cover. It is the recommended starting point for participants familiar
  with deep learning.
- **Parametric** (`baseline_parametric`): derives cloud cover per level from
  relative humidity using a learnable sigmoid threshold, then combines levels
  with a maximum-overlap assumption. Only temperature and specific humidity
  are used. It is compact and interpretable, and trains quickly.
- **Sundqvist** (`baseline_sundquist`): a physically-motivated diagnostic
  scheme (Sundqvist et al., 1989) that computes cloud cover from relative
  humidity relative to a learnable critical threshold. It serves as a physical
  reference and defines the normalization baseline for the skill score (a
  skill score of ~0 corresponds to Sundqvist performance).

All three follow the same training interface and can be swapped by passing
`+experiments=<name>` to `train.py`.

## Own implementation

To implement your own model, you have to write a new PyTorch module that
inherits from `BaseModel` and implements the required methods. You can find the
base class in `starter_kit/model.py`. We recommend to use the provided baseline
models as a reference for your implementation. You can find them in
`starter_kit/baselines/`.

You will see that the baseline models separately implement the neural network
and the training logic. This allows you to easily reuse the training logic and
only implement the neural network when you want to try out different
architectures.

Furthermore, the `BaseModel` provides all needs for training and
validation, except the `estimate_loss` method, which you have to implement in your model. The `estimate_loss` method takes a batch of data as input and returns a dictionary with at least a `loss` key. This allows you to implement any loss function you want.

While the `estimate_loss` method should be as slim as possible to obtain the fastest possible training, you can implement additional metrics tracked during validation in the (non-required) `estimate_auxiliary_loss` method. This method
is called after `estimate_loss` within the validation loop. The batch and the output of `estimate_loss` are passed as input. The output should only contain
additional metrics tracked during validation. By passing the output of `estimate_loss` to `estimate_auxiliary_loss`, you can avoid redundant computations and obtain faster validation. The metrics returned by `estimate_auxiliary_loss` are logged in the same way as the loss returned by `estimate_loss` and can be used for early stopping and model selection.

______________________________________________________________________

When implementing your own model, create also a new model configuration under
`configs/model/` and pass it to the training script with the `model` flag. For
example, if you create a model config named `my_model.yaml`, you can train it
with:

```bash
pixi run snakemake --cores 1 --config runs=my_model_run extra="model=my_model"
```

To obtain reproducible configurations, we recommend to additionally create a
new experiment config that references your model config and pass it with the
`+experiment` flag. For example, if you create an experiment config named
`my_experiment.yaml` that references `my_model.yaml`, you can train it with:

```bash
pixi run snakemake --cores 1 --config runs=my_experiment
```

In this case, you no longer need to pass the `model` flag, since it is
referenced in the experiment config. This also allows you to easily switch
between different experiments and models by just changing the experiment
config.

## Getting help

If you run into problems, have questions about the data, or need your email
address whitelisted for submission, please reach out to the hackathon
organizers.
