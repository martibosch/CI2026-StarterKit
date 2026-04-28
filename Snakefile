import shlex


REGIONS = [
    "era5_region1",
    "era5_region2",
    "aimip_region1",
    "aimip_region2",
]

DEFAULT_RUNS = {
    "unet": {
        "experiment": "unet",
        "overrides": [],
    },
}

SCORES_DIR = config.get("scores_dir", "scores")
LOG_DIR = config.get("log_dir", "logs/snakemake")
DEVICE = config.get("device", "cuda")
EXPERIMENT_GROUP = config.get("experiment_group", "experiments")
TEAM_NAME = config.get("team_name", "my_team")
COMMON_OVERRIDES = config.get("overrides", [])
TRAIN_OVERRIDES = config.get("train_overrides", [])
FORECAST_OVERRIDES = config.get("forecast_overrides", [])
DATA_READY = "data/train_data/.download_complete"


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.split() if item]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_names(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_runs(value):
    if not value:
        return None
    if isinstance(value, dict):
        runs = {}
        for name, spec in value.items():
            if spec is None:
                runs[str(name)] = {"experiment": str(name), "overrides": []}
            elif isinstance(spec, str):
                runs[str(name)] = {"experiment": spec, "overrides": []}
            else:
                runs[str(name)] = dict(spec)
        return runs
    return {
        name: {"experiment": name, "overrides": []}
        for name in _as_names(value)
    }


RUNS = (
    _coerce_runs(config.get("runs"))
    or _coerce_runs(config.get("experiments"))
    or DEFAULT_RUNS
)
RUN_NAMES = list(RUNS.keys())


def _quoted(items):
    return " ".join(shlex.quote(str(item)) for item in items if item is not None)


def _run_spec(run):
    return RUNS[run]


def _experiment_args(run):
    experiment = _run_spec(run).get("experiment", run)
    if experiment in (None, ""):
        return []
    return [f"+{EXPERIMENT_GROUP}={experiment}"]


def _global_config_overrides():
    overrides = [
        f"device={DEVICE}",
    ]
    for key in ("n_epochs", "batch_size", "learning_rate", "seed"):
        if key in config:
            overrides.append(f"{key}={config[key]}")
    return overrides


def _hydra_args(run, stage, extra=None):
    spec = _run_spec(run)
    overrides = []
    overrides.extend(_experiment_args(run))
    overrides.append(f"exp_name={run}")
    overrides.extend(_global_config_overrides())
    overrides.extend(_as_list(COMMON_OVERRIDES))
    overrides.extend(_as_list(spec.get("overrides", [])))

    if stage == "train":
        overrides.extend(_as_list(TRAIN_OVERRIDES))
        overrides.extend(_as_list(spec.get("train_overrides", [])))
    elif stage == "forecast":
        overrides.extend(_as_list(FORECAST_OVERRIDES))
        overrides.extend(_as_list(spec.get("forecast_overrides", [])))

    overrides.extend(_as_list(extra))
    return _quoted(overrides)


wildcard_constraints:
    run="[^/]+",
    split="val|test",
    region="era5_region1|era5_region2|aimip_region1|aimip_region2"


rule all:
    input:
        expand("{scores_dir}/{run}.json", scores_dir=SCORES_DIR, run=RUN_NAMES)


rule download_data:
    output:
        marker=DATA_READY
    log:
        f"{LOG_DIR}/download_data.log"
    shell:
        """
        hf download tobifinn/CI2026Hackathon \
            --repo-type dataset \
            --local-dir data/train_data > {log} 2>&1
        find data/train_data -name "*.zip" | while read -r zip_file; do
            target_dir="${{zip_file%.zip}}"
            unzip -o "$zip_file" -d "$target_dir" >> {log} 2>&1
        done
        touch {output.marker}
        """


rule val_scores:
    input:
        expand("{scores_dir}/{run}.json", scores_dir=SCORES_DIR, run=RUN_NAMES)


rule test_forecasts:
    input:
        expand(
            "data/forecasts/{run}/test_{region}.nc",
            run=RUN_NAMES,
            region=REGIONS,
        )


rule train:
    input:
        data=DATA_READY
    output:
        ckpt="data/models/{run}/best_model.ckpt"
    log:
        f"{LOG_DIR}/{{run}}/train.log"
    params:
        hydra_args=lambda wildcards: _hydra_args(wildcards.run, "train"),
    shell:
        """
        python scripts/train.py {params.hydra_args} > {log} 2>&1
        """


rule forecast:
    input:
        ckpt="data/models/{run}/best_model.ckpt"
    output:
        forecast="data/forecasts/{run}/{split}_{region}.nc"
    log:
        f"{LOG_DIR}/{{run}}/forecast_{{split}}_{{region}}.log"
    params:
        hydra_args=lambda wildcards, input, output: _hydra_args(
            wildcards.run,
            "forecast",
            extra=[
                f"+test_data={wildcards.split}_{wildcards.region}",
                f"ckpt_path={input.ckpt}",
                f"output_path={output.forecast}",
            ],
        ),
    shell:
        """
        python scripts/forecast.py {params.hydra_args} > {log} 2>&1
        """


rule evaluate:
    input:
        expand("data/forecasts/{{run}}/val_{region}.nc", region=REGIONS)
    output:
        score=f"{SCORES_DIR}/{{run}}.json"
    log:
        f"{LOG_DIR}/{{run}}/evaluate.log"
    params:
        prediction_dir=lambda wildcards: f"data/forecasts/{wildcards.run}",
        team_name=lambda wildcards: TEAM_NAME,
    shell:
        """
        python scripts/evaluate.py \
            --prediction_dir {params.prediction_dir} \
            --prefix val \
            --to_json \
            --team_name {params.team_name} \
            --output_path {output.score} > {log} 2>&1
        """
