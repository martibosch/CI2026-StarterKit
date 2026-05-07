import pathlib

import yaml

REGIONS = [
    "era5_region1",
    "era5_region2",
    "aimip_region1",
    "aimip_region2",
]


def _load_run_set(name):
    path = pathlib.Path(f"configs/experiments/{name}.yaml")
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("runs")


def _normalisation_path_for(run):
    r"""Read the experiment config and return its network.normalisation_path,
    or None if unset. Snakemake calls this as a train-rule input function."""
    cfg_path = pathlib.Path(f"configs/experiments/{run}.yaml")
    if not cfg_path.exists():
        return []
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    path = (cfg.get("network") or {}).get("normalisation_path")
    return [path] if path else []


RUN_CONFIG = config.get("runs", "geounet_wide_rh_hilr")
_run_set = _load_run_set(RUN_CONFIG)
RUNS = _run_set if _run_set is not None else [r for r in RUN_CONFIG.split(",") if r]
EXTRA = config.get("extra", "")
FORECAST_EXTRA = config.get("forecast_extra", "")
DEVICE = config.get("device", "cuda")
SCORES_DIR = config.get("scores_dir", "scores")
LOG_DIR = config.get("log_dir", "logs/snakemake")
TEAM_NAME = config.get("team_name", "my_team")
DATA_READY = "data/train_data/.download_complete"
ENSEMBLE_RUNS = [r for r in config.get("ensemble_runs", "").split(",") if r]
ENSEMBLE_NAME = config.get("ensemble_name", "ensemble")
TTA = config.get("tta", False)


def hydra_args(run, extra=""):
    return (
        f"+experiments={run} exp_name={run} device={DEVICE} {EXTRA} {extra}"
    ).strip()


wildcard_constraints:
    run="[^/]+",
    split="val|test",
    region="|".join(REGIONS),


ruleorder: ensemble_forecast > forecast


rule all:
    input:
        expand(f"{SCORES_DIR}/{{run}}.json", run=RUNS),


rule val_scores:
    input:
        expand(f"{SCORES_DIR}/{{run}}.json", run=RUNS),


rule test_forecasts:
    input:
        expand(
            "data/forecasts/{run}/test_{region}.nc",
            run=RUNS,
            region=REGIONS,
        ),


rule compute_normalization:
    input:
        data=DATA_READY,
    output:
        stats="data/stats/normalization_{rh}_aux{n_aux}.json",
    log:
        f"{LOG_DIR}/stats/normalization_{{rh}}_aux{{n_aux}}.log",
    wildcard_constraints:
        rh="rh|no_rh",
        n_aux=r"\d+",
    params:
        zarr_path="data/train_data/train.zarr",
        use_rh_flag=lambda w: "--use_rh" if w.rh == "rh" else "",
    shell:
        """
        python scripts/compute_normalization.py \
            --zarr_path {params.zarr_path} \
            {params.use_rh_flag} \
            --n_auxiliary_fields {wildcards.n_aux} \
            --output_path {output.stats} 2>&1 | tee {log}
        """


rule download_data:
    output:
        marker=DATA_READY,
    log:
        f"{LOG_DIR}/download_data.log",
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


rule train:
    input:
        data=DATA_READY,
        stats=lambda w: _normalisation_path_for(w.run),
    output:
        ckpt="data/models/{run}/best_model.ckpt",
    log:
        f"{LOG_DIR}/{{run}}/train.log",
    resources:
        gpu=1,
    params:
        args=lambda w: hydra_args(w.run),
    shell:
        "python scripts/train.py {params.args} 2>&1 | tee {log}"


rule forecast:
    input:
        ckpt="data/models/{run}/best_model.ckpt",
    output:
        forecast="data/forecasts/{run}/{split}_{region}.nc",
    log:
        f"{LOG_DIR}/{{run}}/forecast_{{split}}_{{region}}.log",
    resources:
        gpu=1,
    params:
        args=lambda w, input, output: hydra_args(
            w.run,
            extra=(
                f"{FORECAST_EXTRA} "
                f"+test_data={w.split}_{w.region} "
                f"ckpt_path={input.ckpt} "
                f"output_path={output.forecast}"
            ),
        ),
    shell:
        "python scripts/forecast.py {params.args} 2>&1 | tee {log}"


rule evaluate:
    input:
        expand("data/forecasts/{{run}}/val_{region}.nc", region=REGIONS),
    output:
        score=f"{SCORES_DIR}/{{run}}.json",
    log:
        f"{LOG_DIR}/{{run}}/evaluate.log",
    params:
        prediction_dir=lambda w: f"data/forecasts/{w.run}",
        team_name=TEAM_NAME,
    shell:
        """
        python scripts/evaluate.py \
            --prediction_dir {params.prediction_dir} \
            --prefix val \
            --to_json \
            --team_name {params.team_name} \
            --output_path {output.score} 2>&1 | tee {log}
        """


rule ensemble_forecast:
    input:
        ckpts=expand("data/models/{run}/best_model.ckpt", run=ENSEMBLE_RUNS),
    output:
        forecast=f"data/forecasts/{ENSEMBLE_NAME}/{{split}}_{{region}}.nc",
    log:
        f"{LOG_DIR}/{ENSEMBLE_NAME}/forecast_{{split}}_{{region}}.log",
    resources:
        gpu=1,
    params:
        base_args=hydra_args(ENSEMBLE_RUNS[0]) if ENSEMBLE_RUNS else "",
        ckpt_list=lambda w, input: "[" + ",".join(input.ckpts) + "]",
        tta=TTA,
    shell:
        """
        python scripts/forecast.py {params.base_args} \
            +test_data={wildcards.split}_{wildcards.region} \
            output_path={output.forecast} \
            "ensemble_ckpt_paths={params.ckpt_list}" \
            tta={params.tta} 2>&1 | tee {log}
        """


rule ensemble_val_scores:
    input:
        f"{SCORES_DIR}/{ENSEMBLE_NAME}.json",


rule ensemble_evaluate:
    input:
        expand(f"data/forecasts/{ENSEMBLE_NAME}/val_{{region}}.nc", region=REGIONS),
    output:
        score=f"{SCORES_DIR}/{ENSEMBLE_NAME}.json",
    log:
        f"{LOG_DIR}/{ENSEMBLE_NAME}/evaluate.log",
    params:
        prediction_dir=f"data/forecasts/{ENSEMBLE_NAME}",
        team_name=TEAM_NAME,
    shell:
        """
        python scripts/evaluate.py \
            --prediction_dir {params.prediction_dir} \
            --prefix val \
            --to_json \
            --team_name {params.team_name} \
            --output_path {output.score} 2>&1 | tee {log}
        """
