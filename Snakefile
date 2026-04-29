import pathlib

import yaml

REGIONS = [
    "era5_region1",
    "era5_region2",
    "aimip_region1",
    "aimip_region2",
]


def _load_run_set(name):
    path = pathlib.Path(f"configs/run_sets/{name}.yaml")
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())["runs"]


def _normalisation_path_for(run):
    r"""Read the experiment yaml and return its network.normalisation_path,
    or None if unset. Snakemake calls this as a train-rule input function."""
    exp_path = pathlib.Path(f"configs/experiments/{run}.yaml")
    if not exp_path.exists():
        return []
    cfg = yaml.safe_load(exp_path.read_text()) or {}
    path = (cfg.get("network") or {}).get("normalisation_path")
    return [path] if path else []


RUN_CONFIG = config.get("runs", "unet")
_run_set = _load_run_set(RUN_CONFIG)
RUNS = _run_set if _run_set is not None else [r for r in RUN_CONFIG.split(",") if r]
EXTRA = config.get("extra", "")
TRAIN_EXTRA = config.get("train_extra", "")
FORECAST_EXTRA = config.get("forecast_extra", "")
TRAIN_SUITE = config.get(
    "train_suite",
    RUN_CONFIG if _run_set is not None else None,
)
DEVICE = config.get("device", "cuda")
SCORES_DIR = config.get("scores_dir", "scores")
LOG_DIR = config.get("log_dir", "logs/snakemake")
TEAM_NAME = config.get("team_name", "my_team")
DATA_READY = "data/train_data/.download_complete"


def hydra_args(run, extra=""):
    return (
        f"+experiments={run} exp_name={run} device={DEVICE} {EXTRA} {extra}"
    ).strip()


def train_extra_args():
    extras = []
    if TRAIN_SUITE:
        extras.append(f"+suite={TRAIN_SUITE}")
    if TRAIN_EXTRA:
        extras.append(TRAIN_EXTRA)
    return " ".join(extras)


wildcard_constraints:
    run="[^/]+",
    split="val|test",
    region="|".join(REGIONS),


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
        args=lambda w: hydra_args(w.run, extra=train_extra_args()),
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
