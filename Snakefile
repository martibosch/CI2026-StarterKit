REGIONS = [
    "era5_region1",
    "era5_region2",
    "aimip_region1",
    "aimip_region2",
]

RUN_SETS = {
    "wandb_grid": [
        "baseline_parametric",
        "baseline_sundquist",
        "mlp_no_rh_aux2",
        "mlp_no_rh_aux5",
        "mlp_rh_aux2",
        "mlp_rh_aux5",
        "unet_no_rh_aux2",
        "unet_no_rh_aux5",
        "unet_rh_aux2",
        "unet_rh_aux5",
    ],
}

RUN_CONFIG = config.get("runs", "unet")
RUNS = RUN_SETS.get(RUN_CONFIG, [r for r in RUN_CONFIG.split(",") if r])
EXTRA = config.get("extra", "")
TRAIN_EXTRA = config.get("train_extra", "")
FORECAST_EXTRA = config.get("forecast_extra", "")
DEVICE = config.get("device", "cuda")
SCORES_DIR = config.get("scores_dir", "scores")
LOG_DIR = config.get("log_dir", "logs/snakemake")
TEAM_NAME = config.get("team_name", "my_team")
DATA_READY = "data/train_data/.download_complete"

# Per-run normalisation artifact. Runs not listed fall back to the hardcoded
# stats baked into the network (only valid for use_rh=False, n_aux=2).
NORMALISATION_STATS = {
    "mlp_rh": "data/stats/normalization_rh_aux2.json",
    "mlp_no_rh": "data/stats/normalization_no_rh_aux2.json",
    "unet_rh": "data/stats/normalization_rh_aux2.json",
    "unet_no_rh": "data/stats/normalization_no_rh_aux2.json",
    "mlp_no_rh_aux2": "data/stats/normalization_no_rh_aux2.json",
    "mlp_no_rh_aux5": "data/stats/normalization_no_rh_aux5.json",
    "mlp_rh_aux2": "data/stats/normalization_rh_aux2.json",
    "mlp_rh_aux5": "data/stats/normalization_rh_aux5.json",
    "unet_no_rh_aux2": "data/stats/normalization_no_rh_aux2.json",
    "unet_no_rh_aux5": "data/stats/normalization_no_rh_aux5.json",
    "unet_rh_aux2": "data/stats/normalization_rh_aux2.json",
    "unet_rh_aux5": "data/stats/normalization_rh_aux5.json",
}


def hydra_args(run, extra=""):
    return (
        f"+experiments={run} exp_name={run} device={DEVICE} {EXTRA} {extra}"
    ).strip()


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
        stats=lambda w: NORMALISATION_STATS.get(w.run, []),
    output:
        ckpt="data/models/{run}/best_model.ckpt",
    log:
        f"{LOG_DIR}/{{run}}/train.log",
    resources:
        gpu=1,
    params:
        args=lambda w: hydra_args(w.run, extra=TRAIN_EXTRA),
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
