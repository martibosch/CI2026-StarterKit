REGIONS = [
    "era5_region1",
    "era5_region2",
    "aimip_region1",
    "aimip_region2",
]

RUNS = [r for r in config.get("runs", "unet").split(",") if r]
EXTRA = config.get("extra", "")
DEVICE = config.get("device", "cuda")
SCORES_DIR = config.get("scores_dir", "scores")
LOG_DIR = config.get("log_dir", "logs/snakemake")
TEAM_NAME = config.get("team_name", "my_team")
DATA_READY = "data/train_data/.download_complete"


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
