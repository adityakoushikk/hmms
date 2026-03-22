#!/usr/bin/env bash
# Run anomaly detection training (autoencoder) with Hydra.
#
# provider_month_csv comes from run_provider_month.sh (e.g. provider_month_NV_organization.csv).
# provider_level_csv is optional:
#   - omit it (null) → computed fresh next to provider_month_csv
#   - provide a path to an existing file → skip recomputation
#
# Loggers:
#   - csv  (default) → writes metrics.csv + hparams.yaml to the Hydra output dir; no account needed
#   - wandb          → streams to Weights & Biases; requires WANDB_API_KEY env var
#
# Examples:
#   # 1. First run — compute provider_level fresh, log to CSV
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month_NV_organization.csv
#
#   # 2. Reuse an existing provider_level (skip recomputation)
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month_NV_organization.csv \
#     data.provider_level_csv=data/outputs/provider_level_NV_organization.csv
#
#   # 3. Switch to WANDB logger
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month_NV_organization.csv \
#     logger=wandb logger.project=my-project
#
#   # 4. Tune autoencoder architecture
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month_NV_organization.csv \
#     model.net.encoder_dims=[128,64,32] model.net.bottleneck_dim=16 model.optimizer.lr=0.0005

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"

python -m anomaly_detect.train "$@"
