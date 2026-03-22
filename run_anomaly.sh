#!/usr/bin/env bash
# Run anomaly detection training (autoencoder) with Hydra.
#
# provider_month_csv comes from run_provider_month.sh (e.g. provider_month_NV_organization.csv).
# provider_level_csv is optional:
#   - omit it (null) → computed fresh next to provider_month_csv
#   - provide a path to an existing file → skip recomputation
#
# Examples:
#   # Compute provider_level fresh (first run)
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month_NV_organization.csv
#
#   # Reuse an existing provider_level (skip recomputation)
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month_NV_organization.csv \
#     data.provider_level_csv=data/outputs/provider_level_NV_organization.csv
#
#   # Tune autoencoder architecture
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month_NV_organization.csv \
#     model.net.encoder_dims=[128,64,32] model.net.bottleneck_dim=16 model.optimizer.lr=0.0005
#
#   # Disable feature selection (use all features)
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month_NV_organization.csv \
#     data.feature_selection.auroc_threshold=null

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"

python -m anomaly_detect.train "$@"
