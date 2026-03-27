#!/usr/bin/env bash
# Generate a fraud/abuse/waste anomaly report for a single NPI.
#
# Usage:
#   ./run_anomaly_report.sh <TARGET_NPI>
#
# Override any path via environment variable before calling:
#   SCORED_CSV=data/outputs/my_run/scored_providers.csv ./run_anomaly_report.sh 1649949124

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"

# ── Target NPI (required) ─────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
  echo "Usage: $0 <TARGET_NPI>"
  exit 1
fi
TARGET_NPI="$1"

# ── Input paths ───────────────────────────────────────────────────────────────
NPPES_CSV="${NPPES_CSV:-$REPO/data/datasets/nppes.csv}"
PROVIDER_MONTH_CSV="${PROVIDER_MONTH_CSV:-$REPO/data/outputs/provider_month_NV_organization.csv}"
SCORED_CSV="${SCORED_CSV:-$REPO/data/outputs/2026-03-22_23-15-56/scored_providers.csv}"
FEAT_ERRORS_CSV="${FEAT_ERRORS_CSV:-$REPO/data/outputs/2026-03-22_23-15-56/provider_feature_errors.csv}"
PROVIDER_LEVEL_CSV="${PROVIDER_LEVEL_CSV:-$REPO/data/outputs/2026-03-21_22-57-39/provider_level.csv}"

# ── Run ───────────────────────────────────────────────────────────────────────
python "$REPO/scripts/generate_anomaly_report.py" "$TARGET_NPI" \
  --nppes-csv          "$NPPES_CSV" \
  --provider-month-csv "$PROVIDER_MONTH_CSV" \
  --scored-csv         "$SCORED_CSV" \
  --feat-errors-csv    "$FEAT_ERRORS_CSV" \
  --provider-level-csv "$PROVIDER_LEVEL_CSV"
