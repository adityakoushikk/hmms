#!/usr/bin/env bash
# Build provider cohorts from NPPES + raw Medicaid billing data.

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
NPPES_CSV="$REPO/data/datasets/npidata_pfile_20050523-20260308.csv"
MEDICAID_CSV="$REPO/data/datasets/medicaid-provider-spending.csv"
OUTPUT_CSV="$REPO/data/outputs/provider_cohorts.csv"

python "$REPO/scripts/build_provider_cohorts.py" \
  --nppes_csv "$NPPES_CSV" \
  --medicaid_csv "$MEDICAID_CSV" \
  --output_csv "$OUTPUT_CSV"

echo "Done: $OUTPUT_CSV"
