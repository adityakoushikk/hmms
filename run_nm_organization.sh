#!/usr/bin/env bash
# Generate provider-month table for NM_organization cohort only.

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
RAW_CSV="$REPO/data/datasets/medicaid-provider-spending.csv"
COHORT_CSV="$REPO/data/outputs/provider_cohorts.csv"
OUTPUT_CSV="$REPO/data/outputs/provider_month_NM_organization.csv"
DICTIONARY_CSV="$REPO/data/outputs/provider_month_data_dictionary.csv"
DICTIONARY_JSON="$REPO/provider_month_data_dictionary.json"

python "$REPO/scripts/create_provider_month_dataset.py" "$RAW_CSV" \
  --cohort-csv "$COHORT_CSV" \
  --cohorts NM_organization \
  --output "$OUTPUT_CSV" \
  --dictionary "$DICTIONARY_CSV" \
  --dictionary-json "$DICTIONARY_JSON"

echo "Done: $OUTPUT_CSV"
