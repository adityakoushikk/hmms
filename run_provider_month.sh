#!/usr/bin/env bash
# Generate provider-month table for one or more cohorts.
# Usage:
#   ./run_provider_month.sh                          # default: NY_individual
#   ./run_provider_month.sh NY_individual            # single cohort
#   ./run_provider_month.sh NY_individual CA_individual  # multiple cohorts

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

RAW_CSV="$REPO/data/datasets/medicaid-provider-spending.csv"
COHORT_CSV="$REPO/data/outputs/provider_cohorts.csv"
LABELS_CSV="$REPO/data/outputs/provider_labels.csv"
OUTPUT_CSV="$REPO/data/outputs/provider_month.csv"
DICTIONARY_CSV="$REPO/data/outputs/provider_month_data_dictionary.csv"
DICTIONARY_JSON="$REPO/provider_month_data_dictionary.json"

# Default to NY_individual if no args given
if [ $# -eq 0 ]; then
  COHORTS="NY_individual"
else
  COHORTS="$@"
fi

echo "Running cohorts: $COHORTS"

python "$REPO/scripts/create_provider_month_dataset.py" "$RAW_CSV" \
  --cohort-csv "$COHORT_CSV" \
  --cohorts $COHORTS \
  --labels-csv "$LABELS_CSV" \
  --output "$OUTPUT_CSV" \
  --dictionary "$DICTIONARY_CSV" \
  --dictionary-json "$DICTIONARY_JSON"

echo "Done: $OUTPUT_CSV"
