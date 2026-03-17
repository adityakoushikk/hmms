"""
build_provider_cohorts.py
Cohort providers by exact (state, entity_type) match.
Output: npi, cohort_label, cohort only. Drops rows with UNKNOWN state or entity type.
State is normalized to official 2-letter codes only; invalid/N/A/ZZ dropped.
Primary taxonomy = code in slot where Healthcare Provider Primary Taxonomy Switch = 'Y';
fallback = first non-null code; else 'UNKNOWN' (used only for filtering, not in output).
NPIs are sourced directly from the raw Medicaid billing CSV (BILLING_PROVIDER_NPI_NUM).
"""
import argparse
import sys
from pathlib import Path

import duckdb


# Official 2-letter state/territory codes (NPPES). ZZ (Foreign) excluded.
VALID_STATE_CODES = {
    "AK", "AL", "AR", "AS", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "FM", "GA", "GU", "HI",
    "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MH", "MI", "MN", "MO", "MP", "MS", "MT",
    "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "PR", "PW", "RI", "SC", "SD",
    "TN", "TX", "UT", "VA", "VI", "VT", "WA", "WI", "WV", "WY",
}

# Map NPPES raw state values to 2-letter code. Uppercase, trimmed input assumed.
STATE_NORMALIZE = {
    "ALASKA": "AK", "ALABAMA": "AL", "ARKANSAS": "AR", "AMERICAN SAMOA": "AS", "ARIZONA": "AZ",
    "CALIFORNIA": "CA", "CA - CALIFORNIA": "CA", "COLORADO": "CO", "CO- COLORADO": "CO", "CO - COLORADO": "CO",
    "CONNECTICUT": "CT", "DISTRICT OF COLUMBIA": "DC", "DELAWARE": "DE", "FLORIDA": "FL",
    "MICRONESIA, FEDERATED STATES OF": "FM", "GEORGIA": "GA", "GUAM": "GU", "HAWAII": "HI",
    "IOWA": "IA", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "KANSAS": "KS", "KENTUCKY": "KY",
    "LOUISIANA": "LA", "MASSACHUSETTS": "MA", "MARYLAND": "MD", "MD-MARYLAND": "MD", "MAINE": "ME",
    "MARSHALL ISLANDS": "MH", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSOURI": "MO",
    "MARIANA ISLANDS, NORTHERN": "MP", "NORTHERN MARIANA ISLANDS": "MP", "MISSISSIPPI": "MS", "MONTANA": "MT",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "NEBRASKA": "NE", "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEVADA": "NV", "NEW YORK": "NY", "OHIO": "OH",
    "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "PUERTO RICO": "PR", "P.R.": "PR",
    "PUESRTO RICO": "PR", "PALAU": "PW", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VIRGINIA": "VA", "VIRGIN ISLANDS": "VI",
    "VERMONT": "VT", "WASHINGTON": "WA", "WISCONSIN": "WI", "WEST VIRGINIA": "WV", "WYOMING": "WY",
}

NPPES_REQUIRED_COLUMNS = {
    "NPI",
    "Entity Type Code",
    "Provider Business Practice Location Address State Name",
    *[f"Healthcare Provider Taxonomy Code_{i}" for i in range(1, 16)],
    *[f"Healthcare Provider Primary Taxonomy Switch_{i}" for i in range(1, 16)],
}

MEDICAID_REQUIRED_COLUMNS = {"BILLING_PROVIDER_NPI_NUM"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build cohort output: one row per provider with npi, cohort_label, cohort only."
        )
    )
    parser.add_argument("--nppes_csv", required=True, help="Path to NPPES CSV (npidata_pfile_*.csv).")
    parser.add_argument("--medicaid_csv", required=True, help="Path to raw Medicaid billing CSV.")
    parser.add_argument("--output_csv", required=True, help="Path to write output CSV.")
    return parser.parse_args()


def validate_columns(
    con: duckdb.DuckDBPyConnection, csv_path: str, required: set[str], label: str
) -> None:
    preview_df = con.execute(
        "SELECT * FROM read_csv_auto(?, SAMPLE_SIZE=10000) LIMIT 0", [csv_path]
    ).fetchdf()
    missing = required.difference(preview_df.columns)
    if missing:
        sys.stderr.write(
            f"{label} CSV missing columns: {', '.join(sorted(missing))}\n"
        )
        raise SystemExit(1)


def run(nppes_csv: str, medicaid_csv: str, output_csv: str) -> None:
    con = duckdb.connect()
    try:
        con.execute("PRAGMA progress_bar = true;")
    except Exception:
        pass

    for path in (nppes_csv, medicaid_csv):
        if not Path(path).is_file():
            sys.stderr.write(f"Input not found: {path}\n")
            raise SystemExit(1)

    validate_columns(con, nppes_csv, NPPES_REQUIRED_COLUMNS, "NPPES")
    validate_columns(con, medicaid_csv, MEDICAID_REQUIRED_COLUMNS, "medicaid")

    # Primary taxonomy: slot where Primary Taxonomy Switch = 'Y'; else first non-null code; else 'UNKNOWN'
    primary_case = " ".join(
        [f"WHEN n.\"Healthcare Provider Primary Taxonomy Switch_{i}\" = 'Y' THEN TRIM(CAST(n.\"Healthcare Provider Taxonomy Code_{i}\" AS VARCHAR))"
         for i in range(1, 16)]
    )
    coalesce_codes = ", ".join(
        f"TRIM(CAST(n.\"Healthcare Provider Taxonomy Code_{i}\" AS VARCHAR))"
        for i in range(1, 16)
    )

    # Normalize NPPES state to 2-letter code: map full names/variants, keep valid 2-letter, else NULL (drop row)
    state_case_parts = [
        f"WHEN state_raw = '{k}' THEN '{v}'" for k, v in sorted(STATE_NORMALIZE.items())
    ]
    valid_list = ", ".join(f"'{c}'" for c in sorted(VALID_STATE_CODES))
    state_case_parts.append(f"WHEN state_raw IN ({valid_list}) THEN state_raw")
    state_normalize_case = "CASE " + " ".join(state_case_parts) + " ELSE NULL END"

    # Join provider_level to NPPES; normalize state; filter invalid/N/A/ZZ; assign cohort by (state, entity_type)
    out_sql = f"""
        WITH nppes AS (
            SELECT
                CAST(n."NPI" AS VARCHAR) AS npi,
                n."Entity Type Code" AS entity_type_code,
                UPPER(TRIM(NULLIF(n."Provider Business Practice Location Address State Name", ''))) AS state_raw,
                CASE {primary_case} ELSE NULL END AS primary_from_switch,
                COALESCE({coalesce_codes}) AS primary_fallback
            FROM read_csv_auto('{nppes_csv}') AS n
        ),
        with_state AS (
            SELECT
                npi,
                entity_type_code,
                ({state_normalize_case}) AS state,
                primary_from_switch,
                primary_fallback
            FROM nppes
            WHERE state_raw IS NOT NULL AND state_raw <> '' AND state_raw <> 'N/A'
        ),
        with_primary AS (
            SELECT
                npi,
                entity_type_code,
                state,
                COALESCE(primary_from_switch, primary_fallback, 'UNKNOWN') AS primary_taxonomy
            FROM with_state
            WHERE state IS NOT NULL
        ),
        filtered AS (
            SELECT *
            FROM with_primary
            WHERE entity_type_code IN ('1', '2')
              AND primary_taxonomy IS NOT NULL
              AND TRIM(primary_taxonomy) <> ''
              AND UPPER(TRIM(primary_taxonomy)) <> 'UNKNOWN'
        ),
        cohort_ids AS (
            SELECT
                state,
                entity_type_code,
                ROW_NUMBER() OVER (ORDER BY state, entity_type_code) AS cohort,
                state || '_' || CASE
                    WHEN entity_type_code = '1' THEN 'individual'
                    WHEN entity_type_code = '2' THEN 'organization'
                    ELSE 'unknown'
                END AS cohort_label
            FROM (SELECT DISTINCT state, entity_type_code FROM filtered)
        )
        SELECT
            CAST(pl.billing_provider_npi AS VARCHAR) AS npi,
            c.cohort_label,
            c.cohort
        FROM (SELECT DISTINCT CAST(BILLING_PROVIDER_NPI_NUM AS VARCHAR) AS billing_provider_npi FROM read_csv_auto('{medicaid_csv}')) pl
        INNER JOIN filtered f
            ON pl.billing_provider_npi = f.npi
        INNER JOIN cohort_ids c
            ON f.state = c.state AND f.entity_type_code = c.entity_type_code
    """

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({out_sql}) TO ? (HEADER, DELIMITER ',')", [str(out_path)])
    print(f"Wrote: {out_path}")


def main() -> None:
    args = parse_args()
    run(args.nppes_csv, args.medicaid_csv, args.output_csv)


if __name__ == "__main__":
    main()
