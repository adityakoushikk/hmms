"""
generate_anomaly_report.py

Generates a self-contained anomaly report folder for a single NPI, mirroring
the analysis in notebooks/inspectTopAnomalies.ipynb.

Output structure:
    anomalyReport/{TARGET_NPI}/
        provider_info.json          — NPPES identity + model output (anomaly score, LEIE label, percentile)
        feature_comparison.csv      — per-feature: actual value, peer median, recon error, peer median recon error
        graphs/
            001_{feature}.png       — time-series plot for each provider-month feature (worst recon first)
        README.md                   — explanation for a downstream fraud/abuse/waste review agent
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Parse arguments ───────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description="Generate anomaly report for a single NPI.")
parser.add_argument("target_npi", help="NPI to inspect (must exist in scored_providers.csv)")
parser.add_argument("--nppes-csv",          default=str(REPO / "data/datasets/nppes.csv"))
parser.add_argument("--provider-month-csv", default=str(REPO / "data/outputs/provider_month_NV_organization.csv"))
parser.add_argument("--scored-csv",         default=str(REPO / "data/outputs/scored_providers.csv"))
parser.add_argument("--feat-errors-csv",    default=str(REPO / "data/outputs/provider_feature_errors.csv"))
parser.add_argument("--provider-level-csv", default=str(REPO / "data/outputs/provider_level.csv"))
args = parser.parse_args()

TARGET_NPI            = args.target_npi
NPPES_CSV             = args.nppes_csv
PROVIDER_MONTH_CSV    = args.provider_month_csv
SCORED_CSV            = args.scored_csv
FEAT_ERRORS_CSV       = args.feat_errors_csv
PROVIDER_LEVEL_DATSET = args.provider_level_csv

# ── Output directories ────────────────────────────────────────────────────────
REPORT_DIR = REPO / "anomalyReport" / TARGET_NPI
GRAPHS_DIR = REPORT_DIR / "graphs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Report dir: {REPORT_DIR}")

# ── Shared SQL snippet: resolve primary taxonomy across 15 NPPES slots ────────
# NULL if no switch is 'Y' — avoids silently falling back to slot 1 for providers
# with no primary taxonomy set.
TAXONOMY_CASE = """
    CASE
        WHEN "Healthcare Provider Primary Taxonomy Switch_1"  = 'Y' THEN "Healthcare Provider Taxonomy Code_1"
        WHEN "Healthcare Provider Primary Taxonomy Switch_2"  = 'Y' THEN "Healthcare Provider Taxonomy Code_2"
        WHEN "Healthcare Provider Primary Taxonomy Switch_3"  = 'Y' THEN "Healthcare Provider Taxonomy Code_3"
        WHEN "Healthcare Provider Primary Taxonomy Switch_4"  = 'Y' THEN "Healthcare Provider Taxonomy Code_4"
        WHEN "Healthcare Provider Primary Taxonomy Switch_5"  = 'Y' THEN "Healthcare Provider Taxonomy Code_5"
        WHEN "Healthcare Provider Primary Taxonomy Switch_6"  = 'Y' THEN "Healthcare Provider Taxonomy Code_6"
        WHEN "Healthcare Provider Primary Taxonomy Switch_7"  = 'Y' THEN "Healthcare Provider Taxonomy Code_7"
        WHEN "Healthcare Provider Primary Taxonomy Switch_8"  = 'Y' THEN "Healthcare Provider Taxonomy Code_8"
        WHEN "Healthcare Provider Primary Taxonomy Switch_9"  = 'Y' THEN "Healthcare Provider Taxonomy Code_9"
        WHEN "Healthcare Provider Primary Taxonomy Switch_10" = 'Y' THEN "Healthcare Provider Taxonomy Code_10"
        WHEN "Healthcare Provider Primary Taxonomy Switch_11" = 'Y' THEN "Healthcare Provider Taxonomy Code_11"
        WHEN "Healthcare Provider Primary Taxonomy Switch_12" = 'Y' THEN "Healthcare Provider Taxonomy Code_12"
        WHEN "Healthcare Provider Primary Taxonomy Switch_13" = 'Y' THEN "Healthcare Provider Taxonomy Code_13"
        WHEN "Healthcare Provider Primary Taxonomy Switch_14" = 'Y' THEN "Healthcare Provider Taxonomy Code_14"
        WHEN "Healthcare Provider Primary Taxonomy Switch_15" = 'Y' THEN "Healthcare Provider Taxonomy Code_15"
    END
"""

# ── Load all data ─────────────────────────────────────────────────────────────
con = duckdb.connect()

scored = pd.read_csv(SCORED_CSV)
scored["billing_provider_npi"] = scored["billing_provider_npi"].astype(str)

feat_err = pd.read_csv(FEAT_ERRORS_CSV)
feat_err["billing_provider_npi"] = feat_err["billing_provider_npi"].astype(str)

prov_level = pd.read_csv(PROVIDER_LEVEL_DATSET)
prov_level["billing_provider_npi"] = prov_level["billing_provider_npi"].astype(str)

pm = con.execute(f"SELECT * FROM read_csv_auto('{PROVIDER_MONTH_CSV}', ignore_errors=true)").df()
pm["billing_provider_npi"] = pm["billing_provider_npi"].astype(str)
pm["month"] = pd.to_datetime(pm["month"])

# ── Derive shared metadata ────────────────────────────────────────────────────
row = scored[scored["billing_provider_npi"] == TARGET_NPI]
if row.empty:
    sys.exit(f"ERROR: NPI {TARGET_NPI} not found in scored_providers.csv")
target_row = row.iloc[0]

is_leie = int(target_row["label"]) == 1
score   = float(target_row["anomaly_score"])
pct     = (scored["anomaly_score"] < score).mean() * 100

ERR_META     = {"billing_provider_npi", "anomaly_score", "label"}
err_features = [c for c in feat_err.columns if c not in ERR_META]

PM_META      = {"billing_provider_npi", "month", "label", "excldate", "revocation_rsn"}
pm_features  = [c for c in pm.columns if c not in PM_META]

def map_to_pm_feature(pl_feat):
    candidates = [f for f in pm_features if pl_feat.startswith(f + "_") or pl_feat == f]
    return max(candidates, key=len) if candidates else None

pl_to_pm = {f: map_to_pm_feature(f) for f in err_features}

npi_pm    = pm[pm["billing_provider_npi"] == TARGET_NPI]
_raw      = npi_pm["excldate"].dropna()
excldates = pd.to_datetime(
    _raw.astype(float).astype(int).astype(str), format="%Y%m%d", errors="coerce"
).dropna()
excldate = excldates.iloc[0] if not excldates.empty else None

_rev_raw = npi_pm["revocation_rsn"].dropna()
revocation_rsn = str(_rev_raw.iloc[0]).strip() if not _rev_raw.empty else None
if revocation_rsn:
    with open(REPORT_DIR / "revocation_reason.txt", "w") as f:
        f.write(revocation_rsn)
    print(f"  Saved revocation_reason.txt")

print(f"NPI: {TARGET_NPI}  |  score: {score:.4f}  |  label: {int(is_leie)}")

# ── Step 1: NPPES full record for TARGET_NPI → provider_info.json ─────────────
print("\n[1/3] Fetching NPPES record...")

raw = con.execute(f"""
    SELECT *, {TAXONOMY_CASE} AS primary_taxonomy_code
    FROM read_csv_auto('{NPPES_CSV}', ignore_errors=true)
    WHERE CAST("NPI" AS VARCHAR) = '{TARGET_NPI}'
""").df()

if raw.empty:
    sys.exit(f"ERROR: NPI {TARGET_NPI} not found in NPPES")

first_row = raw.iloc[0]
keep_cols = [c for c in raw.columns if pd.notna(first_row[c]) and str(first_row[c]).strip() != ""]
target_nppes = raw[keep_cols].copy()
target_nppes["NPI"] = target_nppes["NPI"].astype(str)
target_nppes["anomaly_score"]      = round(score, 4)
target_nppes["anomaly_percentile"] = round(pct, 1)
target_nppes["leie_label"]         = int(is_leie)

# Extract primary taxonomy for peer resolution (done once, reused below)
_tax_raw = target_nppes.iloc[0].get("primary_taxonomy_code")
target_taxonomy = None if (pd.isna(_tax_raw) or str(_tax_raw).strip() == "") else str(_tax_raw).strip()

info_dict = {}
for k, v in target_nppes.iloc[0].items():
    if pd.isna(v):
        info_dict[k] = None
    elif isinstance(v, np.integer):
        info_dict[k] = int(v)
    elif isinstance(v, np.floating):
        info_dict[k] = float(v)
    else:
        info_dict[k] = str(v)

with open(REPORT_DIR / "provider_info.json", "w") as f:
    json.dump(info_dict, f, indent=2)
print(f"  Saved provider_info.json  ({len(info_dict)} fields)")

# ── Step 2: peer resolution + feature comparison → feature_comparison.csv ────
print("\n[2/3] Building feature comparison table...")

# Single NPPES scan: resolve primary taxonomy for every provider in feat_err
con.register("feat_err_duck", feat_err)
feat_with_tax = con.execute(f"""
    SELECT f.billing_provider_npi,
           {TAXONOMY_CASE} AS primary_taxonomy_code
    FROM feat_err_duck f
    LEFT JOIN read_csv_auto('{NPPES_CSV}', ignore_errors=true) n
        ON CAST(n."NPI" AS VARCHAR) = f.billing_provider_npi
""").df()
feat_with_tax["billing_provider_npi"] = feat_with_tax["billing_provider_npi"].astype(str)

if target_taxonomy:
    peer_mask = (
        feat_with_tax["primary_taxonomy_code"].notna() &
        (feat_with_tax["primary_taxonomy_code"] == target_taxonomy) &
        (feat_with_tax["billing_provider_npi"] != TARGET_NPI)
    )
    peer_npis = set(feat_with_tax.loc[peer_mask, "billing_provider_npi"])
else:
    peer_npis = set()

print(f"  Taxonomy: {target_taxonomy or 'N/A'}  |  {len(peer_npis)} peers")

target_prov_row = prov_level[prov_level["billing_provider_npi"] == TARGET_NPI]
if target_prov_row.empty:
    sys.exit(f"ERROR: NPI {TARGET_NPI} not found in PROVIDER_LEVEL_DATSET")

target_err_row = feat_err[feat_err["billing_provider_npi"] == TARGET_NPI][err_features]
if target_err_row.empty:
    sys.exit(f"ERROR: NPI {TARGET_NPI} not found in feat_err")

peers_prov = prov_level[prov_level["billing_provider_npi"].isin(peer_npis)]
peers_err  = feat_err[feat_err["billing_provider_npi"].isin(peer_npis)][err_features]

feat_comparison = pd.DataFrame({
    "feature_value":     target_prov_row.iloc[0].reindex(err_features),
    "peer_median_value": peers_prov[err_features].median() if len(peers_prov) > 0 else pd.Series(float("nan"), index=err_features),
    "recon_error":       target_err_row.iloc[0],
    "peer_median_recon": peers_err.median() if len(peers_err) > 0 else pd.Series(float("nan"), index=err_features),
}).round(4).sort_values("recon_error", ascending=False)
feat_comparison.index.name = "feature"

feat_comparison.to_csv(REPORT_DIR / "feature_comparison.csv")
print(f"  Saved feature_comparison.csv  ({len(feat_comparison)} features)")

# ── Step 3: provider-month feature graphs → graphs/ ───────────────────────────
print("\n[3/3] Generating feature graphs...")

def get_group_stats(npi):
    err_row = feat_err[feat_err["billing_provider_npi"] == npi]
    if err_row.empty:
        return {}
    err_vals = err_row[err_features].iloc[0]
    group_sum, group_count = {}, {}
    for pl_feat, err in err_vals.items():
        pm_feat = pl_to_pm.get(pl_feat)
        if pm_feat and pm_feat in pm.columns:
            group_sum[pm_feat]   = group_sum.get(pm_feat, 0.0) + err
            group_count[pm_feat] = group_count.get(pm_feat, 0) + 1
    stats = {f: {"mean_err": group_sum[f] / group_count[f]} for f in group_sum}
    return dict(sorted(stats.items(), key=lambda x: x[1]["mean_err"], reverse=True))

prov        = npi_pm.sort_values("month")
group_stats = get_group_stats(TARGET_NPI)

if not group_stats:
    print("  WARNING: no feature mapping found, skipping graphs")
else:
    color = "#dc2626" if is_leie else "#2563eb"
    parts = []
    if is_leie:
        parts.append("LEIE EXCLUDED")
    if excldate is not None:
        parts.append(f"excl: {excldate.date()}")

    for idx, (pm_feat, stats) in enumerate(group_stats.items(), start=1):
        subtitle = "  ·  ".join(parts + [f"group recon error: {stats['mean_err']:.4f}"])

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        plt.subplots_adjust(top=0.78)

        if not prov.empty and pm_feat in prov.columns:
            vals = prov[pm_feat]
            ax.fill_between(prov["month"], vals, alpha=0.18, color=color)
            ax.plot(prov["month"], vals, color=color, lw=2.8, zorder=3)
            ax.scatter(prov["month"], vals, s=60, color=color, zorder=4,
                       edgecolors="white", linewidths=1.5)

        fig.suptitle(pm_feat, fontsize=22, fontweight="bold", x=0.05, ha="left", y=0.97)
        ax.set_title(subtitle, fontsize=13, color="#6b7280", loc="left", pad=10)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", rotation=45, labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_xlabel("Month", fontsize=15)
        ax.set_ylabel("Value", fontsize=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--", color="#9ca3af")

        plt.tight_layout(rect=[0, 0, 1, 0.78])
        fig.savefig(str(GRAPHS_DIR / f"{idx:03d}_{pm_feat}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {idx:03d}  {pm_feat}  (recon err: {stats['mean_err']:.4f})")

    print(f"  Saved {len(group_stats)} graphs to graphs/")

# ── Step 4: README for downstream agent ──────────────────────────────────────
shutil.copy(REPORT_DIR.parent / "README.md", REPORT_DIR / "README.md")
print(f"\nREADME.md copied.")
print(f"\nReport complete: {REPORT_DIR}")
