"""Build provider-level anomaly detection feature table from a provider-month CSV."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_CSV = "provider_level.csv"
MIN_MONTHS_DEFAULT = 12       # threshold for insufficient_history_flag; not a hard exclusion
DATE_CUTOFF = "2024-12-31"    # only months on or before this date are used
ROBUST_Z_WINDOW = 6           # prior-N-observed-rows baseline for robust z-score flagging
ROBUST_Z_THRESHOLD = 2.5      # |robust z| above this → flagged month
FLAG_RECENT_MONTHS = 6        # last-N-observed-rows window for recent flag count
MOM_MIN_DENOMINATOR = 10.0    # minimum prior paid_t before computing pct growth
HIGH_ENTROPY_CHANGE = 0.15    # monthlyized entropy change above this → high code-mix shift
HIGH_HHI_CHANGE = 0.05        # monthlyized HHI change above this → high concentration shift
HIGH_CONCENTRATION_HHI = 0.5  # hcpcs_hhi above this → "high concentration" month
CHANGEPOINT_PENALTY = 10.0    # PELT penalty (higher = fewer breakpoints)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _gap_months(months: pd.Series) -> np.ndarray:
    """
    Compute the calendar-month gap between each consecutive pair of observed months.

    Uses datetime64[M] arithmetic so the result is exact regardless of
    day-of-month alignment.

    Args:
        months: sorted pd.Series of datetime64 values (one per observed month).

    Returns:
        np.ndarray of length len(months)-1.  Entry i is the number of calendar
        months between months[i] and months[i+1].  A gap of 1 means truly
        adjacent calendar months; a gap of 3 means e.g. Jan → Apr.

    Example:
        [Jan-2023, Apr-2023, May-2023] → [3, 1]
    """
    m = months.to_numpy(dtype="datetime64[M]").astype(int)  # months since Unix epoch
    return np.diff(m).astype(float)


def _valid_obs_with_gaps(
    prov: pd.DataFrame,
    col: str = "paid_t",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract non-NaN observations of `col` and the calendar-month gaps between
    consecutive valid observations (aligned to the same month ordering as prov).

    Returns:
        values: 1-D array of non-NaN values, sorted by month.
        gaps:   1-D array of length len(values)-1.  Entry i is the calendar-month
                distance between valid observation i and i+1.  Empty if len(values) < 2.
    """
    mask = ~prov[col].isna()
    valid = prov[mask].sort_values("month")
    values = valid[col].to_numpy(dtype=float)
    gaps = _gap_months(valid["month"]) if len(values) > 1 else np.array([], dtype=float)
    return values, gaps


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (NaN-safe)."""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(np.median(np.abs(x - np.median(x))))


def _rolling_robust_z(series: pd.Series, window: int) -> pd.Series:
    """
    For each observed row t, compute the robust z-score of series[t] against
    the prior `window` OBSERVED rows (no look-ahead).  Returns NaN until a
    full window is available.

    NOTE: This is a prior-N-observed-row baseline, NOT a prior-N-calendar-month
    baseline.  For providers with irregular gaps, these concepts differ
    substantially.  Features derived from this are documented accordingly.
    """
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for i in range(window, len(values)):
        win = values[i - window: i]
        med = np.nanmedian(win)
        mad = float(np.nanmedian(np.abs(win - med)))
        if np.isnan(mad) or mad == 0:
            continue
        result[i] = (values[i] - med) / (1.4826 * mad)
    return pd.Series(result, index=series.index)


def _linear_slope(x: np.ndarray) -> float:
    """
    Slope of OLS regression of x on integer index (units: x-units per observed step,
    NOT per calendar month — observation spacing is irregular).
    """
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    t = np.arange(len(x), dtype=float)
    slope, *_ = stats.linregress(t, x)
    return float(slope)


# ──────────────────────────────────────────────────────────────────────────────
# Feature groups — each takes one provider's sorted DataFrame, returns a dict
# ──────────────────────────────────────────────────────────────────────────────
def compute_volume_scale_features(prov: pd.DataFrame) -> dict:
    paid = prov["paid_t"].to_numpy(dtype=float)
    return {
        "months_active": int(prov["month"].nunique()),
        "sum_paid":      float(np.nansum(paid)),
        "mean_paid":     float(np.nanmean(paid))   if len(paid) else np.nan,
        "median_paid":   float(np.nanmedian(paid)) if len(paid) else np.nan,
        "max_paid":      float(np.nanmax(paid))    if len(paid) else np.nan,
    }


def compute_unit_economics_features(prov: pd.DataFrame) -> dict:
    """
    Paid-per-claim summaries.

    - mean_paid_per_claim:           unweighted mean of monthly paid_per_claim_t.
                                     Treats every observed month equally regardless
                                     of volume.
    - median_paid_per_claim:         unweighted median.
    - std_paid_per_claim:            unweighted std.
    - claim_weighted_paid_per_claim: sum(paid_t) / sum(claims_t) over months where
                                     claims_t > 0.  More robust than the unweighted
                                     mean because high-volume months contribute
                                     proportionally to the rate.
    """
    ppc = prov["paid_per_claim_t"].to_numpy(dtype=float)

    if "claims_t" in prov.columns:
        valid = prov["claims_t"] > 0
        if valid.any():
            cwppc = float(prov.loc[valid, "paid_t"].sum() / prov.loc[valid, "claims_t"].sum())
        else:
            cwppc = np.nan
    else:
        cwppc = np.nan

    return {
        "mean_paid_per_claim":            float(np.nanmean(ppc))        if len(ppc) else np.nan,
        "median_paid_per_claim":          float(np.nanmedian(ppc))      if len(ppc) else np.nan,
        "std_paid_per_claim":             float(np.nanstd(ppc, ddof=1)) if len(ppc) > 1 else np.nan,
        "claim_weighted_paid_per_claim":  cwppc,
    }


def compute_volatility_features(prov: pd.DataFrame) -> dict:
    """
    Volatility of paid_t.

    - cv_paid, mad_paid: level-based metrics; not sensitive to observation gaps.
    - mean_abs_monthlyized_change_paid: mean |paid_{i+1} - paid_i| / gap_months_i
        over consecutive non-NaN observations, normalized by calendar distance.
        A jump across a 3-month gap is divided by 3 so it is comparable to a
        1-month-gap change.
    """
    paid = prov["paid_t"].to_numpy(dtype=float)
    clean = paid[~np.isnan(paid)]
    mean_p = float(np.nanmean(clean)) if len(clean) else np.nan
    std_p  = float(np.nanstd(clean, ddof=1)) if len(clean) > 1 else np.nan
    cv = float(std_p / mean_p) if (mean_p and mean_p != 0 and not np.isnan(std_p)) else np.nan

    values, gaps = _valid_obs_with_gaps(prov)
    if len(values) > 1 and len(gaps):
        safe_gaps = np.where(gaps <= 0, 1.0, gaps)
        adj_abs = np.abs(np.diff(values)) / safe_gaps
        mean_abs_monthlyized = float(np.mean(adj_abs))
    else:
        mean_abs_monthlyized = np.nan

    return {
        "cv_paid":                          cv,
        "mad_paid":                         _mad(paid),
        "mean_abs_monthlyized_change_paid": mean_abs_monthlyized,
    }


def compute_spike_drop_features(prov: pd.DataFrame) -> dict:
    """
    Spike, drop, and growth features.

    - spike_ratio_paid: max_paid / median_paid — cross-sectional ratio; not
        sensitive to observation order or gaps.
    - max_monthlyized_paid_growth: max monthlyized pct change over consecutive
        non-NaN observations:
            ((paid_{i+1} - paid_i) / max(paid_i, MOM_MIN_DENOMINATOR)) / gap_months_i
    - largest_monthlyized_paid_drop: largest monthlyized absolute drop:
            max(-adj_diff_i  for  adj_diff_i < 0)
        where adj_diff_i = (paid_{i+1} - paid_i) / gap_months_i.
    """
    paid = prov["paid_t"].to_numpy(dtype=float)
    clean = paid[~np.isnan(paid)]

    if len(clean) == 0:
        return {
            "spike_ratio_paid":              np.nan,
            "max_monthlyized_paid_growth":   np.nan,
            "largest_monthlyized_paid_drop": np.nan,
        }

    median_p = float(np.nanmedian(clean))
    max_p    = float(np.nanmax(clean))
    spike_ratio = float(max_p / median_p) if median_p > 0 else np.nan

    values, gaps = _valid_obs_with_gaps(prov)
    if len(values) < 2 or not len(gaps):
        return {
            "spike_ratio_paid":              spike_ratio,
            "max_monthlyized_paid_growth":   np.nan,
            "largest_monthlyized_paid_drop": np.nan,
        }

    safe_gaps = np.where(gaps <= 0, 1.0, gaps)
    diffs     = np.diff(values)
    adj_diffs = diffs / safe_gaps

    # Largest monthlyized absolute drop
    drops = -adj_diffs[adj_diffs < 0]
    largest_monthlyized_drop = float(drops.max()) if len(drops) else 0.0

    # Max monthlyized pct growth — gate denominator to avoid near-zero blow-up
    prior   = values[:-1]
    current = values[1:]
    gate    = prior >= MOM_MIN_DENOMINATOR
    if gate.any():
        adj_pct = (current[gate] - prior[gate]) / prior[gate] / safe_gaps[gate]
        max_growth = float(np.max(adj_pct))
    else:
        max_growth = np.nan

    return {
        "spike_ratio_paid":              spike_ratio,
        "max_monthlyized_paid_growth":   max_growth,
        "largest_monthlyized_paid_drop": largest_monthlyized_drop,
    }


def compute_changepoint_features(prov: pd.DataFrame) -> dict:
    """
    Structural break detection via PELT on the ordered observed paid_t sequence.

    IMPORTANT: PELT operates on observed rows in calendar order; calendar gaps
    between observations are not accounted for.  Changepoints mark breaks in
    the observed billing pattern, not in a continuous calendar series.

    - changepoint_count_paid:            number of PELT breakpoints.
    - largest_level_shift_paid:          abs(median(seg_{j+1}) - median(seg_j))
                                         over adjacent segment pairs.  This is a
                                         true absolute level shift in dollars.
    - largest_relative_level_shift_paid: largest_level_shift_paid / max(|median(seg_j)|, 1)
                                         Relative to the pre-shift segment.
    """
    empty = {
        "changepoint_count_paid":            np.nan,
        "largest_level_shift_paid":          np.nan,
        "largest_relative_level_shift_paid": np.nan,
    }

    if not RUPTURES_AVAILABLE:
        return empty

    values, _ = _valid_obs_with_gaps(prov)
    if len(values) < 6:
        return empty

    try:
        signal      = values.reshape(-1, 1)
        model       = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(signal)
        breakpoints = [b for b in model.predict(pen=CHANGEPOINT_PENALTY) if b < len(values)]
        count       = len(breakpoints)

        largest_abs_shift = np.nan
        largest_rel_shift = np.nan
        if breakpoints:
            segments   = [0] + breakpoints + [len(values)]
            abs_shifts = []
            rel_shifts = []
            for i in range(len(segments) - 2):
                pre_med  = float(np.median(values[segments[i]:segments[i + 1]]))
                post_med = float(np.median(values[segments[i + 1]:segments[i + 2]]))
                abs_shift = abs(post_med - pre_med)
                rel_shift = abs_shift / max(abs(pre_med), 1.0)
                abs_shifts.append(abs_shift)
                rel_shifts.append(rel_shift)
            largest_abs_shift = float(max(abs_shifts))
            largest_rel_shift = float(max(rel_shifts))
    except Exception:
        return empty

    return {
        "changepoint_count_paid":            count,
        "largest_level_shift_paid":          largest_abs_shift,
        "largest_relative_level_shift_paid": largest_rel_shift,
    }


def compute_flagged_month_features(prov: pd.DataFrame) -> dict:
    """
    Flag anomalous observed months using a rolling robust z-score.

    Baseline: prior ROBUST_Z_WINDOW OBSERVED rows (not calendar months).
    For irregular data, "prior 6 observed months" and "prior 6 calendar months"
    are distinct concepts.  All features here are documented as observed-row
    counts, not calendar-month counts.

    Features:
    - num_flagged_months:             total observed months with |robust_z| > threshold.
    - fraction_flagged_months:        num_flagged_months / months_active.
    - num_flagged_last_6_obs:         flagged count in the last 6 OBSERVED rows.
    - max_consecutive_flagged_obs:    longest run of consecutively flagged observed rows.
    """
    paid     = prov["paid_t"].reset_index(drop=True)
    robust_z = _rolling_robust_z(paid, ROBUST_Z_WINDOW)
    flagged  = robust_z.abs() > ROBUST_Z_THRESHOLD

    # Longest consecutive run over observed rows
    max_consec, cur = 0, 0
    for f in flagged:
        if f:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    n = len(flagged)
    recent = int(flagged.iloc[-FLAG_RECENT_MONTHS:].sum()) if n >= FLAG_RECENT_MONTHS else int(flagged.sum())

    return {
        "num_flagged_months":          int(flagged.sum()),
        "fraction_flagged_months":     float(flagged.mean()) if n else np.nan,
        "num_flagged_last_6_obs":      recent,
        "max_consecutive_flagged_obs": max_consec,
    }


def compute_code_mix_summary_features(prov: pd.DataFrame) -> dict:
    """
    Code-mix summaries.  Change features for entropy and HHI are gap-aware:
    raw differences between consecutive observed rows are divided by
    gap_months so they represent per-calendar-month rates.
    """
    out: dict = {}

    # ── top-code paid share ────────────────────────────────────────────────────
    if "top_code_paid_share" in prov.columns:
        tcps = prov["top_code_paid_share"].to_numpy(dtype=float)
        out["mean_top_code_paid_share"] = float(np.nanmean(tcps))
        out["max_top_code_paid_share"]  = float(np.nanmax(tcps)) if len(tcps) else np.nan
    else:
        out["mean_top_code_paid_share"] = np.nan
        out["max_top_code_paid_share"]  = np.nan

    # ── top-3 paid share ──────────────────────────────────────────────────────
    if "top_3_code_paid_share" in prov.columns:
        t3 = prov["top_3_code_paid_share"].to_numpy(dtype=float)
        out["mean_top_3_code_paid_share"]       = float(np.nanmean(t3))
        out["max_top_3_code_paid_share"]        = float(np.nanmax(t3)) if len(t3) else np.nan
        out["fraction_months_top3_above_80pct"] = float(np.nanmean(t3 > 0.80)) if len(t3) else np.nan
    else:
        out["mean_top_3_code_paid_share"]       = np.nan
        out["max_top_3_code_paid_share"]        = np.nan
        out["fraction_months_top3_above_80pct"] = np.nan

    # ── entropy ────────────────────────────────────────────────────────────────
    if "hcpcs_entropy" in prov.columns:
        ent = prov["hcpcs_entropy"].to_numpy(dtype=float)
        out["min_hcpcs_entropy"]  = float(np.nanmin(ent))  if len(ent) else np.nan
        out["mean_hcpcs_entropy"] = float(np.nanmean(ent)) if len(ent) else np.nan

        ent_vals, gaps = _valid_obs_with_gaps(prov, col="hcpcs_entropy")
        if len(ent_vals) > 1 and len(gaps):
            safe_gaps = np.where(gaps <= 0, 1.0, gaps)
            adj = np.abs(np.diff(ent_vals)) / safe_gaps
            out["max_monthlyized_entropy_change"]  = float(np.max(adj))
            out["mean_monthlyized_entropy_change"] = float(np.mean(adj))
            out["num_months_high_entropy_change"]  = int(np.sum(adj > HIGH_ENTROPY_CHANGE))
        else:
            out["max_monthlyized_entropy_change"]  = np.nan
            out["mean_monthlyized_entropy_change"] = np.nan
            out["num_months_high_entropy_change"]  = 0
    else:
        out.update({k: np.nan for k in (
            "min_hcpcs_entropy", "mean_hcpcs_entropy",
            "max_monthlyized_entropy_change", "mean_monthlyized_entropy_change",
        )})
        out["num_months_high_entropy_change"] = 0

    # ── HHI ────────────────────────────────────────────────────────────────────
    if "hcpcs_hhi" in prov.columns:
        hhi = prov["hcpcs_hhi"].to_numpy(dtype=float)
        out["mean_hcpcs_hhi"]                    = float(np.nanmean(hhi))
        out["max_hcpcs_hhi"]                     = float(np.nanmax(hhi)) if len(hhi) else np.nan
        out["fraction_months_high_concentration"] = float(np.nanmean(hhi > HIGH_CONCENTRATION_HHI))

        hhi_vals, gaps = _valid_obs_with_gaps(prov, col="hcpcs_hhi")
        if len(hhi_vals) > 1 and len(gaps):
            safe_gaps = np.where(gaps <= 0, 1.0, gaps)
            adj = np.abs(np.diff(hhi_vals)) / safe_gaps
            out["max_monthlyized_hhi_change"]  = float(np.max(adj))
            out["mean_monthlyized_hhi_change"] = float(np.mean(adj))
            out["num_months_high_hhi_change"]  = int(np.sum(adj > HIGH_HHI_CHANGE))
        else:
            out["max_monthlyized_hhi_change"]  = np.nan
            out["mean_monthlyized_hhi_change"] = np.nan
            out["num_months_high_hhi_change"]  = 0
    else:
        out.update({k: np.nan for k in (
            "mean_hcpcs_hhi", "max_hcpcs_hhi", "fraction_months_high_concentration",
            "max_monthlyized_hhi_change", "mean_monthlyized_hhi_change",
        )})
        out["num_months_high_hhi_change"] = 0

    return out


def compute_code_diversity_features(prov: pd.DataFrame) -> dict:
    """
    hcpcs_count_trend is slope per observed step (not per calendar month)
    because observation spacing is irregular.
    """
    if "hcpcs_count_t" not in prov.columns:
        return {k: np.nan for k in (
            "mean_hcpcs_count", "max_hcpcs_count", "std_hcpcs_count",
            "hcpcs_count_trend", "fraction_months_single_code",
        )}

    hc = prov["hcpcs_count_t"].to_numpy(dtype=float)
    return {
        "mean_hcpcs_count":            float(np.nanmean(hc)),
        "max_hcpcs_count":             float(np.nanmax(hc))         if len(hc) else np.nan,
        "std_hcpcs_count":             float(np.nanstd(hc, ddof=1)) if len(hc) > 1 else np.nan,
        "hcpcs_count_trend":           _linear_slope(hc),
        "fraction_months_single_code": float(np.mean(hc == 1.0))   if len(hc) else np.nan,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main collapse
# ──────────────────────────────────────────────────────────────────────────────
def build_provider_level(df: pd.DataFrame, min_months: int = MIN_MONTHS_DEFAULT) -> pd.DataFrame:
    """
    Collapse provider-month df to one row per provider with all feature groups.

    All providers are kept regardless of months_active.
    `insufficient_history_flag` = 1 when months_active < min_months.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.sort_values(["billing_provider_npi", "month"])

    rows = []
    for npi, prov in df.groupby("billing_provider_npi", sort=False):
        prov = prov.sort_values("month").reset_index(drop=True)
        row: dict = {"billing_provider_npi": npi}
        row.update(compute_volume_scale_features(prov))
        row.update(compute_unit_economics_features(prov))
        row.update(compute_volatility_features(prov))
        row.update(compute_spike_drop_features(prov))
        row.update(compute_changepoint_features(prov))
        row.update(compute_flagged_month_features(prov))
        row.update(compute_code_mix_summary_features(prov))
        row.update(compute_code_diversity_features(prov))
        row["insufficient_history_flag"] = int(row["months_active"] < min_months)
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────
def load_provider_month(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Provider-month CSV not found: {path}")
    df = pd.read_csv(p, low_memory=False)
    required = {"billing_provider_npi", "month", "paid_t", "claims_t", "paid_per_claim_t"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    return df


def run(
    input_csv: str,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    min_months: int = MIN_MONTHS_DEFAULT,
    date_cutoff: str = DATE_CUTOFF,
    filter_output: bool = True,
) -> pd.DataFrame:
    """
    Load, apply date cutoff, build features for ALL providers, then optionally
    filter the saved output to providers with months_active >= min_months.

    Args:
        filter_output: if True (default), drop rows with insufficient_history_flag=1
                       from the saved CSV.  Features are always built for all
                       providers before any filtering.
    """
    if not RUPTURES_AVAILABLE:
        print(
            "Warning: ruptures not installed — changepoint features will be NaN.\n"
            "Install with: pip install ruptures",
            file=sys.stderr,
        )

    df = load_provider_month(input_csv)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    # ── Date cutoff ───────────────────────────────────────────────────────────
    cutoff = pd.Timestamp(date_cutoff)
    before = len(df)
    df = df[df["month"] <= cutoff]
    print(f"Date cutoff {date_cutoff}: {before - len(df):,} rows removed, {len(df):,} remaining.")

    provider_level = build_provider_level(df, min_months=min_months)

    n_insufficient = int(provider_level["insufficient_history_flag"].sum())
    print(
        f"Providers built: {len(provider_level):,}  "
        f"({n_insufficient:,} with months_active < {min_months} → insufficient_history_flag=1)"
    )

    if filter_output:
        provider_level = provider_level[provider_level["insufficient_history_flag"] == 0].copy()
        print(f"After output filter: {len(provider_level):,} providers retained.")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provider_level.to_csv(out_path, index=False)
    return provider_level


def main():
    parser = argparse.ArgumentParser(
        description="Build provider-level anomaly detection features from a provider-month CSV."
    )
    parser.add_argument("input_csv", help="Path to provider-month CSV.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output path for provider-level CSV (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--min-months",
        type=int,
        default=MIN_MONTHS_DEFAULT,
        help=(
            f"Threshold for insufficient_history_flag (default: {MIN_MONTHS_DEFAULT}). "
            "Features are built for all providers; this only controls the flag value "
            "and optional output filtering."
        ),
    )
    parser.add_argument(
        "--date-cutoff",
        default=DATE_CUTOFF,
        help=f"Exclude months after this date (default: {DATE_CUTOFF}).",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        default=False,
        help=(
            "Keep all providers in the output CSV regardless of months_active. "
            "By default, providers with insufficient_history_flag=1 are dropped."
        ),
    )
    args = parser.parse_args()

    provider_level = run(
        args.input_csv,
        args.output,
        args.min_months,
        args.date_cutoff,
        filter_output=not args.no_filter,
    )

    print(f"\nprovider_level shape: {provider_level.shape}")
    print(f"Columns:              {list(provider_level.columns)}")
    print(f"\nHead:\n{provider_level.head()}")
    print(f"\nMissing value counts:\n{provider_level.isna().sum()}")
    print(f"\nOutput: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
