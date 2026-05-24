
"""
Poll ingestion and weighted-average builder for the Senate model.

This module intentionally uses a semi-automated workflow:
- You paste/export individual polls into inputs/polls_raw.csv.
- This script cleans them, computes poll weights, and updates race_inputs.csv.
- The core forecast engine then runs normally.

This avoids brittle scraping while still removing most manual spreadsheet work.
"""

from __future__ import annotations

from pathlib import Path
import math
import pandas as pd


REQUIRED_POLL_COLUMNS = {
    "state",
    "pollster",
    "start_date",
    "end_date",
    "dem_candidate",
    "gop_candidate",
    "dem_share",
    "gop_share",
}


def _ensure_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_POLL_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"polls_raw.csv missing required columns: {sorted(missing)}")


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    if out["end_date"].isna().any():
        bad = out.loc[out["end_date"].isna()]
        raise ValueError(f"Could not parse some end_date values:\n{bad}")
    return out


def _sample_size_weight(n: float | int | None) -> float:
    if pd.isna(n) or n is None or float(n) <= 0:
        return 1.0
    # sqrt prevents huge samples from dominating too much
    return math.sqrt(float(n) / 800.0)


def _pollster_quality_weight(rating: str | None) -> float:
    if rating is None or pd.isna(rating):
        return 1.0
    r = str(rating).strip().upper()
    mapping = {
        "A+": 1.15,
        "A": 1.12,
        "A-": 1.08,
        "B+": 1.04,
        "B": 1.00,
        "B-": 0.96,
        "C+": 0.90,
        "C": 0.84,
        "C-": 0.78,
        "D": 0.65,
    }
    return mapping.get(r, 1.0)


def _mode_weight(mode: str | None) -> float:
    if mode is None or pd.isna(mode):
        return 1.0
    m = str(mode).lower()
    if "live" in m and "phone" in m:
        return 1.04
    if "online" in m:
        return 0.98
    if "ivr" in m or "robocall" in m:
        return 0.94
    return 1.0


def _sponsor_weight(sponsor: str | None) -> float:
    if sponsor is None or pd.isna(sponsor) or str(sponsor).strip() == "":
        return 1.0
    s = str(sponsor).lower()
    partisan_terms = ["dem", "republican", "gop", "pac", "campaign", "party"]
    if any(term in s for term in partisan_terms):
        return 0.85
    return 0.95


def compute_weighted_polling_averages(
    polls_path: str | Path,
    as_of_date: str | None = None,
    half_life_days: float = 21.0,
) -> pd.DataFrame:
    polls_path = Path(polls_path)
    polls = pd.read_csv(polls_path)
    _ensure_columns(polls)

    polls = _parse_dates(polls)
    as_of = pd.Timestamp(as_of_date) if as_of_date else polls["end_date"].max()

    # Optional columns
    if "sample_size" not in polls.columns:
        polls["sample_size"] = pd.NA
    if "pollster_rating" not in polls.columns:
        polls["pollster_rating"] = pd.NA
    if "mode" not in polls.columns:
        polls["mode"] = pd.NA
    if "sponsor" not in polls.columns:
        polls["sponsor"] = pd.NA

    polls["margin_dem"] = polls["dem_share"].astype(float) - polls["gop_share"].astype(float)
    polls["age_days"] = (as_of - polls["end_date"]).dt.days.clip(lower=0)

    # Exponential time decay
    polls["recency_weight"] = 0.5 ** (polls["age_days"] / float(half_life_days))
    polls["sample_weight"] = polls["sample_size"].apply(_sample_size_weight)
    polls["quality_weight"] = polls["pollster_rating"].apply(_pollster_quality_weight)
    polls["mode_weight"] = polls["mode"].apply(_mode_weight)
    polls["sponsor_weight"] = polls["sponsor"].apply(_sponsor_weight)

    polls["final_weight"] = (
        polls["recency_weight"]
        * polls["sample_weight"]
        * polls["quality_weight"]
        * polls["mode_weight"]
        * polls["sponsor_weight"]
    )

    grouped = []
    for state, g in polls.groupby("state", sort=True):
        total_weight = g["final_weight"].sum()
        if total_weight <= 0:
            weighted_margin = g["margin_dem"].mean()
        else:
            weighted_margin = (g["margin_dem"] * g["final_weight"]).sum() / total_weight

        grouped.append({
            "state": state,
            "polling_margin_dem": weighted_margin,
            "poll_count": len(g),
            "latest_poll_end_date": g["end_date"].max().date().isoformat(),
            "avg_poll_age_days": (g["age_days"] * g["final_weight"]).sum() / max(total_weight, 1e-9),
            "total_poll_weight": total_weight,
        })

    return pd.DataFrame(grouped)


def update_race_inputs_from_polls(
    input_dir: str | Path = "inputs",
    as_of_date: str | None = None,
    half_life_days: float = 21.0,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    race_path = input_dir / "race_inputs.csv"
    polls_path = input_dir / "polls_raw.csv"

    if not race_path.exists():
        raise FileNotFoundError(f"Missing {race_path}")
    if not polls_path.exists():
        raise FileNotFoundError(f"Missing {polls_path}")

    races = pd.read_csv(race_path)
    avgs = compute_weighted_polling_averages(polls_path, as_of_date, half_life_days)

    merged = races.merge(
        avgs[["state", "polling_margin_dem", "poll_count", "latest_poll_end_date", "avg_poll_age_days"]],
        on="state",
        how="left",
        suffixes=("", "_new"),
    )

    has_new = merged["polling_margin_dem_new"].notna()
    merged.loc[has_new, "polling_margin_dem"] = merged.loc[has_new, "polling_margin_dem_new"]

    for col in ["poll_count", "latest_poll_end_date", "avg_poll_age_days"]:
        if col not in merged.columns:
            continue

    drop_cols = [c for c in merged.columns if c.endswith("_new")]
    merged = merged.drop(columns=drop_cols)

    # Save a copy of the weighted averages for auditability.
    avgs.to_csv(input_dir / "polling_averages_generated.csv", index=False)
    merged.to_csv(race_path, index=False)
    return avgs
