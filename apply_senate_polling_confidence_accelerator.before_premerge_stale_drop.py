from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

POLL_FILES = [
    INPUTS / "manual_polls_adjusted.csv",
    INPUTS / "manual_polls.csv",
]

BAYES_FILES = [
    INPUTS / "bayesian_update_generated.csv",
    OUTPUTS / "bayesian_update_generated.csv",
]

AUDIT_OUT = OUTPUTS / "senate_polling_confidence_accelerator_audit.csv"

RECENT_DAYS = 45

# Uniform accelerator rule.
BOOST_BY_RECENT_POLL_COUNT = {
    0: 0.00,
    1: 0.00,
    2: 0.03,
    3: 0.06,
}

BOOST_FOR_4_PLUS = 0.09

# Do not let early-summer polling take over the model.
PRE_LABOR_DAY_ABSOLUTE_CAP = 0.30

# After Labor Day, still allow the normal time curve to dominate, but don't let the
# accelerator itself create extreme behavior.
POST_LABOR_DAY_ACCELERATOR_ABSOLUTE_CAP = 0.45

LABOR_DAY_2026 = pd.Timestamp("2026-09-07")


def read_first_existing(paths):
    for path in paths:
        if path.exists():
            return path, pd.read_csv(path)
    raise FileNotFoundError(f"None of these files exist: {[str(p) for p in paths]}")


def pick_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def safe_num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def normalize_state(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def extract_state_from_race(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()

    # Common race formats: "ME", "ME-SEN", "ME Senate", "SEN-ME".
    if len(text) == 2 and text.isalpha():
        return text

    parts = (
        text.replace("_", "-")
        .replace("/", "-")
        .replace(" ", "-")
        .split("-")
    )

    for part in parts:
        part = part.strip()
        if len(part) == 2 and part.isalpha():
            return part

    return ""


def build_poll_counts(polls):
    state_col = pick_col(polls, ["state", "State"])
    race_col = pick_col(polls, ["race", "Race", "contest", "Contest"])

    if state_col:
        polls["state_norm"] = polls[state_col].apply(normalize_state)
    elif race_col:
        polls["state_norm"] = polls[race_col].apply(extract_state_from_race)
    else:
        raise ValueError("Manual polls file must contain either state or race column.")

    end_col = pick_col(
        polls,
        [
            "end_date",
            "poll_end_date",
            "field_end",
            "date",
            "poll_date",
        ],
    )

    if end_col is None:
        raise ValueError("Manual polls file must contain an end_date-like column.")

    polls["poll_end_dt"] = pd.to_datetime(polls[end_col], errors="coerce")

    sample_col = pick_col(
        polls,
        [
            "sample_size",
            "n",
            "Sample Size",
            "sample",
        ],
    )

    if sample_col:
        polls["sample_size_num"] = safe_num(polls[sample_col], 0.0)
    else:
        polls["sample_size_num"] = 1.0

    # Keep only usable dated state polls.
    usable = polls[
        polls["state_norm"].astype(str).str.len().eq(2)
        & polls["poll_end_dt"].notna()
        & polls["sample_size_num"].gt(0)
    ].copy()

    if usable.empty:
        return pd.DataFrame(
            columns=[
                "state",
                "recent_poll_count_45d",
                "polling_confidence_boost",
                "most_recent_poll_end_date",
            ]
        )

    today = pd.Timestamp(datetime.now().date())
    cutoff = today - pd.Timedelta(days=RECENT_DAYS)

    recent = usable[usable["poll_end_dt"] >= cutoff].copy()

    grouped = (
        recent.groupby("state_norm")
        .agg(
            recent_poll_count_45d=("state_norm", "size"),
            most_recent_poll_end_date=("poll_end_dt", "max"),
        )
        .reset_index()
        .rename(columns={"state_norm": "state"})
    )

    def boost_for_count(n):
        n = int(n)
        if n >= 4:
            return BOOST_FOR_4_PLUS
        return BOOST_BY_RECENT_POLL_COUNT.get(n, 0.0)

    grouped["polling_confidence_boost"] = grouped["recent_poll_count_45d"].apply(boost_for_count)

    grouped["most_recent_poll_end_date"] = grouped["most_recent_poll_end_date"].dt.strftime("%Y-%m-%d")

    return grouped


def identify_state_col(df):
    col = pick_col(df, ["state", "State"])
    if col:
        return col

    race_col = pick_col(df, ["race", "Race", "contest", "Contest"])
    if race_col:
        df["state"] = df[race_col].apply(extract_state_from_race)
        return "state"

    raise ValueError("Bayesian update file must contain state or race column.")


def identify_weight_column(df):
    """
    Always start from the base capped Bayesian polling weight.

    This script may be run repeatedly in the pipeline, so it must not select
    previously generated accelerator columns as inputs.
    """
    preferred = [
        "bayesian_polling_weight_capped",
        "bayesian_polling_weight",
        "original_bayesian_polling_weight",
    ]

    for col in preferred:
        if col in df.columns:
            return col

    candidates = []
    for col in df.columns:
        low = col.lower()
        if "after_polling_confidence_accelerator" in low:
            continue
        if "before_polling_confidence_accelerator" in low:
            continue
        if "polling_confidence_weight_change" in low:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue

        if "poll" in low and "weight" in low:
            candidates.append(col)
        elif "bayes" in low and "weight" in low:
            candidates.append(col)

    if not candidates:
        raise ValueError(
            "Could not identify a base polling/Bayesian weight column to adjust. "
            "Expected bayesian_polling_weight_capped or similar."
        )

    return candidates[0]

def main():
    poll_path, polls = read_first_existing(POLL_FILES)
    bayes_path, bayes = read_first_existing(BAYES_FILES)

    print(f"Using polls file: {poll_path}")
    print(f"Using Bayesian file: {bayes_path}")

    poll_counts = build_poll_counts(polls)

    state_col = identify_state_col(bayes)
    bayes["state_norm"] = bayes[state_col].apply(normalize_state)

    weight_col = identify_weight_column(bayes)
    weight_cols = [weight_col]

    today = pd.Timestamp(datetime.now().date())
    absolute_cap = (
        PRE_LABOR_DAY_ABSOLUTE_CAP
        if today < LABOR_DAY_2026
        else POST_LABOR_DAY_ACCELERATOR_ABSOLUTE_CAP
    )

    before_col = f"{weight_col}_before_polling_confidence_accelerator"
    bayes[before_col] = safe_num(bayes[weight_col], 0.0)

    merged = bayes.merge(
        poll_counts,
        left_on="state_norm",
        right_on="state",
        how="left",
        suffixes=("", "_poll_counts"),
    )

    merged["recent_poll_count_45d"] = safe_num(merged["recent_poll_count_45d"], 0).astype(int)
    merged["polling_confidence_boost"] = safe_num(merged["polling_confidence_boost"], 0.0)
    merged["polling_confidence_absolute_cap"] = absolute_cap

    merged[f"{weight_col}_after_polling_confidence_accelerator"] = (
        merged[before_col] + merged["polling_confidence_boost"]
    ).clip(lower=0.0, upper=absolute_cap)

    merged["polling_confidence_weight_change"] = (
        merged[f"{weight_col}_after_polling_confidence_accelerator"] - merged[before_col]
    )

    # Write adjusted value back into the model-driving weight column.
    merged[weight_col] = merged[f"{weight_col}_after_polling_confidence_accelerator"]

    # Recompute capped Bayesian model margin if the needed columns are present.
    prior_col = pick_col(merged, ["prior_margin_dem", "fundamentals_margin_dem", "model_prior_margin_dem"])
    poll_margin_col = pick_col(merged, ["polling_margin_used", "polling_margin_dem", "polling_average_dem_margin"])
    capped_margin_col = pick_col(merged, ["bayesian_model_margin_dem_capped", "bayesian_model_margin_dem"])

    if prior_col and poll_margin_col and capped_margin_col:
        before_margin_col = f"{capped_margin_col}_before_polling_confidence_accelerator"
        merged[before_margin_col] = safe_num(merged[capped_margin_col], 0.0)

        prior = safe_num(merged[prior_col], 0.0)
        poll_margin = safe_num(merged[poll_margin_col], 0.0)
        weight = safe_num(merged[weight_col], 0.0).clip(lower=0.0, upper=1.0)

        merged[capped_margin_col] = prior * (1.0 - weight) + poll_margin * weight
        merged["polling_confidence_margin_change_dem"] = (
            merged[capped_margin_col] - merged[before_margin_col]
        )

        fund_weight_col = pick_col(
            merged,
            ["bayesian_fundamentals_weight_capped", "bayesian_prior_weight_capped"]
        )
        if fund_weight_col:
            merged[fund_weight_col] = 1.0 - weight
    else:
        merged["polling_confidence_margin_change_dem"] = 0.0

    # Clean stale accelerator columns from prior runs so the script is idempotent.
    stale_cols = [
        c for c in merged.columns
        if (
            ("polling_confidence_accelerator" in c)
            or c in [
                "polling_confidence_weight_change",
                "polling_confidence_margin_change_dem",
                "polling_confidence_boost",
                "polling_confidence_absolute_cap",
                "recent_poll_count_45d",
                "most_recent_poll_end_date",
            ]
        )
    ]

    keep_cols = {
        before_col,
        f"{weight_col}_after_polling_confidence_accelerator",
        "polling_confidence_weight_change",
        "polling_confidence_margin_change_dem",
        "polling_confidence_boost",
        "polling_confidence_absolute_cap",
        "recent_poll_count_45d",
        "most_recent_poll_end_date",
    }

    for possible_margin_col in [
        "bayesian_model_margin_dem_capped_before_polling_confidence_accelerator",
        "bayesian_model_margin_dem_before_polling_confidence_accelerator",
    ]:
        if possible_margin_col in merged.columns:
            keep_cols.add(possible_margin_col)

    stale_cols = [c for c in stale_cols if c not in keep_cols]

    drop_cols = ["state_poll_counts"] + stale_cols
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")

    merged.to_csv(bayes_path, index=False)

    audit_cols = [
        "state_norm",
        "recent_poll_count_45d",
        "most_recent_poll_end_date",
        "polling_confidence_boost",
        "polling_confidence_absolute_cap",
        before_col,
        f"{weight_col}_after_polling_confidence_accelerator",
        "polling_confidence_weight_change",
        "polling_confidence_margin_change_dem",
    ]

    for extra_col in [
        "bayesian_model_margin_dem_capped_before_polling_confidence_accelerator",
        "bayesian_model_margin_dem_before_polling_confidence_accelerator",
        "bayesian_model_margin_dem_capped",
        "bayesian_model_margin_dem",
    ]:
        if extra_col in merged.columns and extra_col not in audit_cols:
            audit_cols.append(extra_col)
    audit_cols = [c for c in audit_cols if c in merged.columns]

    audit = (
        merged[audit_cols]
        .drop_duplicates("state_norm")
        .sort_values(["polling_confidence_weight_change", "recent_poll_count_45d"], ascending=[False, False])
    )

    audit.to_csv(AUDIT_OUT, index=False)

    print()
    print(f"Base Bayesian polling weight column adjusted: {weight_col}")
    print(f"Detected possible weight columns: {weight_cols}")
    print(f"Wrote audit to {AUDIT_OUT}")
    print()
    print("Largest polling-confidence boosts:")
    print(audit.head(25).to_string(index=False))

    print()
    print("Maine:")
    me = audit[audit["state_norm"].eq("ME")]
    print(me.to_string(index=False) if not me.empty else "No ME row in audit.")

    print()
    print("Georgia:")
    ga = audit[audit["state_norm"].eq("GA")]
    print(ga.to_string(index=False) if not ga.empty else "No GA row in audit.")


if __name__ == "__main__":
    main()
