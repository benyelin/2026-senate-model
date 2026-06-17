from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

POLL_FILE_CANDIDATES = [
    INPUTS / "manual_polls_adjusted.csv",
    INPUTS / "manual_polls.csv",
]

BAYES_FILE = INPUTS / "bayesian_update_generated.csv"
RACE_INPUTS = INPUTS / "race_inputs.csv"
AUDIT_OUT = OUTPUTS / "senate_polling_confidence_accelerator_audit.csv"

RECENT_DAYS = 45
LABOR_DAY_2026 = pd.Timestamp("2026-09-07")

PRE_LABOR_DAY_ABSOLUTE_CAP = 0.30
POST_LABOR_DAY_ABSOLUTE_CAP = 0.45


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


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files exist: {[str(p) for p in paths]}")


def pick_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def recent_poll_counts(polls):
    state_col = pick_col(polls, ["state", "State"])
    race_col = pick_col(polls, ["race", "Race", "contest", "Contest"])
    end_col = pick_col(polls, ["end_date", "poll_end_date", "field_end", "date", "poll_date"])
    sample_col = pick_col(polls, ["sample_size", "n", "Sample Size", "sample"])

    if state_col:
        polls["state_norm"] = polls[state_col].apply(normalize_state)
    elif race_col:
        polls["state_norm"] = polls[race_col].apply(extract_state_from_race)
    else:
        raise ValueError("Poll file must contain state or race column.")

    if end_col is None:
        raise ValueError("Poll file must contain end_date-like column.")

    polls["poll_end_dt"] = pd.to_datetime(polls[end_col], errors="coerce")

    if sample_col:
        polls["sample_size_num"] = safe_num(polls[sample_col], 0.0)
    else:
        polls["sample_size_num"] = 1.0

    usable = polls[
        polls["state_norm"].astype(str).str.len().eq(2)
        & polls["poll_end_dt"].notna()
        & polls["sample_size_num"].gt(0)
    ].copy()

    if usable.empty:
        return pd.DataFrame(columns=[
            "state",
            "recent_poll_count_45d",
            "most_recent_poll_end_date",
            "polling_confidence_boost",
        ])

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

    def boost(n):
        n = int(n)
        if n >= 4:
            return 0.09
        if n == 3:
            return 0.06
        if n == 2:
            return 0.03
        return 0.00

    grouped["polling_confidence_boost"] = grouped["recent_poll_count_45d"].apply(boost)
    grouped["most_recent_poll_end_date"] = grouped["most_recent_poll_end_date"].dt.strftime("%Y-%m-%d")

    return grouped


def drop_old_accelerator_cols(df):
    old_cols = [
        c for c in df.columns
        if (
            "polling_confidence" in c
            or "recent_poll_count_45d" in c
            or "most_recent_poll_end_date" in c
            or "after_polling_confidence_accelerator" in c
            or "before_polling_confidence_accelerator" in c
        )
    ]
    return df.drop(columns=old_cols, errors="ignore")


def main():
    if not BAYES_FILE.exists():
        raise FileNotFoundError(f"Missing {BAYES_FILE}")

    poll_path = first_existing(POLL_FILE_CANDIDATES)

    polls = pd.read_csv(poll_path)
    bayes = pd.read_csv(BAYES_FILE)

    print(f"Using polls file: {poll_path}")
    print(f"Using Bayesian file: {BAYES_FILE}")

    bayes = drop_old_accelerator_cols(bayes)

    if "state" not in bayes.columns:
        raise ValueError("Bayesian update file must contain state column.")

    required = [
        "prior_margin_dem",
        "polling_margin_used",
        "bayesian_polling_weight_capped",
        "bayesian_model_margin_dem_capped",
    ]

    missing = [c for c in required if c not in bayes.columns]
    if missing:
        raise ValueError(f"Bayesian file missing required columns: {missing}")

    counts = recent_poll_counts(polls)

    bayes["state_norm"] = bayes["state"].apply(normalize_state)

    merged = bayes.merge(
        counts,
        left_on="state_norm",
        right_on="state",
        how="left",
        suffixes=("", "_poll_counts"),
    )

    today = pd.Timestamp(datetime.now().date())
    absolute_cap = PRE_LABOR_DAY_ABSOLUTE_CAP if today < LABOR_DAY_2026 else POST_LABOR_DAY_ABSOLUTE_CAP

    merged["recent_poll_count_45d"] = safe_num(merged["recent_poll_count_45d"], 0).astype(int)
    merged["polling_confidence_boost"] = safe_num(merged["polling_confidence_boost"], 0.0)
    merged["polling_confidence_absolute_cap"] = absolute_cap

    base_weight = safe_num(merged["bayesian_polling_weight_capped"], 0.0).clip(lower=0.0, upper=1.0)

    merged["bayesian_polling_weight_capped_before_polling_confidence_accelerator"] = base_weight
    merged["bayesian_model_margin_dem_capped_before_polling_confidence_accelerator"] = safe_num(
        merged["bayesian_model_margin_dem_capped"], 0.0
    )

    accelerated_weight = (base_weight + merged["polling_confidence_boost"]).clip(
        lower=0.0,
        upper=absolute_cap,
    )

    prior = safe_num(merged["prior_margin_dem"], 0.0)
    poll_margin = safe_num(merged["polling_margin_used"], 0.0)

    accelerated_margin = prior * (1.0 - accelerated_weight) + poll_margin * accelerated_weight

    # Make the accelerated/capped values the canonical Bayesian values.
    # Several downstream scripts use different aliases, so keep them synchronized.
    merged["bayesian_polling_weight_capped"] = accelerated_weight
    merged["bayesian_polling_weight"] = accelerated_weight
    merged["bayesian_prior_weight"] = 1.0 - accelerated_weight
    merged["bayesian_fundamentals_weight_capped"] = 1.0 - accelerated_weight

    merged["bayesian_model_margin_dem_capped"] = accelerated_margin
    merged["bayesian_model_margin_dem"] = accelerated_margin
    merged["posterior_margin_dem"] = accelerated_margin
    merged["posterior_margin_dem_capped"] = accelerated_margin

    merged["bayesian_polling_weight_capped_after_polling_confidence_accelerator"] = accelerated_weight
    merged["polling_confidence_weight_change"] = (
        merged["bayesian_polling_weight_capped_after_polling_confidence_accelerator"]
        - merged["bayesian_polling_weight_capped_before_polling_confidence_accelerator"]
    )
    merged["polling_confidence_margin_change_dem"] = (
        merged["bayesian_model_margin_dem_capped"]
        - merged["bayesian_model_margin_dem_capped_before_polling_confidence_accelerator"]
    )

    # Keep Bayesian file clean but preserve useful audit fields.
    bayes_out = merged.drop(columns=["state_norm", "state_poll_counts"], errors="ignore")
    bayes_out.to_csv(BAYES_FILE, index=False)

    # Sync into race_inputs.csv canonical model fields.
    if RACE_INPUTS.exists():
        races = pd.read_csv(RACE_INPUTS)

        if "state" in races.columns:
            races = drop_old_accelerator_cols(races)
            races["state_norm"] = races["state"].apply(normalize_state)

            sync_cols = [
                "state_norm",
                "bayesian_polling_weight",
                "bayesian_polling_weight_capped",
                "bayesian_fundamentals_weight_capped",
                "bayesian_model_margin_dem",
                "bayesian_model_margin_dem_capped",
                "polling_confidence_boost",
                "polling_confidence_absolute_cap",
                "polling_confidence_weight_change",
                "polling_confidence_margin_change_dem",
                "recent_poll_count_45d",
                "most_recent_poll_end_date",
                "bayesian_polling_weight_capped_before_polling_confidence_accelerator",
                "bayesian_polling_weight_capped_after_polling_confidence_accelerator",
                "bayesian_model_margin_dem_capped_before_polling_confidence_accelerator",
            ]

            sync_cols = [c for c in sync_cols if c in merged.columns]

            sync = merged[sync_cols].drop_duplicates("state_norm")

            races = races.merge(
                sync,
                on="state_norm",
                how="left",
                suffixes=("", "_accelerated"),
            )

            for col in sync_cols:
                if col == "state_norm":
                    continue

                acc = f"{col}_accelerated"

                if acc in races.columns:
                    races[col] = races[acc].combine_first(
                        races[col] if col in races.columns else pd.Series(index=races.index, dtype=object)
                    )
                elif col in sync.columns and col not in races.columns:
                    # If merge introduced the column without suffix because it did not exist previously.
                    pass

            # Most important canonical model fields.
            for base_col in [
                "bayesian_polling_weight",
                "bayesian_polling_weight_capped",
                "bayesian_fundamentals_weight_capped",
                "bayesian_model_margin_dem",
                "bayesian_model_margin_dem_capped",
            ]:
                acc = f"{base_col}_accelerated"
                if acc in races.columns:
                    races[base_col] = races[acc].combine_first(races[base_col])

            drop_cols = [c for c in races.columns if c.endswith("_accelerated")]
            races = races.drop(columns=drop_cols + ["state_norm"], errors="ignore")
            races.to_csv(RACE_INPUTS, index=False)

    audit_cols = [
        "state_norm",
        "recent_poll_count_45d",
        "most_recent_poll_end_date",
        "polling_confidence_boost",
        "polling_confidence_absolute_cap",
        "bayesian_polling_weight_capped_before_polling_confidence_accelerator",
        "bayesian_polling_weight_capped_after_polling_confidence_accelerator",
        "polling_confidence_weight_change",
        "polling_confidence_margin_change_dem",
        "bayesian_model_margin_dem_capped_before_polling_confidence_accelerator",
        "bayesian_model_margin_dem_capped",
    ]

    audit_cols = [c for c in audit_cols if c in merged.columns]

    audit = (
        merged[audit_cols]
        .drop_duplicates("state_norm")
        .sort_values(["polling_confidence_weight_change", "recent_poll_count_45d"], ascending=[False, False])
    )

    audit.to_csv(AUDIT_OUT, index=False)

    print()
    print("Base Bayesian polling weight column adjusted: bayesian_polling_weight_capped")
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
