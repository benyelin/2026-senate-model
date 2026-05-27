from pathlib import Path
import pandas as pd
import numpy as np

RACE_INPUTS_PATH = Path("inputs/race_inputs.csv")

NEW_COLUMNS_DEFAULTS = {
    # Presidential results, expressed as Democratic margin.
    # Example: Trump +13 in Florida = -13.0
    "pres_2024_margin_dem": np.nan,
    "pres_2020_margin_dem": np.nan,
    "pres_2016_margin_dem": np.nan,

    # Calculated partisan baseline from recent presidential results.
    "state_partisan_baseline_dem": np.nan,

    # State responsiveness to national environment.
    # 1.0 = fully elastic, 0.7 = partially elastic, etc.
    "state_elasticity": np.nan,

    # National environment and state-applied national adjustment.
    "national_environment_margin_dem": np.nan,
    "state_environment_adjustment_dem": np.nan,

    # Race-specific adjustments.
    # Positive helps Democrats; negative helps Republicans.
    "incumbency_adjustment_dem": 0.0,
    "candidate_quality_adjustment_dem": 0.0,
    "special_adjustment_dem": 0.0,

    # Human-readable notes/audit field.
    "fundamentals_notes": "",
}


def main():
    if not RACE_INPUTS_PATH.exists():
        raise FileNotFoundError(f"Could not find {RACE_INPUTS_PATH}")

    df = pd.read_csv(RACE_INPUTS_PATH)

    if "state" not in df.columns:
        raise ValueError("race_inputs.csv must contain a state column")

    df["state"] = df["state"].astype(str).str.strip().str.upper()

    added = []

    for col, default in NEW_COLUMNS_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            added.append(col)

    # If the older column 'elasticity' exists, use it as the initial state_elasticity
    # wherever state_elasticity is blank.
    if "elasticity" in df.columns:
        df["state_elasticity"] = pd.to_numeric(
            df["state_elasticity"],
            errors="coerce"
        )

        old_elasticity = pd.to_numeric(
            df["elasticity"],
            errors="coerce"
        )

        df["state_elasticity"] = df["state_elasticity"].fillna(old_elasticity)

    # Keep fundamentals_margin_dem, if present, as the current active value for now.
    # The next script will recalculate it once presidential margins are filled in.
    if "fundamentals_margin_dem" not in df.columns:
        df["fundamentals_margin_dem"] = np.nan
        added.append("fundamentals_margin_dem")

    # Recommended column order: preserve important existing columns first,
    # then append the new fundamentals audit columns.
    preferred_first = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "current_holder",
        "polling_margin_dem",
        "fundamentals_margin_dem",
        "elasticity",
        "state_elasticity",
        "dem_win_counts_for_seat_change",
        "race_tier",
        "notes",
        "tier_error_multiplier",
        "polling_active",
        "candidate_status_as_of",
        "poll_count",
        "latest_poll_end_date",
        "avg_poll_age_days",
        "election_system",
        "rcv_enabled",
        "rcv_transfer_dem_from_ind",
        "rcv_transfer_rep_from_ind",
        "rcv_exhaust_from_ind",
        "rcv_transfer_dem_from_other",
        "rcv_transfer_rep_from_other",
        "rcv_exhaust_from_other",
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
        "state_partisan_baseline_dem",
        "national_environment_margin_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "fundamentals_notes",
        "bayesian_model_margin_dem",
        "bayesian_polling_weight",
        "bayesian_posterior_sd",
    ]

    ordered_cols = [
        col for col in preferred_first
        if col in df.columns
    ]

    remaining_cols = [
        col for col in df.columns
        if col not in ordered_cols
    ]

    df = df[ordered_cols + remaining_cols]

    df.to_csv(RACE_INPUTS_PATH, index=False)

    print(f"Updated {RACE_INPUTS_PATH}")

    if added:
        print("Added columns:")
        for col in added:
            print(f"- {col}")
    else:
        print("No new columns were needed; all fundamentals columns already exist.")

    print("\nCurrent fundamentals-related columns:")
    cols_to_show = [
        "state",
        "fundamentals_margin_dem",
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
        "state_partisan_baseline_dem",
        "state_elasticity",
        "national_environment_margin_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]

    cols_to_show = [c for c in cols_to_show if c in df.columns]

    print(df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
