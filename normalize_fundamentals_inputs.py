from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path("inputs/race_inputs.csv")

# Default incumbency assumptions.
# Positive helps Democrats; negative helps Republicans.
DEFAULT_INCUMBENCY_ADJUSTMENTS = {
    "D": 2.0,
    "D-APPOINTED": 1.0,
    "R": -2.0,
    "R-APPOINTED": -1.0,
    "I": 0.0,
    "OPEN": 0.0,
    "VACANT": 0.0,
    "UNKNOWN": 0.0,
}

# More nuanced state-specific overrides.
# These preserve the judgment calls we've already discussed.
STATE_INCUMBENCY_OVERRIDES = {
    "AK": -1.5,   # Sullivan incumbent, but AK candidate dynamics are unusual.
    "FL": -1.0,   # Moody appointed incumbent.
    "ME": -2.5,   # Collins overperformance/incumbency.
    "NC": 0.0,    # Open/uncertain; Cooper strength handled as candidate quality.
    "OH": 0.0,    # Brown candidate strength handled separately.
    "TX": 0.0,    # GOP field/runoff uncertainty; candidate quality handled separately.
    "GA": 0.0,    # Keep neutral pending field clarity.
}

# Candidate-quality placeholders.
# Positive helps Democrats; negative helps Republicans.
STATE_CANDIDATE_QUALITY_OVERRIDES = {
    "AK": 1.0,
    "NC": 2.0,
    "OH": 2.0,
    "TX": 1.0,
}


def normalize_holder(x):
    if pd.isna(x):
        return "UNKNOWN"

    x = str(x).strip().upper()

    if x in ["R", "REP", "REPUBLICAN", "GOP"]:
        return "R"

    if x in ["D", "DEM", "DEMOCRAT", "DEMOCRATIC"]:
        return "D"

    if "R" in x and "APPOINT" in x:
        return "R-APPOINTED"

    if "D" in x and "APPOINT" in x:
        return "D-APPOINTED"

    if "OPEN" in x:
        return "OPEN"

    if "VACANT" in x:
        return "VACANT"

    if x == "I":
        return "I"

    return x


def main():
    if not PATH.exists():
        raise FileNotFoundError(f"Could not find {PATH}")

    df = pd.read_csv(PATH)

    if "state" not in df.columns:
        raise ValueError("race_inputs.csv must include a state column")

    df["state"] = df["state"].astype(str).str.strip().str.upper()

    for col in [
        "fundamentals_margin_dem",
        "state_partisan_baseline_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "fundamentals_notes",
    ]:
        if col not in df.columns:
            if col == "fundamentals_notes":
                df[col] = ""
            else:
                df[col] = np.nan

    df["fundamentals_margin_dem"] = pd.to_numeric(
        df["fundamentals_margin_dem"],
        errors="coerce"
    )

    df["state_partisan_baseline_dem"] = pd.to_numeric(
        df["state_partisan_baseline_dem"],
        errors="coerce"
    )

    df["incumbency_adjustment_dem"] = pd.to_numeric(
        df["incumbency_adjustment_dem"],
        errors="coerce"
    )

    df["candidate_quality_adjustment_dem"] = pd.to_numeric(
        df["candidate_quality_adjustment_dem"],
        errors="coerce"
    )

    df["special_adjustment_dem"] = pd.to_numeric(
        df["special_adjustment_dem"],
        errors="coerce"
    ).fillna(0.0)

    df["fundamentals_notes"] = df["fundamentals_notes"].fillna("").astype(str)

    # 1. Fill missing state_partisan_baseline_dem with the existing fundamentals
    # as a temporary legacy baseline.
    #
    # This avoids blank dashboard audit fields for states where we have not yet
    # entered 2024/2020/2016 presidential results.
    missing_baseline = df["state_partisan_baseline_dem"].isna()

    df.loc[
        missing_baseline,
        "state_partisan_baseline_dem"
    ] = df.loc[
        missing_baseline,
        "fundamentals_margin_dem"
    ]

    df.loc[
        missing_baseline,
        "fundamentals_notes"
    ] = df.loc[
        missing_baseline,
        "fundamentals_notes"
    ].apply(
        lambda x: (
            (x + " " if x else "")
            + "Legacy baseline used pending presidential-margin entry."
        ).strip()
    )

    # 2. Apply incumbency adjustments.
    #
    # We overwrite only where current_holder exists and a specific state override
    # or default rule is available.
    if "current_holder" in df.columns:
        holder_norm = df["current_holder"].apply(normalize_holder)

        default_adjustments = holder_norm.map(
            DEFAULT_INCUMBENCY_ADJUSTMENTS
        ).fillna(0.0)

        df["incumbency_adjustment_dem"] = default_adjustments

    # State-specific overrides
    for state, val in STATE_INCUMBENCY_OVERRIDES.items():
        df.loc[
            df["state"] == state,
            "incumbency_adjustment_dem"
        ] = val

    # Candidate-quality overrides
    df["candidate_quality_adjustment_dem"] = df[
        "candidate_quality_adjustment_dem"
    ].fillna(0.0)

    for state, val in STATE_CANDIDATE_QUALITY_OVERRIDES.items():
        df.loc[
            df["state"] == state,
            "candidate_quality_adjustment_dem"
        ] = val

    # Add notes for automatic incumbency logic
    df["fundamentals_notes"] = df["fundamentals_notes"].apply(
        lambda x: x.strip()
    )

    df.to_csv(PATH, index=False)

    print(f"Normalized fundamentals inputs in {PATH}")

    show_cols = [
        "state",
        "current_holder",
        "state_partisan_baseline_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "fundamentals_margin_dem",
        "fundamentals_notes",
    ]

    show_cols = [c for c in show_cols if c in df.columns]

    print()
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
