from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
RACE_INPUTS_PATH = INPUTS / "race_inputs.csv"
NATIONAL_ENV_PATH = INPUTS / "national_environment.csv"

# Presidential baseline weights
WEIGHT_2024 = 0.60
WEIGHT_2020 = 0.25
WEIGHT_2016 = 0.15


def read_national_environment():
    """
    Reads national_environment.csv if available.

    Expected possible columns:
      - national_environment_margin_dem
      - generic_ballot_margin_dem
      - presidential_approval
      - presidential_approval_adjustment_dem
      - midterm_adjustment_dem

    If national_environment_margin_dem exists, use it directly.
    Otherwise, build a simple estimate from available columns.

    Positive = Democratic national environment.
    Negative = Republican national environment.
    """

    if not NATIONAL_ENV_PATH.exists():
        print("No national_environment.csv found. Using national environment = 0.0")
        return 0.0

    env = pd.read_csv(NATIONAL_ENV_PATH)

    if env.empty:
        print("national_environment.csv is empty. Using national environment = 0.0")
        return 0.0

    row = env.iloc[-1]

    if "national_environment_margin_dem" in env.columns:
        val = pd.to_numeric(
            row["national_environment_margin_dem"],
            errors="coerce"
        )

        if pd.notna(val):
            print(f"Using national_environment_margin_dem directly: {float(val):.2f}")
            return float(val)

    generic_ballot = 0.0
    approval_adjustment = 0.0
    midterm_adjustment = 0.0

    if "generic_ballot_margin_dem" in env.columns:
        val = pd.to_numeric(
            row["generic_ballot_margin_dem"],
            errors="coerce"
        )

        if pd.notna(val):
            generic_ballot = float(val)

    if "presidential_approval_adjustment_dem" in env.columns:
        val = pd.to_numeric(
            row["presidential_approval_adjustment_dem"],
            errors="coerce"
        )

        if pd.notna(val):
            approval_adjustment = float(val)

    elif "presidential_approval" in env.columns:
        approval = pd.to_numeric(
            row["presidential_approval"],
            errors="coerce"
        )

        # Very simple placeholder:
        # approval below 45 hurts the president's party.
        # This assumes the president is Republican, so low approval helps Democrats.
        if pd.notna(approval):
            approval_adjustment = max(-3.0, min(3.0, (45.0 - float(approval)) / 3.0))

    if "midterm_adjustment_dem" in env.columns:
        val = pd.to_numeric(
            row["midterm_adjustment_dem"],
            errors="coerce"
        )

        if pd.notna(val):
            midterm_adjustment = float(val)

    national_environment = (
        generic_ballot
        + approval_adjustment
        + midterm_adjustment
    )

    print(
        "Built national environment from components: "
        f"generic_ballot={generic_ballot:.2f}, "
        f"approval_adjustment={approval_adjustment:.2f}, "
        f"midterm_adjustment={midterm_adjustment:.2f}, "
        f"total={national_environment:.2f}"
    )

    return float(national_environment)


def coerce_numeric(df, columns, default=np.nan):
    for col in columns:
        if col not in df.columns:
            df[col] = default

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


def main():
    if not RACE_INPUTS_PATH.exists():
        raise FileNotFoundError(f"Could not find {RACE_INPUTS_PATH}")

    races = pd.read_csv(RACE_INPUTS_PATH)

    if "state" not in races.columns:
        raise ValueError("race_inputs.csv must include a state column")

    races["state"] = races["state"].astype(str).str.strip().str.upper()

    required_cols = [
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
        "state_elasticity",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]

    races = coerce_numeric(races, required_cols, default=np.nan)

    # Fall back to old elasticity column if state_elasticity is blank.
    if "elasticity" in races.columns:
        old_elasticity = pd.to_numeric(
            races["elasticity"],
            errors="coerce"
        )

        races["state_elasticity"] = races["state_elasticity"].fillna(
            old_elasticity
        )

    races["state_elasticity"] = races["state_elasticity"].fillna(0.75)

    for col in [
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]:
        races[col] = races[col].fillna(0.0)

    national_environment = read_national_environment()

    races["national_environment_margin_dem"] = national_environment

    # Calculate weighted state partisan baseline.
    # Positive = Democratic lean, negative = Republican lean.
    has_all_pres_margins = (
        races["pres_2024_margin_dem"].notna()
        & races["pres_2020_margin_dem"].notna()
        & races["pres_2016_margin_dem"].notna()
    )

    calculated_baseline = (
        WEIGHT_2024 * races["pres_2024_margin_dem"]
        + WEIGHT_2020 * races["pres_2020_margin_dem"]
        + WEIGHT_2016 * races["pres_2016_margin_dem"]
    )

    if "state_partisan_baseline_dem" not in races.columns:
        races["state_partisan_baseline_dem"] = np.nan

    races["state_partisan_baseline_dem"] = pd.to_numeric(
        races["state_partisan_baseline_dem"],
        errors="coerce"
    )

    # Only overwrite baseline where all presidential inputs exist.
    races.loc[
        has_all_pres_margins,
        "state_partisan_baseline_dem"
    ] = calculated_baseline.loc[has_all_pres_margins]

    races["state_environment_adjustment_dem"] = (
        races["national_environment_margin_dem"]
        * races["state_elasticity"]
    )

    calculated_fundamentals = (
        races["state_partisan_baseline_dem"]
        + races["state_environment_adjustment_dem"]
        + races["incumbency_adjustment_dem"]
        + races["candidate_quality_adjustment_dem"]
        + races["special_adjustment_dem"]
    )

    if "fundamentals_margin_dem" not in races.columns:
        races["fundamentals_margin_dem"] = np.nan

    races["fundamentals_margin_dem"] = pd.to_numeric(
        races["fundamentals_margin_dem"],
        errors="coerce"
    )

    # Only overwrite fundamentals when the state baseline exists.
    can_calculate_fundamentals = races["state_partisan_baseline_dem"].notna()

    races.loc[
        can_calculate_fundamentals,
        "fundamentals_margin_dem"
    ] = calculated_fundamentals.loc[can_calculate_fundamentals]

    # Audit notes
    if "fundamentals_notes" not in races.columns:
        races["fundamentals_notes"] = ""

    races["fundamentals_notes"] = races["fundamentals_notes"].fillna("").astype(str)

    races.loc[
        has_all_pres_margins,
        "fundamentals_notes"
    ] = (
        "Calculated from 2024/2020/2016 presidential margins "
        f"({WEIGHT_2024:.0%}/{WEIGHT_2020:.0%}/{WEIGHT_2016:.0%}), "
        "state elasticity, national environment, and race adjustments."
    )

    races.loc[
        ~has_all_pres_margins,
        "fundamentals_notes"
    ] = races.loc[
        ~has_all_pres_margins,
        "fundamentals_notes"
    ].replace("", "Presidential margins incomplete; retained existing fundamentals where needed.")

    races.to_csv(RACE_INPUTS_PATH, index=False)

    print(f"Updated fundamentals in {RACE_INPUTS_PATH}")
    print(f"National environment used: {national_environment:.2f}")

    show_cols = [
        "state",
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
        "fundamentals_margin_dem",
    ]

    show_cols = [c for c in show_cols if c in races.columns]

    print("\nFundamentals preview:")
    print(races[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
