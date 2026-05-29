from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path("inputs/race_inputs.csv")

# Component values
PRIOR_ELECTED_OFFICE_POINTS = 0.5
PRIOR_STATEWIDE_WIN_POINTS = 1.0
OVERPERFORMANCE_MULTIPLIER = 0.25

# Caps
OVERPERFORMANCE_CAP = 2.5
LIABILITY_CAP = 2.0
OBJECTIVE_CANDIDATE_QUALITY_CAP = 4.0
FINAL_CANDIDATE_QUALITY_CAP = 4.0


def parse_bool(x):
    if pd.isna(x):
        return False

    if isinstance(x, bool):
        return x

    x = str(x).strip().lower()

    return x in ["true", "1", "yes", "y", "confirmed"]


def clamp_series(s, low, high):
    return s.clip(lower=low, upper=high)


def nominee_gate(dem_confirmed, gop_confirmed):
    """
    Candidate quality should only fully apply when nominees are known.

    Rules:
      both nominees confirmed: 1.0
      one nominee confirmed: 0.5
      neither confirmed: 0.0

    Manual candidate-quality adjustments are also gated by this value.
    """
    dem_confirmed = bool(dem_confirmed)
    gop_confirmed = bool(gop_confirmed)

    if dem_confirmed and gop_confirmed:
        return 1.0

    if dem_confirmed or gop_confirmed:
        return 0.5

    return 0.0


def liability_scale_note():
    return (
        "Liability scale: 0=no known liability; 0.5=mild weakness; "
        "1.0=clear electoral weakness; 1.5=major documented liability; "
        "2.0=severe nominee-quality problem. Dem liability hurts Democrats; "
        "GOP liability helps Democrats."
    )


def main():
    if not PATH.exists():
        raise FileNotFoundError(f"Could not find {PATH}")

    df = pd.read_csv(PATH)

    if "state" not in df.columns:
        raise ValueError("race_inputs.csv must include a state column")

    df["state"] = df["state"].astype(str).str.strip().str.upper()

    # Preserve any existing candidate_quality_adjustment_dem as manual adjustment
    # if manual_candidate_quality_adjustment_dem has not already been created.
    if "manual_candidate_quality_adjustment_dem" not in df.columns:
        if "candidate_quality_adjustment_dem" in df.columns:
            df["manual_candidate_quality_adjustment_dem"] = pd.to_numeric(
                df["candidate_quality_adjustment_dem"],
                errors="coerce"
            ).fillna(0.0)
        else:
            df["manual_candidate_quality_adjustment_dem"] = 0.0

    required_cols = [
        "dem_nominee_confirmed",
        "gop_nominee_confirmed",
        "dem_prior_elected_office",
        "gop_prior_elected_office",
        "dem_prior_statewide_win",
        "gop_prior_statewide_win",
        "dem_prior_overperformance",
        "gop_prior_overperformance",
        "dem_candidate_liability",
        "gop_candidate_liability",
        "manual_candidate_quality_adjustment_dem",
    ]

    for col in required_cols:
        if col not in df.columns:
            if col.endswith("_confirmed") or col.endswith("_office") or col.endswith("_win"):
                df[col] = False
            else:
                df[col] = 0.0

    bool_cols = [
        "dem_nominee_confirmed",
        "gop_nominee_confirmed",
        "dem_prior_elected_office",
        "gop_prior_elected_office",
        "dem_prior_statewide_win",
        "gop_prior_statewide_win",
    ]

    for col in bool_cols:
        df[col] = df[col].apply(parse_bool)

    numeric_cols = [
        "dem_prior_overperformance",
        "gop_prior_overperformance",
        "dem_candidate_liability",
        "gop_candidate_liability",
        "manual_candidate_quality_adjustment_dem",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Experience component
    df["prior_elected_experience_adjustment_dem"] = (
        df["dem_prior_elected_office"].astype(float) * PRIOR_ELECTED_OFFICE_POINTS
        - df["gop_prior_elected_office"].astype(float) * PRIOR_ELECTED_OFFICE_POINTS
    )

    # Statewide win component
    df["prior_statewide_win_adjustment_dem"] = (
        df["dem_prior_statewide_win"].astype(float) * PRIOR_STATEWIDE_WIN_POINTS
        - df["gop_prior_statewide_win"].astype(float) * PRIOR_STATEWIDE_WIN_POINTS
    )

    # Overperformance component
    raw_overperformance = (
        OVERPERFORMANCE_MULTIPLIER * df["dem_prior_overperformance"]
        - OVERPERFORMANCE_MULTIPLIER * df["gop_prior_overperformance"]
    )

    df["overperformance_adjustment_dem"] = clamp_series(
        raw_overperformance,
        -OVERPERFORMANCE_CAP,
        OVERPERFORMANCE_CAP,
    )

    # Liability component
    # Dem liability hurts Democrats.
    # GOP liability helps Democrats.
    raw_liability = (
        -df["dem_candidate_liability"]
        + df["gop_candidate_liability"]
    )

    df["candidate_liability_adjustment_dem"] = clamp_series(
        raw_liability,
        -LIABILITY_CAP,
        LIABILITY_CAP,
    )

    df["candidate_quality_gate"] = df.apply(
        lambda row: nominee_gate(
            row["dem_nominee_confirmed"],
            row["gop_nominee_confirmed"],
        ),
        axis=1,
    )

    objective_raw = (
        df["prior_elected_experience_adjustment_dem"]
        + df["prior_statewide_win_adjustment_dem"]
        + df["overperformance_adjustment_dem"]
        + df["candidate_liability_adjustment_dem"]
    )

    df["objective_candidate_quality_adjustment_dem"] = clamp_series(
        objective_raw,
        -OBJECTIVE_CANDIDATE_QUALITY_CAP,
        OBJECTIVE_CANDIDATE_QUALITY_CAP,
    )

    # Apply nominee gate to both manual and objective components.
    gated_manual = (
        df["manual_candidate_quality_adjustment_dem"]
        * df["candidate_quality_gate"]
    )

    gated_objective = (
        df["objective_candidate_quality_adjustment_dem"]
        * df["candidate_quality_gate"]
    )

    final = gated_manual + gated_objective

    df["candidate_quality_adjustment_dem"] = clamp_series(
        final,
        -FINAL_CANDIDATE_QUALITY_CAP,
        FINAL_CANDIDATE_QUALITY_CAP,
    )

    notes = []

    for _, row in df.iterrows():
        parts = []

        if row["candidate_quality_gate"] == 0:
            parts.append("No candidate-quality effect because nominees are not confirmed.")
        elif row["candidate_quality_gate"] == 0.5:
            parts.append("Half candidate-quality effect because only one nominee is confirmed.")
        else:
            parts.append("Full candidate-quality effect because both nominees are confirmed.")

        parts.append(f"Manual CQ input: {row['manual_candidate_quality_adjustment_dem']:+.2f}.")
        parts.append(f"Objective CQ before gate: {row['objective_candidate_quality_adjustment_dem']:+.2f}.")

        if row["dem_prior_elected_office"]:
            parts.append("Dem has prior elected-office experience.")
        if row["gop_prior_elected_office"]:
            parts.append("GOP has prior elected-office experience.")

        if row["dem_prior_statewide_win"]:
            parts.append("Dem has prior statewide win.")
        if row["gop_prior_statewide_win"]:
            parts.append("GOP has prior statewide win.")

        if row["dem_prior_overperformance"] != 0:
            parts.append(f"Dem prior overperformance: {row['dem_prior_overperformance']:+.1f}.")
        if row["gop_prior_overperformance"] != 0:
            parts.append(f"GOP prior overperformance: {row['gop_prior_overperformance']:+.1f}.")

        if row["dem_candidate_liability"] != 0:
            parts.append(f"Dem liability: {row['dem_candidate_liability']:.1f}.")
        if row["gop_candidate_liability"] != 0:
            parts.append(f"GOP liability: {row['gop_candidate_liability']:.1f}.")

        parts.append(f"Final candidate-quality adjustment: {row['candidate_quality_adjustment_dem']:+.2f}.")
        parts.append(liability_scale_note())

        notes.append(" ".join(parts))

    df["candidate_quality_notes"] = notes

    df.to_csv(PATH, index=False)

    print(f"Updated candidate quality in {PATH}")

    show_cols = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "dem_nominee_confirmed",
        "gop_nominee_confirmed",
        "manual_candidate_quality_adjustment_dem",
        "objective_candidate_quality_adjustment_dem",
        "prior_elected_experience_adjustment_dem",
        "prior_statewide_win_adjustment_dem",
        "overperformance_adjustment_dem",
        "candidate_liability_adjustment_dem",
        "candidate_quality_gate",
        "candidate_quality_adjustment_dem",
    ]

    show_cols = [c for c in show_cols if c in df.columns]

    print()
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
