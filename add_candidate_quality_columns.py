from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path("inputs/race_inputs.csv")

NEW_COLUMNS = {
    # Nominee status
    # Use True/False. Candidate quality only applies fully if nominee is confirmed.
    "dem_nominee_confirmed": False,
    "gop_nominee_confirmed": False,

    # Prior elected experience
    "dem_prior_elected_office": False,
    "gop_prior_elected_office": False,

    # Prior statewide winner
    "dem_prior_statewide_win": False,
    "gop_prior_statewide_win": False,

    # Prior candidate overperformance relative to partisan baseline.
    # Example: Dem wins statewide by D+2 in a state baseline of R+3 => +5.
    # Example: GOP wins statewide by R+10 in a state baseline of R+4 => +6 for GOP.
    "dem_prior_overperformance": 0.0,
    "gop_prior_overperformance": 0.0,

    # Candidate liabilities.
    # Positive numbers mean that candidate has a liability.
    # Dem liability hurts Democrats; GOP liability helps Democrats.
    "dem_candidate_liability": 0.0,
    "gop_candidate_liability": 0.0,

    # Manual/legacy candidate-quality adjustment.
    # This preserves earlier judgment calls or expert overrides.
    "manual_candidate_quality_adjustment_dem": 0.0,

    # Components calculated by update_candidate_quality.py
    "prior_elected_experience_adjustment_dem": 0.0,
    "prior_statewide_win_adjustment_dem": 0.0,
    "overperformance_adjustment_dem": 0.0,
    "candidate_liability_adjustment_dem": 0.0,

    # Objective total before manual override.
    "objective_candidate_quality_adjustment_dem": 0.0,

    # Gate factor based on nominee status.
    "candidate_quality_gate": 0.0,

    # Human-readable audit note.
    "candidate_quality_notes": "",
}


def main():
    if not PATH.exists():
        raise FileNotFoundError(f"Could not find {PATH}")

    df = pd.read_csv(PATH)

    if "state" not in df.columns:
        raise ValueError("race_inputs.csv must include a state column")

    added = []

    for col, default in NEW_COLUMNS.items():
        if col not in df.columns:
            df[col] = default
            added.append(col)

    if "candidate_quality_adjustment_dem" not in df.columns:
        df["candidate_quality_adjustment_dem"] = 0.0
        added.append("candidate_quality_adjustment_dem")

    # Keep state normalized
    df["state"] = df["state"].astype(str).str.strip().str.upper()

    df.to_csv(PATH, index=False)

    print(f"Updated {PATH}")

    if added:
        print("Added candidate-quality columns:")
        for col in added:
            print(f"- {col}")
    else:
        print("No new columns needed; candidate-quality columns already exist.")

    show_cols = [
        "state",
        "dem_candidate",
        "gop_candidate",
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
        "candidate_quality_adjustment_dem",
    ]

    show_cols = [c for c in show_cols if c in df.columns]

    print()
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
