from pathlib import Path
import pandas as pd

RACE_INPUTS = Path("inputs/race_inputs.csv")
FUNDAMENTALS = Path("inputs/fundamentals_generated.csv")

def main():
    if not RACE_INPUTS.exists():
        raise FileNotFoundError(f"Missing {RACE_INPUTS}")

    if not FUNDAMENTALS.exists():
        raise FileNotFoundError(f"Missing {FUNDAMENTALS}")

    races = pd.read_csv(RACE_INPUTS)
    fundamentals = pd.read_csv(FUNDAMENTALS)

    if "state" not in races.columns or "state" not in fundamentals.columns:
        raise ValueError("Both race_inputs.csv and fundamentals_generated.csv must contain a state column.")

    for col, default in {
        "independent_candidate_present": False,
        "independent_candidate_name": "",
        "independent_candidate_party_lean": "none",
        "independent_vote_share_estimate": 0.0,
        "independent_asymmetry_adjustment_dem": 0.0,
        "independent_adjustment_rationale": "",
    }.items():
        if col not in races.columns:
            races[col] = default

    races["state"] = races["state"].astype(str).str.upper()
    fundamentals["state"] = fundamentals["state"].astype(str).str.upper()

    races["independent_candidate_present"] = (
        races["independent_candidate_present"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    races["independent_vote_share_estimate"] = pd.to_numeric(
        races["independent_vote_share_estimate"],
        errors="coerce",
    ).fillna(0.0)

    races["independent_asymmetry_adjustment_dem"] = pd.to_numeric(
        races["independent_asymmetry_adjustment_dem"],
        errors="coerce",
    ).fillna(0.0)

    keep_cols = [
        "state",
        "independent_candidate_present",
        "independent_candidate_name",
        "independent_candidate_party_lean",
        "independent_vote_share_estimate",
        "independent_asymmetry_adjustment_dem",
        "independent_adjustment_rationale",
    ]

    merged = fundamentals.merge(
        races[keep_cols],
        on="state",
        how="left",
        suffixes=("", "_race_input"),
    )

    merged["independent_candidate_present"] = (
        merged["independent_candidate_present"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    merged["independent_asymmetry_adjustment_dem"] = pd.to_numeric(
        merged["independent_asymmetry_adjustment_dem"],
        errors="coerce",
    ).fillna(0.0)

    # Identify the best available fundamentals margin column.
    margin_candidates = [
        "fundamentals_margin_dem",
        "fundamental_margin_dem",
        "baseline_margin_dem",
        "model_margin_dem",
    ]

    margin_col = next((c for c in margin_candidates if c in merged.columns), None)

    if margin_col is None:
        print("No known fundamentals margin column found. Available columns:")
        print(list(merged.columns))
        raise SystemExit(1)

    adjusted_col = f"{margin_col}_pre_independent_adjustment"

    if adjusted_col not in merged.columns:
        merged[adjusted_col] = merged[margin_col]

    merged[margin_col] = (
        pd.to_numeric(merged[adjusted_col], errors="coerce").fillna(0.0)
        + merged["independent_asymmetry_adjustment_dem"]
    )

    merged["independent_adjustment_applied"] = merged["independent_asymmetry_adjustment_dem"].ne(0)

    merged.to_csv(FUNDAMENTALS, index=False)

    affected = merged[merged["independent_adjustment_applied"]][
        [
            "state",
            adjusted_col,
            "independent_asymmetry_adjustment_dem",
            margin_col,
            "independent_candidate_name",
            "independent_candidate_party_lean",
        ]
    ]

    print(f"Applied Senate independent adjustments to {FUNDAMENTALS}")
    print(f"Margin column adjusted: {margin_col}")

    if affected.empty:
        print("No nonzero independent adjustments currently applied.")
    else:
        print(affected.to_string(index=False))

if __name__ == "__main__":
    main()
