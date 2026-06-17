from pathlib import Path
import pandas as pd

INPUTS = Path("inputs")
RACE_INPUTS = INPUTS / "race_inputs.csv"

COMPONENT_COLS = [
    "state_partisan_baseline_dem",
    "state_environment_adjustment_dem",
    "incumbency_adjustment_dem",
    "candidate_quality_adjustment_dem",
    "special_adjustment_dem",
]

def safe_num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)

def main():
    if not RACE_INPUTS.exists():
        raise FileNotFoundError(f"Missing {RACE_INPUTS}")

    df = pd.read_csv(RACE_INPUTS)

    if "state" not in df.columns:
        raise ValueError("race_inputs.csv must contain state.")

    df["state"] = df["state"].astype(str).str.upper().str.strip()

    needed = [
        "mechanical_candidate_adjustment_dem",
        "candidate_scandal_adjustment_dem",
        "candidate_quality_framework_version",
    ]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing framework columns: {missing}")

    framework_active = (
        df["candidate_quality_framework_version"]
        .fillna("")
        .astype(str)
        .str.contains("senate_candidate_quality_framework_v3_option_b", case=False, na=False)
    )

    if not framework_active.any():
        print("No framework-active rows found. Nothing to finalize.")
        return

    mechanical = safe_num(df["mechanical_candidate_adjustment_dem"], 0.0)
    scandal = safe_num(df["candidate_scandal_adjustment_dem"], 0.0).clip(lower=-3.0, upper=3.0)

    df.loc[framework_active, "candidate_quality_adjustment_dem"] = (
        mechanical[framework_active] + scandal[framework_active]
    ).clip(lower=-4.0, upper=4.0)

    # Zero old ad hoc candidate-side columns so they cannot double-count.
    for col in [
        "overperformance_adjustment_dem",
        "candidate_liability_adjustment_dem",
        "objective_candidate_quality_adjustment_dem",
        "manual_candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]:
        if col in df.columns and col != "special_adjustment_dem":
            df.loc[framework_active, col] = 0.0

    for col in COMPONENT_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = safe_num(df[col], 0.0)

    df["fundamentals_component_sum_dem"] = sum(df[col] for col in COMPONENT_COLS)
    df["fundamentals_margin_dem"] = df["fundamentals_component_sum_dem"]

    if "pre_bayes_model_margin_dem" in df.columns:
        df["pre_bayes_model_margin_dem"] = df["fundamentals_margin_dem"]

    df.to_csv(RACE_INPUTS, index=False)

    print("Finalized framework candidate quality and recomputed fundamentals.")
    show_cols = [
        "state",
        "fundamentals_margin_dem",
        "candidate_quality_adjustment_dem",
        "mechanical_candidate_adjustment_dem",
        "candidate_scandal_adjustment_dem",
        "fundamentals_component_sum_dem",
        "candidate_quality_framework_version",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df.loc[df["state"].isin(["ME", "GA", "NC", "OH", "TX", "AK"]), show_cols].to_string(index=False))

if __name__ == "__main__":
    main()
