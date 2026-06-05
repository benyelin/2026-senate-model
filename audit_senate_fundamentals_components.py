from pathlib import Path
import pandas as pd

INPUT_PATH = Path("inputs/race_inputs.csv")
OUTPUT_PATH = Path("outputs/senate_fundamentals_component_audit.csv")

COMPONENTS = [
    "state_partisan_baseline_dem",
    "state_environment_adjustment_dem",
    "incumbency_adjustment_dem",
    "overperformance_adjustment_dem",
    "candidate_liability_adjustment_dem",
    "candidate_quality_adjustment_dem",
    "special_adjustment_dem",
]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError("inputs/race_inputs.csv not found.")

    df = pd.read_csv(INPUT_PATH)

    for col in COMPONENTS + ["fundamentals_margin_dem"]:
        if col not in df.columns:
            df[col] = 0.0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["fundamentals_component_sum_dem"] = df[COMPONENTS].sum(axis=1)
    df["fundamentals_component_gap"] = (
        df["fundamentals_margin_dem"] - df["fundamentals_component_sum_dem"]
    )

    audit_cols = [
        "state",
        "dem_candidate",
        "rep_candidate",
        "fundamentals_margin_dem",
        "fundamentals_component_sum_dem",
        "fundamentals_component_gap",
    ] + COMPONENTS

    audit_cols = [c for c in audit_cols if c in df.columns]

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df[audit_cols].to_csv(OUTPUT_PATH, index=False)

    flagged = df[df["fundamentals_component_gap"].abs() > 0.01].copy()

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Flagged rows: {len(flagged)}")

    if not flagged.empty:
        show_cols = [
            "state",
            "fundamentals_margin_dem",
            "fundamentals_component_sum_dem",
            "fundamentals_component_gap",
        ]
        print(flagged[show_cols].to_string(index=False))
        raise SystemExit("Fundamentals component audit failed.")

    print("Fundamentals component audit passed.")


if __name__ == "__main__":
    main()
