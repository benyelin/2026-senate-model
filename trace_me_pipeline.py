from pathlib import Path
import pandas as pd

STATE = "ME"

FILES = [
    ("Manual polls", Path("inputs/manual_polls.csv")),
    ("Polling averages generated", Path("inputs/polling_averages_generated.csv")),
    ("Race inputs", Path("inputs/race_inputs.csv")),
    ("Bayesian update generated", Path("inputs/bayesian_update_generated.csv")),
    ("Race stats output", Path("outputs/race_stats.csv")),
    ("Forecast summary", Path("outputs/forecast_summary.csv")),
]

COLUMNS = [
    "state",
    "fundamentals_margin_dem",
    "polling_margin_dem",
    "polling_margin_used",
    "poll_count",
    "total_poll_weight",
    "avg_poll_age_days",
    "original_bayesian_polling_weight",
    "poll_count_weight_multiplier",
    "cycle_max_polling_weight",
    "bayesian_polling_weight",
    "bayesian_polling_weight_capped",
    "bayesian_fundamentals_weight_capped",
    "bayesian_model_margin_dem",
    "bayesian_model_margin_dem_capped",
    "model_margin_dem",
    "model_margin_source",
    "objective_candidate_quality_adjustment_dem",
    "manual_candidate_quality_adjustment_dem",
    "candidate_quality_gate",
    "candidate_quality_adjustment_dem",
    "incumbency_adjustment_dem",
    "special_adjustment_dem",
    "dem_win_probability",
    "simulated_dem_win_probability",
    "simulated_dem_win_prob",
]


def show_file(label, path):
    print()
    print(label)
    print("=" * len(label))
    print(path)

    if not path.exists():
        print("MISSING")
        return

    df = pd.read_csv(path)

    if df.empty:
        print("EMPTY")
        return

    if "state" not in df.columns:
        print(df.head().to_string(index=False))
        return

    rows = df[df["state"].astype(str).str.upper() == STATE]

    if rows.empty:
        print(f"No {STATE} row found.")
        return

    cols = [c for c in COLUMNS if c in rows.columns]
    print(rows[cols].to_string(index=False))


def main():
    print(f"Pipeline trace for {STATE}")
    print("=====================")

    for label, path in FILES:
        show_file(label, path)


if __name__ == "__main__":
    main()
