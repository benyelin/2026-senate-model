from pathlib import Path
from datetime import datetime
import pandas as pd

OUTPUTS = Path("outputs")

SUMMARY_PATH = OUTPUTS / "forecast_summary.csv"
HISTORY_PATH = OUTPUTS / "senate_forecast_history.csv"
RACE_STATS_PATH = OUTPUTS / "race_stats.csv"


def first_available(row, names, default=None):
    for name in names:
        if name in row.index:
            value = row.get(name)
            if pd.notna(value):
                return value
    return default


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("outputs/forecast_summary.csv not found. Run the Senate pipeline first.")

    summary = pd.read_csv(SUMMARY_PATH)

    if summary.empty:
        raise ValueError("outputs/forecast_summary.csv is empty.")

    srow = summary.iloc[-1]

    timestamp = datetime.now().isoformat(timespec="seconds")

    snapshot = {
        "timestamp": timestamp,
        "run_date": datetime.now().date().isoformat(),
        "n_sims": first_available(srow, ["n_sims"]),
        "days_out": first_available(srow, ["days_out"]),
        "expected_dem_seats": first_available(srow, ["expected_dem_seats"]),
        "median_dem_seats": first_available(srow, ["median_dem_seats"]),
        "dem_control_probability": first_available(
            srow,
            ["dem_control_probability", "dem_majority_probability"],
        ),
        "dem_control_threshold": first_available(srow, ["dem_control_threshold"], 51),
        "national_environment_margin": first_available(
            srow,
            [
                "national_environment_margin",
                "national_environment",
                "national_environment_margin_dem",
            ],
        ),
        "polling_weight": first_available(srow, ["polling_weight"]),
        "fundamentals_weight": first_available(srow, ["fundamentals_weight"]),
        "national_error_sd": first_available(srow, ["national_error_sd"]),
        "race_error_sd": first_available(srow, ["race_error_sd"]),
        "total_error_sd": first_available(srow, ["total_error_sd"]),
        "implied_correlation": first_available(srow, ["implied_correlation"]),
    }

    if RACE_STATS_PATH.exists():
        race = pd.read_csv(RACE_STATS_PATH)
        if "rating" in race.columns:
            counts = race["rating"].value_counts(dropna=False).to_dict()
            for rating in [
                "Safe D",
                "Likely D",
                "Lean D",
                "Tilt D",
                "Toss-Up",
                "Tilt R",
                "Lean R",
                "Likely R",
                "Safe R",
            ]:
                key = "rating_" + rating.lower().replace(" ", "_").replace("-", "")
                snapshot[key] = counts.get(rating, 0)

    new_row = pd.DataFrame([snapshot])

    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row

    history = history.drop_duplicates(subset=["timestamp"], keep="last")
    history.to_csv(HISTORY_PATH, index=False)

    print(f"Appended Senate forecast snapshot to {HISTORY_PATH}")
    print(new_row.to_string(index=False))


if __name__ == "__main__":
    main()
