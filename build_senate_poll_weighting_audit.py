from pathlib import Path
from datetime import date

import pandas as pd

from poll_weighting_engine import weight_polls


INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

POLL_PATHS = [
    INPUTS / "manual_polls_adjusted.csv",
    INPUTS / "manual_polls.csv",
]

RACE_INPUTS = INPUTS / "race_inputs.csv"
CURRENT_AVERAGES = INPUTS / "polling_averages_generated.csv"

POLL_AUDIT = OUTPUTS / "senate_poll_weighting_audit.csv"
RACE_AUDIT = OUTPUTS / "senate_polling_average_comparison.csv"


def first_existing(paths):
    return next((p for p in paths if p.exists()), None)


def normalize_state(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def main():
    OUTPUTS.mkdir(exist_ok=True)

    poll_path = first_existing(POLL_PATHS)

    if poll_path is None:
        raise FileNotFoundError("No Senate manual poll file found.")

    polls = pd.read_csv(poll_path)
    races = pd.read_csv(RACE_INPUTS)

    polls["state"] = polls["state"].apply(normalize_state)
    races["state"] = races["state"].apply(normalize_state)

    end_dates = pd.to_datetime(polls["end_date"], errors="coerce")
    today = pd.Timestamp(date.today())

    polls["poll_age_days"] = (
        today - end_dates
    ).dt.days.clip(lower=0)

    current_environment = pd.to_numeric(
        races.get("national_environment_margin_dem", 0.0),
        errors="coerce",
    )

    if current_environment.notna().any():
        current_environment_value = float(current_environment.dropna().iloc[0])
    else:
        current_environment_value = 0.0

    polls["current_national_environment_dem"] = current_environment_value

    race_cols = [
        "state",
        "state_elasticity",
    ]
    race_cols = [c for c in race_cols if c in races.columns]

    polls = polls.merge(
        races[race_cols].drop_duplicates("state"),
        on="state",
        how="left",
    )

    polls["race_environment_sensitivity"] = pd.to_numeric(
        polls.get("state_elasticity", 1.0),
        errors="coerce",
    ).fillna(1.0)

    # Until a dated national-environment history exists, translation remains zero.
    if "national_environment_at_poll_date_dem" not in polls.columns:
        polls["national_environment_at_poll_date_dem"] = pd.NA

    days_out = 127

    if "days_out" in races.columns:
        parsed_days = pd.to_numeric(races["days_out"], errors="coerce").dropna()
        if not parsed_days.empty:
            days_out = float(parsed_days.iloc[0])

    poll_audit, race_averages = weight_polls(
        polls,
        race_col="state",
        pollster_col="pollster",
        days_out=days_out,
    )

    poll_audit.to_csv(POLL_AUDIT, index=False)

    if CURRENT_AVERAGES.exists():
        current = pd.read_csv(CURRENT_AVERAGES)
        current["state"] = current["state"].apply(normalize_state)

        keep = [
            c for c in [
                "state",
                "polling_margin_dem",
                "poll_count",
                "total_poll_weight",
                "avg_poll_age_days",
            ]
            if c in current.columns
        ]

        current = current[keep].rename(
            columns={
                "polling_margin_dem": "current_polling_margin_dem",
                "poll_count": "current_poll_count",
                "total_poll_weight": "current_total_poll_weight",
                "avg_poll_age_days": "current_avg_poll_age_days",
            }
        )

        race_averages = race_averages.merge(
            current,
            on="state",
            how="left",
        )

        race_averages["polling_margin_change_dem"] = (
            race_averages["polling_margin_dem"]
            - race_averages["current_polling_margin_dem"]
        )

    race_averages.to_csv(RACE_AUDIT, index=False)

    print(f"Wrote poll audit: {POLL_AUDIT}")
    print(f"Wrote race comparison: {RACE_AUDIT}")

    show = race_averages.sort_values(
        "effective_poll_count",
        ascending=False,
    )

    print("\nNew polling averages:")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
