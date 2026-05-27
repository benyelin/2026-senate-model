from pathlib import Path
import pandas as pd
import subprocess
import sys

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

NATIONAL_ENV_PATH = INPUTS / "national_environment.csv"


def read_national_environment():
    if not NATIONAL_ENV_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(NATIONAL_ENV_PATH)


def get_current_environment_value(national):
    """
    Supports both old long format:

        parameter,value

    and new wide format:

        as_of_date,generic_ballot_margin_dem,presidential_approval,
        midterm_adjustment_dem,approval_adjustment_dem,
        national_environment_margin_dem,notes
    """

    if national.empty:
        return 0.0

    # Old format
    if "parameter" in national.columns and "value" in national.columns:
        rows = national[national["parameter"] == "manual_adjustment"]

        if not rows.empty:
            return float(rows.iloc[-1]["value"])

        rows = national[national["parameter"] == "national_environment_margin_dem"]

        if not rows.empty:
            return float(rows.iloc[-1]["value"])

        return 0.0

    # New wide format
    row = national.iloc[-1]

    if "national_environment_margin_dem" in national.columns:
        val = pd.to_numeric(
            row["national_environment_margin_dem"],
            errors="coerce"
        )

        if pd.notna(val):
            return float(val)

    total = 0.0

    for col in [
        "generic_ballot_margin_dem",
        "approval_adjustment_dem",
        "presidential_approval_adjustment_dem",
        "midterm_adjustment_dem",
    ]:
        if col in national.columns:
            val = pd.to_numeric(row[col], errors="coerce")

            if pd.notna(val):
                total += float(val)

    return float(total)


def run_scenarios():
    national = read_national_environment()
    base_env = get_current_environment_value(national)

    scenarios = [
        {
            "scenario": "Current environment",
            "national_environment_margin_dem": base_env,
        },
        {
            "scenario": "Democratic environment +2",
            "national_environment_margin_dem": base_env + 2.0,
        },
        {
            "scenario": "Republican environment +2",
            "national_environment_margin_dem": base_env - 2.0,
        },
    ]

    rows = []

    summary_path = OUTPUTS / "forecast_summary.csv"

    if summary_path.exists():
        summary = pd.read_csv(summary_path)

        if not summary.empty:
            current = summary.iloc[-1].to_dict()
        else:
            current = {}
    else:
        current = {}

    for scenario in scenarios:
        row = {
            "scenario": scenario["scenario"],
            "national_environment_margin_dem": scenario["national_environment_margin_dem"],
            "current_dem_control_probability": current.get("dem_control_probability", None),
            "current_expected_dem_seats": current.get("expected_dem_seats", None),
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    scenario_path = OUTPUTS / "scenario_summary.csv"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(scenario_path, index=False)

    return out


if __name__ == "__main__":
    print(run_scenarios().to_string(index=False))
