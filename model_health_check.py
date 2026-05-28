from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

RACE_INPUTS = INPUTS / "race_inputs.csv"
NATIONAL_ENV = INPUTS / "national_environment.csv"
BAYESIAN_OUTPUT = INPUTS / "bayesian_update_generated.csv"
POLLING_AVERAGES = INPUTS / "polling_averages_generated.csv"

RACE_STATS = OUTPUTS / "race_stats.csv"
FORECAST_SUMMARY = OUTPUTS / "forecast_summary.csv"
SCENARIO_SUMMARY = OUTPUTS / "scenario_summary.csv"

ELECTION_DAY = date(2026, 11, 3)


def days_out():
    return max(0, (ELECTION_DAY - date.today()).days)


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def as_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fmt_margin(x):
    x = as_float(x)
    if pd.isna(x):
        return "NA"
    if x > 0:
        return f"D+{x:.2f}"
    if x < 0:
        return f"R+{abs(x):.2f}"
    return "Even"


def expected_cycle_cap():
    d = days_out()

    if d > 180:
        return 0.12
    if d > 120:
        return 0.18
    if d > 60:
        return 0.35
    if d > 30:
        return 0.50
    return 0.70


def print_messages(title, rows):
    print()
    print(title)
    print("-" * len(title))

    if not rows:
        print("None")
        return

    for msg in rows:
        print(f"- {msg}")


def main():
    errors = []
    warnings = []
    info = []

    print("2026 Senate Model Health Check")
    print("==============================")
    print(f"Days out: {days_out()}")

    required_files = [
        RACE_INPUTS,
        NATIONAL_ENV,
        BAYESIAN_OUTPUT,
        POLLING_AVERAGES,
        RACE_STATS,
        FORECAST_SUMMARY,
        SCENARIO_SUMMARY,
    ]

    for path in required_files:
        if not path.exists():
            errors.append(f"Missing required file: {path}")
        else:
            info.append(f"Found {path}")

    races = read_csv(RACE_INPUTS)
    national = read_csv(NATIONAL_ENV)
    bayes = read_csv(BAYESIAN_OUTPUT)
    stats = read_csv(RACE_STATS)
    summary = read_csv(FORECAST_SUMMARY)
    scenarios = read_csv(SCENARIO_SUMMARY)

    # Race inputs checks
    if races.empty:
        errors.append("race_inputs.csv is missing or empty.")
    else:
        required_race_cols = [
            "state",
            "fundamentals_margin_dem",
            "state_partisan_baseline_dem",
            "pres_2024_margin_dem",
            "pres_2020_margin_dem",
            "pres_2016_margin_dem",
            "state_elasticity",
            "incumbency_adjustment_dem",
        ]

        for col in required_race_cols:
            if col not in races.columns:
                errors.append(f"race_inputs.csv missing column: {col}")

        if "state" in races.columns:
            races["state"] = races["state"].astype(str).str.strip().str.upper()

        pres_cols = [
            "pres_2024_margin_dem",
            "pres_2020_margin_dem",
            "pres_2016_margin_dem",
        ]

        if all(c in races.columns for c in pres_cols):
            missing_pres = races[races[pres_cols].isna().any(axis=1)]

            if missing_pres.empty:
                info.append("All races have 2024/2020/2016 presidential margins.")
            else:
                warnings.append(
                    "Missing presidential margins for: "
                    + ", ".join(missing_pres["state"].tolist())
                )

        if "state_partisan_baseline_dem" in races.columns:
            missing_baseline = races[races["state_partisan_baseline_dem"].isna()]

            if missing_baseline.empty:
                info.append("All races have state_partisan_baseline_dem.")
            else:
                warnings.append(
                    "Missing state_partisan_baseline_dem for: "
                    + ", ".join(missing_baseline["state"].tolist())
                )

        if "current_holder" in races.columns and "incumbency_adjustment_dem" in races.columns:
            for _, row in races.iterrows():
                state = row.get("state", "")
                holder = str(row.get("current_holder", "")).upper()
                inc = as_float(row.get("incumbency_adjustment_dem"))

                if pd.isna(inc):
                    warnings.append(f"{state}: missing incumbency_adjustment_dem.")
                    continue

                notes = str(row.get("notes", "")).lower()
                fundamentals_notes = str(row.get("fundamentals_notes", "")).lower()
                combined_notes = notes + " " + fundamentals_notes

                incumbent_not_running_terms = [
                    "lost primary",
                    "lost his primary",
                    "lost her primary",
                    "retired",
                    "retiring",
                    "not running",
                    "open seat",
                    "runoff",
                    "nominee",
                    "primary",
                ]

                incumbent_not_running = any(
                    term in combined_notes
                    for term in incumbent_not_running_terms
                )

                if holder == "D" and inc <= 0 and not incumbent_not_running:
                    warnings.append(
                        f"{state}: current_holder is D but incumbency_adjustment_dem is {inc:+.2f}."
                    )

                if holder == "R" and inc >= 0 and not incumbent_not_running:
                    warnings.append(
                        f"{state}: current_holder is R but incumbency_adjustment_dem is {inc:+.2f}."
                    )

                if "R-APPOINTED" in holder and inc > 0:
                    warnings.append(
                        f"{state}: R-appointed holder has pro-Dem incumbency adjustment {inc:+.2f}."
                    )

                if "D-APPOINTED" in holder and inc < 0:
                    warnings.append(
                        f"{state}: D-appointed holder has pro-GOP incumbency adjustment {inc:+.2f}."
                    )

    # National environment checks
    if national.empty:
        errors.append("national_environment.csv is missing or empty.")
    elif "national_environment_margin_dem" not in national.columns:
        errors.append("national_environment.csv missing national_environment_margin_dem.")
    else:
        input_env = as_float(national.iloc[-1]["national_environment_margin_dem"])
        info.append(f"Input national environment: {fmt_margin(input_env)}")

        if not summary.empty and "national_environment_margin" in summary.columns:
            output_env = as_float(summary.iloc[-1]["national_environment_margin"])

            if abs(input_env - output_env) > 0.01:
                warnings.append(
                    f"National environment mismatch: input={fmt_margin(input_env)}, "
                    f"forecast_summary={fmt_margin(output_env)}."
                )
            else:
                info.append("National environment matches forecast_summary.")
        else:
            warnings.append("forecast_summary.csv missing national_environment_margin.")

    # Bayesian checks
    if bayes.empty:
        errors.append("bayesian_update_generated.csv is missing or empty.")
    else:
        if "state" not in bayes.columns:
            errors.append("bayesian_update_generated.csv missing state column.")
        else:
            bayes["state"] = bayes["state"].astype(str).str.strip().str.upper()

        weight_col = None
        for col in ["bayesian_polling_weight", "polling_weight", "poll_weight"]:
            if col in bayes.columns:
                weight_col = col
                break

        if weight_col is None:
            warnings.append("No Bayesian polling weight column found.")
        else:
            bayes[weight_col] = pd.to_numeric(bayes[weight_col], errors="coerce")
            cap = expected_cycle_cap()

            high = bayes[bayes[weight_col] > cap + 0.001]

            if high.empty:
                info.append("Bayesian polling weights are within cycle cap.")
            else:
                warnings.append(
                    "Some Bayesian polling weights exceed cycle cap "
                    f"({cap:.3f}): "
                    + ", ".join(
                        f"{row['state']}={row[weight_col]:.3f}"
                        for _, row in high.iterrows()
                    )
                )

    # Race stats checks
    if stats.empty:
        errors.append("race_stats.csv is missing or empty.")
    else:
        for col in ["state", "model_margin_dem", "simulated_dem_win_prob"]:
            if col not in stats.columns:
                errors.append(f"race_stats.csv missing column: {col}")

        if all(c in stats.columns for c in ["state", "model_margin_dem", "simulated_dem_win_prob"]):
            stats["state"] = stats["state"].astype(str).str.strip().str.upper()
            stats["model_margin_dem"] = pd.to_numeric(stats["model_margin_dem"], errors="coerce")
            stats["simulated_dem_win_prob"] = pd.to_numeric(stats["simulated_dem_win_prob"], errors="coerce")

            suspicious = []

            for _, row in stats.iterrows():
                state = row["state"]
                margin = row["model_margin_dem"]
                prob = row["simulated_dem_win_prob"]

                if pd.isna(margin) or pd.isna(prob):
                    continue

                if margin > 3 and prob < 0.45:
                    suspicious.append(f"{state}: D margin {margin:.2f} but Dem prob {prob:.1%}")

                if margin < -3 and prob > 0.55:
                    suspicious.append(f"{state}: R margin {abs(margin):.2f} but Dem prob {prob:.1%}")

            if suspicious:
                warnings.append(
                    "Potential margin/probability inconsistencies: "
                    + "; ".join(suspicious)
                )
            else:
                info.append("Race margin/probability directions look consistent.")

    # Scenario checks
    if scenarios.empty:
        warnings.append("scenario_summary.csv is missing or empty.")
    else:
        required_scenario_cols = [
            "scenario",
            "national_environment_margin_dem",
            "dem_control_probability",
            "expected_dem_seats",
        ]

        missing = [c for c in required_scenario_cols if c not in scenarios.columns]

        if missing:
            warnings.append(f"scenario_summary.csv missing columns: {missing}")
        else:
            info.append(f"Scenario summary contains {len(scenarios)} scenarios.")

    print_messages("Errors", errors)
    print_messages("Warnings", warnings)
    print_messages("Info", info)

    print()

    if errors:
        print("Health check result: FAIL")
        raise SystemExit(1)

    if warnings:
        print("Health check result: PASS WITH WARNINGS")
        raise SystemExit(0)

    print("Health check result: PASS")


if __name__ == "__main__":
    main()
