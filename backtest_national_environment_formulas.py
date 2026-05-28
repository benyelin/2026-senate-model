import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("outputs/national_environment_formula_backtest.csv")
SUMMARY_PATH = Path("outputs/national_environment_formula_backtest_summary.csv")

# Historical midterm test set.
#
# Positive margins favor Democrats.
# Negative margins favor Republicans.
#
# These are starter values for testing the formulas. You can refine them later
# with your preferred final generic-ballot averages and approval averages.
#
# target_actual_house_margin_dem = actual national House popular-vote margin.
DATA = [
    {
        "year": 2006,
        "president_party": "R",
        "generic_ballot_margin_dem": 11.5,
        "presidential_approval": 38.0,
        "presidential_disapproval": 58.0,
        "midterm_adjustment_dem": 1.0,
        "actual_house_margin_dem": 7.9,
        "notes": "Bush second midterm; strong Democratic wave.",
    },
    {
        "year": 2010,
        "president_party": "D",
        "generic_ballot_margin_dem": -9.4,
        "presidential_approval": 45.0,
        "presidential_disapproval": 50.0,
        "midterm_adjustment_dem": -1.0,
        "actual_house_margin_dem": -6.8,
        "notes": "Obama first midterm; Republican wave.",
    },
    {
        "year": 2014,
        "president_party": "D",
        "generic_ballot_margin_dem": -2.4,
        "presidential_approval": 42.0,
        "presidential_disapproval": 53.0,
        "midterm_adjustment_dem": -1.0,
        "actual_house_margin_dem": -5.7,
        "notes": "Obama second midterm; GOP-favorable environment.",
    },
    {
        "year": 2018,
        "president_party": "R",
        "generic_ballot_margin_dem": 7.3,
        "presidential_approval": 42.0,
        "presidential_disapproval": 53.0,
        "midterm_adjustment_dem": 1.0,
        "actual_house_margin_dem": 8.6,
        "notes": "Trump first midterm; Democratic wave.",
    },
    {
        "year": 2022,
        "president_party": "D",
        "generic_ballot_margin_dem": -1.2,
        "presidential_approval": 42.0,
        "presidential_disapproval": 54.0,
        "midterm_adjustment_dem": -1.0,
        "actual_house_margin_dem": -2.7,
        "notes": "Biden first midterm; modest GOP national House edge.",
    },
]


def clamp(value, low, high):
    return max(low, min(high, value))


def approval_adjustment_dem(row):
    """
    Convert approval into Democratic-margin terms.

    If president is Republican:
      low approval helps Democrats.

    If president is Democratic:
      low approval hurts Democrats.

    Adjustment is capped at +/- 3.
    """
    approval = float(row["presidential_approval"])
    president_party = str(row["president_party"]).upper()

    republican_president_adjustment = clamp((45.0 - approval) / 3.0, -3.0, 3.0)

    if president_party == "R":
        return republican_president_adjustment

    if president_party == "D":
        return -republican_president_adjustment

    return 0.0


FORMULAS = {
    "Full additive": {
        "description": "generic + approval + midterm",
        "generic_weight": 1.00,
        "approval_weight": 1.00,
        "midterm_weight": 1.00,
    },
    "Reduced double-count": {
        "description": "0.85*generic + 0.50*approval + 0.50*midterm",
        "generic_weight": 0.85,
        "approval_weight": 0.50,
        "midterm_weight": 0.50,
    },
    "Medium": {
        "description": "generic + 0.50*approval + 0.50*midterm",
        "generic_weight": 1.00,
        "approval_weight": 0.50,
        "midterm_weight": 0.50,
    },
    "Generic plus half approval": {
        "description": "generic + 0.50*approval",
        "generic_weight": 1.00,
        "approval_weight": 0.50,
        "midterm_weight": 0.00,
    },
}


def run_backtest():
    rows = []

    data = pd.DataFrame(DATA)

    data["approval_adjustment_dem"] = data.apply(approval_adjustment_dem, axis=1)

    for _, row in data.iterrows():
        for formula_name, formula in FORMULAS.items():
            prediction = (
                formula["generic_weight"] * row["generic_ballot_margin_dem"]
                + formula["approval_weight"] * row["approval_adjustment_dem"]
                + formula["midterm_weight"] * row["midterm_adjustment_dem"]
            )

            actual = row["actual_house_margin_dem"]
            error = prediction - actual

            rows.append(
                {
                    "year": row["year"],
                    "formula": formula_name,
                    "description": formula["description"],
                    "president_party": row["president_party"],
                    "generic_ballot_margin_dem": row["generic_ballot_margin_dem"],
                    "presidential_approval": row["presidential_approval"],
                    "presidential_disapproval": row["presidential_disapproval"],
                    "approval_adjustment_dem": row["approval_adjustment_dem"],
                    "midterm_adjustment_dem": row["midterm_adjustment_dem"],
                    "predicted_environment_dem": prediction,
                    "actual_house_margin_dem": actual,
                    "error": error,
                    "absolute_error": abs(error),
                    "notes": row["notes"],
                }
            )

    results = pd.DataFrame(rows)

    summary = (
        results
        .groupby(["formula", "description"], as_index=False)
        .agg(
            mean_error=("error", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            max_absolute_error=("absolute_error", "max"),
        )
        .sort_values("mean_absolute_error")
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    return results, summary


def format_margin(x):
    if pd.isna(x):
        return ""
    if x > 0:
        return f"D+{x:.1f}"
    if x < 0:
        return f"R+{abs(x):.1f}"
    return "Even"


def print_readable(results, summary):
    print()
    print("National Environment Formula Backtest")
    print("=====================================")

    print()
    print("Formula Summary")
    print("---------------")

    display_summary = summary.copy()
    display_summary["mean_error"] = display_summary["mean_error"].map(lambda x: f"{x:+.2f}")
    display_summary["mean_absolute_error"] = display_summary["mean_absolute_error"].map(lambda x: f"{x:.2f}")
    display_summary["max_absolute_error"] = display_summary["max_absolute_error"].map(lambda x: f"{x:.2f}")

    print(display_summary.to_string(index=False))

    print()
    print("Year-by-Year Detail")
    print("-------------------")

    detail = results.copy()
    detail["predicted"] = detail["predicted_environment_dem"].map(format_margin)
    detail["actual"] = detail["actual_house_margin_dem"].map(format_margin)
    detail["error_fmt"] = detail["error"].map(lambda x: f"{x:+.2f}")
    detail["abs_error_fmt"] = detail["absolute_error"].map(lambda x: f"{x:.2f}")

    detail_cols = [
        "year",
        "formula",
        "predicted",
        "actual",
        "error_fmt",
        "abs_error_fmt",
    ]

    print(detail[detail_cols].to_string(index=False))

    print()
    print(f"Wrote detailed results to {OUTPUT_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    results, summary = run_backtest()
    print_readable(results, summary)
