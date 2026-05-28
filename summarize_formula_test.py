from pathlib import Path
import pandas as pd
import numpy as np

SUMMARY_PATH = Path("outputs/national_environment_formula_summary.csv")
RACE_PATH = Path("outputs/national_environment_formula_race_detail.csv")

KEY_STATES = ["FL", "AK", "TX", "GA", "NC", "OH", "ME"]


def fmt_margin(x):
    try:
        x = float(x)
    except Exception:
        return ""

    if pd.isna(x):
        return ""

    if x > 0:
        return f"D+{x:.1f}"
    if x < 0:
        return f"R+{abs(x):.1f}"
    return "Even"


def fmt_pct(x):
    try:
        x = float(x)
    except Exception:
        return ""

    if pd.isna(x):
        return ""

    return f"{x:.1%}"


def fmt_num(x):
    try:
        x = float(x)
    except Exception:
        return ""

    if pd.isna(x):
        return ""

    return f"{x:.2f}"


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Could not find {SUMMARY_PATH}")

    if not RACE_PATH.exists():
        raise FileNotFoundError(f"Could not find {RACE_PATH}")

    summary = pd.read_csv(SUMMARY_PATH)
    races = pd.read_csv(RACE_PATH)

    if summary.empty:
        raise ValueError("Formula summary is empty")

    if races.empty:
        raise ValueError("Formula race detail is empty")

    races["state"] = races["state"].astype(str).str.strip().str.upper()

    compact_rows = []

    for _, srow in summary.iterrows():
        formula = srow["formula_name"]

        row = {
            "Formula": formula,
            "National env.": fmt_margin(srow.get("national_environment_margin_dem")),
            "Dem control": fmt_pct(srow.get("dem_control_probability")),
            "Expected Dem seats": fmt_num(srow.get("expected_dem_seats")),
            "Median Dem seats": fmt_num(srow.get("median_dem_seats")),
        }

        formula_races = races[races["formula_name"] == formula]

        for state in KEY_STATES:
            state_row = formula_races[formula_races["state"] == state]

            if state_row.empty:
                row[f"{state} odds"] = ""
                row[f"{state} margin"] = ""
                continue

            state_row = state_row.iloc[0]

            row[f"{state} odds"] = fmt_pct(state_row.get("simulated_dem_win_prob"))
            row[f"{state} margin"] = fmt_margin(state_row.get("model_margin_dem"))

        compact_rows.append(row)

    compact = pd.DataFrame(compact_rows)

    # Sort by national environment from most GOP-friendly to most Dem-friendly.
    if "National env." in compact.columns and "national_environment_margin_dem" in summary.columns:
        order = summary[["formula_name", "national_environment_margin_dem"]].copy()
        order = order.rename(columns={"formula_name": "Formula"})
        compact = compact.merge(order, on="Formula", how="left")
        compact = compact.sort_values("national_environment_margin_dem")
        compact = compact.drop(columns=["national_environment_margin_dem"])

    print()
    print("National Environment Formula Comparison")
    print("=======================================")
    print()
    print(compact.to_string(index=False))

    print()
    print("Interpretation guide")
    print("--------------------")
    print("- National env. is the Democratic national environment used by the model.")
    print("- State margins are Democratic model margins; D+ means Democrat ahead, R+ means Republican ahead.")
    print("- Odds are simulated Democratic win probabilities for that state.")
    print()

    out_path = Path("outputs/national_environment_formula_readable_summary.csv")
    compact.to_csv(out_path, index=False)
    print(f"Wrote readable summary to {out_path}")


if __name__ == "__main__":
    main()
