from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path("inputs/race_inputs.csv")

# Democratic presidential margins by state.
# Positive = Democratic margin.
# Negative = Republican margin.
#
# These are entered as Democratic % minus Republican %.
# Source basis: state-level presidential results from 2024, 2020, 2016.
PRES_MARGINS = {
    "AL": {"pres_2024_margin_dem": -30.5, "pres_2020_margin_dem": -25.5, "pres_2016_margin_dem": -27.7},
    "AK": {"pres_2024_margin_dem": -13.13, "pres_2020_margin_dem": -10.06, "pres_2016_margin_dem": -14.7},
    "AR": {"pres_2024_margin_dem": -30.6, "pres_2020_margin_dem": -27.6, "pres_2016_margin_dem": -26.9},
    "CO": {"pres_2024_margin_dem": 11.0, "pres_2020_margin_dem": 13.5, "pres_2016_margin_dem": 4.9},
    "DE": {"pres_2024_margin_dem": 14.7, "pres_2020_margin_dem": 19.0, "pres_2016_margin_dem": 11.4},
    "FL": {"pres_2024_margin_dem": -13.10, "pres_2020_margin_dem": -3.36, "pres_2016_margin_dem": -1.2},
    "GA": {"pres_2024_margin_dem": -2.20, "pres_2020_margin_dem": 0.24, "pres_2016_margin_dem": -5.1},
    "IA": {"pres_2024_margin_dem": -13.3, "pres_2020_margin_dem": -8.2, "pres_2016_margin_dem": -9.4},
    "ID": {"pres_2024_margin_dem": -36.5, "pres_2020_margin_dem": -30.8, "pres_2016_margin_dem": -31.8},
    "IL": {"pres_2024_margin_dem": 10.9, "pres_2020_margin_dem": 17.0, "pres_2016_margin_dem": 16.9},
    "KS": {"pres_2024_margin_dem": -16.0, "pres_2020_margin_dem": -14.7, "pres_2016_margin_dem": -20.4},
    "KY": {"pres_2024_margin_dem": -30.6, "pres_2020_margin_dem": -25.9, "pres_2016_margin_dem": -29.8},
    "LA": {"pres_2024_margin_dem": -22.0, "pres_2020_margin_dem": -18.6, "pres_2016_margin_dem": -19.6},
    "MA": {"pres_2024_margin_dem": 24.7, "pres_2020_margin_dem": 33.5, "pres_2016_margin_dem": 27.2},
    "ME": {"pres_2024_margin_dem": 6.94, "pres_2020_margin_dem": 9.07, "pres_2016_margin_dem": 2.9},
    "MI": {"pres_2024_margin_dem": -1.4, "pres_2020_margin_dem": 2.8, "pres_2016_margin_dem": -0.2},
    "MN": {"pres_2024_margin_dem": 4.3, "pres_2020_margin_dem": 7.1, "pres_2016_margin_dem": 1.5},
    "MS": {"pres_2024_margin_dem": -23.0, "pres_2020_margin_dem": -16.5, "pres_2016_margin_dem": -17.8},
    "MT": {"pres_2024_margin_dem": -19.9, "pres_2020_margin_dem": -16.4, "pres_2016_margin_dem": -20.4},
    "NC": {"pres_2024_margin_dem": -3.21, "pres_2020_margin_dem": -1.34, "pres_2016_margin_dem": -3.6},
    "NE": {"pres_2024_margin_dem": -20.5, "pres_2020_margin_dem": -19.1, "pres_2016_margin_dem": -25.1},
    "NH": {"pres_2024_margin_dem": 2.7, "pres_2020_margin_dem": 7.4, "pres_2016_margin_dem": 0.4},
    "NJ": {"pres_2024_margin_dem": 5.7, "pres_2020_margin_dem": 15.9, "pres_2016_margin_dem": 14.1},
    "NM": {"pres_2024_margin_dem": 6.0, "pres_2020_margin_dem": 10.8, "pres_2016_margin_dem": 8.2},
    "OH": {"pres_2024_margin_dem": -11.21, "pres_2020_margin_dem": -8.03, "pres_2016_margin_dem": -8.1},
    "OK": {"pres_2024_margin_dem": -34.3, "pres_2020_margin_dem": -33.1, "pres_2016_margin_dem": -36.4},
    "OR": {"pres_2024_margin_dem": 14.5, "pres_2020_margin_dem": 16.1, "pres_2016_margin_dem": 11.0},
    "RI": {"pres_2024_margin_dem": 13.5, "pres_2020_margin_dem": 20.8, "pres_2016_margin_dem": 15.5},
    "SC": {"pres_2024_margin_dem": -17.8, "pres_2020_margin_dem": -11.7, "pres_2016_margin_dem": -14.3},
    "SD": {"pres_2024_margin_dem": -29.2, "pres_2020_margin_dem": -26.2, "pres_2016_margin_dem": -29.8},
    "TN": {"pres_2024_margin_dem": -29.7, "pres_2020_margin_dem": -23.2, "pres_2016_margin_dem": -26.0},
    "TX": {"pres_2024_margin_dem": -13.68, "pres_2020_margin_dem": -5.58, "pres_2016_margin_dem": -9.0},
    "VA": {"pres_2024_margin_dem": 5.6, "pres_2020_margin_dem": 10.1, "pres_2016_margin_dem": 5.3},
    "WV": {"pres_2024_margin_dem": -42.0, "pres_2020_margin_dem": -38.9, "pres_2016_margin_dem": -41.7},
    "WY": {"pres_2024_margin_dem": -44.9, "pres_2020_margin_dem": -43.4, "pres_2016_margin_dem": -46.3},
}


def main():
    if not PATH.exists():
        raise FileNotFoundError(f"Could not find {PATH}")

    df = pd.read_csv(PATH)

    if "state" not in df.columns:
        raise ValueError("race_inputs.csv must include a state column")

    df["state"] = df["state"].astype(str).str.strip().str.upper()

    for col in [
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    updated = []
    missing = []

    for state in sorted(df["state"].unique()):
        if state in PRES_MARGINS:
            mask = df["state"] == state

            for col, value in PRES_MARGINS[state].items():
                df.loc[mask, col] = value

            updated.append(state)
        else:
            missing.append(state)

    df.to_csv(PATH, index=False)

    print(f"Updated presidential margins in {PATH}")
    print("Updated states:")
    print(", ".join(updated))

    if missing:
        print()
        print("States still missing from PRES_MARGINS table:")
        print(", ".join(missing))

    preview_cols = [
        "state",
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "pres_2016_margin_dem",
    ]

    print()
    print(df[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
