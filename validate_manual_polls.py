from pathlib import Path
import pandas as pd
import numpy as np

RAW_PATH = Path("inputs/manual_polls.csv")
OUT_PATH = Path("outputs/manual_polls_clean.csv")

REQUIRED_COLUMNS = [
    "race", "state", "chamber", "pollster",
    "start_date", "end_date", "sample_size", "sample_type",
    "dem_candidate", "rep_candidate",
    "dem_pct", "rep_pct", "ind_pct", "other_pct", "undecided_pct"
]

PCT_COLUMNS = ["dem_pct", "rep_pct", "ind_pct", "other_pct", "undecided_pct"]

POLLSTER_GRADE_WEIGHTS = {
    "A+": 1.15,
    "A": 1.10,
    "A-": 1.05,
    "B+": 1.00,
    "B": 0.95,
    "B-": 0.90,
    "C+": 0.80,
    "C": 0.70,
    "C-": 0.60,
    "D": 0.45,
    "Unknown": 0.75
}


def pct_to_number(x):
    if pd.isna(x) or x == "":
        return 0.0

    x = float(x)

    if 0 < x <= 1:
        x *= 100

    return x


def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_manual_polls():
    df = pd.read_csv(RAW_PATH)
    validate_columns(df)

    errors = []
    warnings = []

    if "pollster_grade" not in df.columns:
        df["pollster_grade"] = "Unknown"

    df["pollster_grade"] = df["pollster_grade"].fillna("Unknown")

    if "house_effect_dem" not in df.columns:
        df["house_effect_dem"] = 0.0

    df["house_effect_dem"] = pd.to_numeric(
        df["house_effect_dem"],
        errors="coerce"
    ).fillna(0.0)

    for col in PCT_COLUMNS:
        df[col] = df[col].apply(pct_to_number)

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["sample_size"] = pd.to_numeric(df["sample_size"], errors="coerce")

    for idx, row in df.iterrows():
        row_num = idx + 2

        if pd.isna(row["start_date"]):
            errors.append(f"Row {row_num}: invalid start_date")

        if pd.isna(row["end_date"]):
            errors.append(f"Row {row_num}: invalid end_date")

        if pd.notna(row["start_date"]) and pd.notna(row["end_date"]):
            if row["end_date"] < row["start_date"]:
                errors.append(f"Row {row_num}: end_date before start_date")

        if pd.isna(row["sample_size"]) or row["sample_size"] <= 0:
            errors.append(f"Row {row_num}: invalid sample_size")

        if pd.notna(row["sample_size"]) and row["sample_size"] < 300:
            warnings.append(f"Row {row_num}: small sample size ({row['sample_size']})")

        pct_sum = sum(row[c] for c in PCT_COLUMNS)

        if pct_sum < 90 or pct_sum > 105:
            warnings.append(f"Row {row_num}: percentages sum to {pct_sum:.1f}")

        if row["pollster_grade"] not in POLLSTER_GRADE_WEIGHTS:
            warnings.append(
                f"Row {row_num}: unrecognized pollster_grade '{row['pollster_grade']}', using Unknown weight"
            )

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors))

    today = pd.Timestamp.today().normalize()

    df["mid_date"] = df["start_date"] + (df["end_date"] - df["start_date"]) / 2
    df["days_old"] = (today - df["mid_date"]).dt.days

    vote_cols = {
        "Dem": "dem_pct",
        "Rep": "rep_pct",
        "Ind": "ind_pct",
        "Other": "other_pct"
    }

    leaders = []
    leader_pcts = []
    second_pcts = []
    leader_margins = []

    for _, row in df.iterrows():
        vals = {
            label: row[col]
            for label, col in vote_cols.items()
        }

        sorted_vals = sorted(vals.items(), key=lambda x: x[1], reverse=True)

        leader = sorted_vals[0][0]
        leader_pct = sorted_vals[0][1]
        second_pct = sorted_vals[1][1]
        leader_margin = leader_pct - second_pct

        leaders.append(leader)
        leader_pcts.append(leader_pct)
        second_pcts.append(second_pct)
        leader_margins.append(leader_margin)

    df["leader"] = leaders
    df["leader_pct"] = leader_pcts
    df["second_place_pct"] = second_pcts
    df["leader_margin"] = leader_margins

    df["dem_margin_vs_top_opponent"] = (
        df["dem_pct"] - df[["rep_pct", "ind_pct", "other_pct"]].max(axis=1)
    )

    df["rep_margin_vs_top_opponent"] = (
        df["rep_pct"] - df[["dem_pct", "ind_pct", "other_pct"]].max(axis=1)
    )

    df["ind_margin_vs_top_opponent"] = (
        df["ind_pct"] - df[["dem_pct", "rep_pct", "other_pct"]].max(axis=1)
    )

    df["is_three_way_race"] = df["ind_pct"] >= 10

    df["base_poll_weight"] = (
        np.sqrt(df["sample_size"])
        / (1 + (df["days_old"] / 30))
    )

    df["pollster_quality_weight"] = (
        df["pollster_grade"]
        .map(POLLSTER_GRADE_WEIGHTS)
        .fillna(POLLSTER_GRADE_WEIGHTS["Unknown"])
    )

    df["poll_weight"] = df["base_poll_weight"] * df["pollster_quality_weight"]

    clean_cols = [
        "race", "state", "chamber", "pollster", "pollster_grade", "house_effect_dem",
        "start_date", "end_date", "mid_date", "days_old",
        "sample_size", "sample_type",
        "dem_candidate", "rep_candidate", "ind_candidate", "other_candidate",
        "dem_pct", "rep_pct", "ind_pct", "other_pct", "undecided_pct",
        "leader", "leader_pct", "second_place_pct", "leader_margin",
        "dem_margin_vs_top_opponent",
        "rep_margin_vs_top_opponent",
        "ind_margin_vs_top_opponent",
        "is_three_way_race",
        "base_poll_weight",
        "pollster_quality_weight",
        "poll_weight",
        "notes"
    ]

    for col in clean_cols:
        if col not in df.columns:
            df[col] = ""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[clean_cols].to_csv(OUT_PATH, index=False)

    print(f"Saved clean manual polls to {OUT_PATH}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")


if __name__ == "__main__":
    normalize_manual_polls()