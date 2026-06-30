from pathlib import Path
import pandas as pd


def update_race_inputs_from_polls(
    input_dir="inputs",
    output_dir="outputs",
    as_of=None,
    as_of_date=None,
    half_life_days=None,
):
    """
    Manual-only polling ingestion.

    Reads:
        outputs/manual_polls_clean.csv

    Writes:
        inputs/polling_averages_generated.csv

    Uses final_poll_margin_dem when available, which includes:
    - undecided-voter discount
    - house-effect adjustment
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    manual_path = output_dir / "manual_polls_clean.csv"
    out_path = input_dir / "polling_averages_generated.csv"

    columns = [
        "state",
        "polling_margin_dem",
        "poll_count",
        "latest_poll_end_date",
        "avg_poll_age_days",
        "total_poll_weight",
    ]

    input_dir.mkdir(parents=True, exist_ok=True)

    if not manual_path.exists():
        avgs = pd.DataFrame(columns=columns)
        avgs.to_csv(out_path, index=False)
        print("No manual polls found. Wrote empty polling averages.")
        return avgs

    polls = pd.read_csv(manual_path)

    if polls.empty:
        avgs = pd.DataFrame(columns=columns)
        avgs.to_csv(out_path, index=False)
        print("Manual polls file is empty. Wrote empty polling averages.")
        return avgs

    required_cols = ["state", "dem_pct", "rep_pct"]
    missing = [c for c in required_cols if c not in polls.columns]

    if missing:
        raise ValueError(f"Manual polls file missing required columns: {missing}")

    polls["state"] = polls["state"].astype(str).str.strip().str.upper()

    # Use the fully adjusted poll margin if available.
    if "final_poll_margin_dem" in polls.columns:
        polls["polling_margin_dem"] = pd.to_numeric(
            polls["final_poll_margin_dem"],
            errors="coerce"
        )
    elif "house_effect_adjusted_dem_margin" in polls.columns:
        polls["polling_margin_dem"] = pd.to_numeric(
            polls["house_effect_adjusted_dem_margin"],
            errors="coerce"
        )
    else:
        polls["polling_margin_dem"] = (
            pd.to_numeric(polls["dem_pct"], errors="coerce")
            - pd.to_numeric(polls["rep_pct"], errors="coerce")
        )

    if "poll_weight" not in polls.columns:
        polls["poll_weight"] = 1.0

    if "days_old" not in polls.columns:
        polls["days_old"] = 0

    polls["poll_weight"] = pd.to_numeric(
        polls["poll_weight"],
        errors="coerce"
    ).fillna(1.0)

    polls["days_old"] = pd.to_numeric(
        polls["days_old"],
        errors="coerce"
    ).fillna(0)

    polls["end_date"] = pd.to_datetime(
        polls["end_date"],
        errors="coerce"
    )

    polls = polls.dropna(subset=["polling_margin_dem"])

    if polls.empty:
        avgs = pd.DataFrame(columns=columns)
        avgs.to_csv(out_path, index=False)
        print("Manual polls had no usable margins. Wrote empty polling averages.")
        return avgs

    def weighted_avg(group):
        weights = group["poll_weight"].fillna(1.0)
        margins = group["polling_margin_dem"]

        if weights.sum() == 0:
            margin = margins.mean()
        else:
            margin = (margins * weights).sum() / weights.sum()

        return pd.Series({
            "polling_margin_dem": margin,
            "poll_count": len(group),
            "latest_poll_end_date": group["end_date"].max(),
            "avg_poll_age_days": group["days_old"].mean(),
            "total_poll_weight": weights.sum(),
        })

    avgs = (
        polls
        .groupby("state", as_index=False)
        .apply(weighted_avg)
        .reset_index(drop=True)
    )

    avgs.to_csv(out_path, index=False)

    print(f"Wrote manual-only polling averages to {out_path}")

    return avgs
