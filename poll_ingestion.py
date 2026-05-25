from pathlib import Path
import argparse
import pandas as pd


def ingest_manual_polls(input_dir="inputs", output_dir="outputs", as_of=None):
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

    if not manual_path.exists():
        pd.DataFrame(columns=columns).to_csv(out_path, index=False)
        print("No manual polls found. Wrote empty polling averages.")
        return

    polls = pd.read_csv(manual_path)

    if polls.empty:
        pd.DataFrame(columns=columns).to_csv(out_path, index=False)
        print("Manual polls file is empty. Wrote empty polling averages.")
        return

    # Use Dem-vs-Rep margin for model partisan control.
    # This avoids old aggregator/imported polling entirely.
    polls["polling_margin_dem"] = polls["dem_pct"] - polls["rep_pct"]

    if "poll_weight" not in polls.columns:
        polls["poll_weight"] = 1.0

    if "days_old" not in polls.columns:
        polls["days_old"] = 0

    polls["end_date"] = pd.to_datetime(polls["end_date"], errors="coerce")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="inputs")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args()

    ingest_manual_polls(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        as_of=args.as_of,
    )


if __name__ == "__main__":
    main()