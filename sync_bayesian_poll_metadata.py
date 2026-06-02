from pathlib import Path
import pandas as pd

BAYESIAN_PATH = Path("inputs/bayesian_update_generated.csv")
POLLING_PATH = Path("inputs/polling_averages_generated.csv")


def main():
    if not BAYESIAN_PATH.exists():
        raise FileNotFoundError("inputs/bayesian_update_generated.csv not found.")

    if not POLLING_PATH.exists():
        print("No polling_averages_generated.csv found; skipping Bayesian poll metadata sync.")
        return

    bayes = pd.read_csv(BAYESIAN_PATH)
    polling = pd.read_csv(POLLING_PATH)

    if bayes.empty or polling.empty:
        print("Bayesian or polling file is empty; skipping sync.")
        return

    bayes["state"] = bayes["state"].astype(str).str.strip().str.upper()
    polling["state"] = polling["state"].astype(str).str.strip().str.upper()

    polling_cols = [
        "state",
        "polling_margin_dem",
        "poll_count",
        "total_poll_weight",
        "avg_poll_age_days",
        "latest_poll_end_date",
    ]

    polling_cols = [c for c in polling_cols if c in polling.columns]
    polling = polling[polling_cols].copy()

    merged = bayes.merge(
        polling,
        on="state",
        how="left",
        suffixes=("", "_truth"),
    )

    mapping = {
        "polling_margin_dem_truth": "polling_margin_used",
        "poll_count_truth": "poll_count",
        "total_poll_weight_truth": "total_poll_weight",
        "avg_poll_age_days_truth": "avg_poll_age_days",
    }

    changed = []

    for truth_col, target_col in mapping.items():
        if truth_col in merged.columns and target_col in merged.columns:
            before = merged[target_col].copy()
            merged[target_col] = merged[truth_col].combine_first(merged[target_col])
            changed.append(target_col)

    drop_cols = [c for c in merged.columns if c.endswith("_truth")]
    merged = merged.drop(columns=drop_cols)

    merged.to_csv(BAYESIAN_PATH, index=False)

    print("Synced Bayesian poll metadata from polling_averages_generated.csv")
    print("Updated fields:", ", ".join(changed) if changed else "None")

    me = merged[merged["state"] == "ME"]
    if not me.empty:
        row = me.iloc[0]
        print()
        print("Maine Bayesian metadata after sync")
        print("----------------------------------")
        for col in [
            "polling_margin_used",
            "poll_count",
            "total_poll_weight",
            "avg_poll_age_days",
            "bayesian_polling_weight",
        ]:
            if col in merged.columns:
                print(f"{col}: {row.get(col)}")


if __name__ == "__main__":
    main()
