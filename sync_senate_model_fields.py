from pathlib import Path
import pandas as pd
import numpy as np

RACE_INPUTS = Path("inputs/race_inputs.csv")
POLLING_AVERAGES = Path("inputs/polling_averages_generated.csv")
BAYESIAN_GENERATED = Path("inputs/bayesian_update_generated.csv")


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def to_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def main():
    if not RACE_INPUTS.exists():
        raise FileNotFoundError("inputs/race_inputs.csv not found.")

    races = pd.read_csv(RACE_INPUTS)

    if races.empty:
        raise ValueError("inputs/race_inputs.csv is empty.")

    races["state"] = races["state"].astype(str).str.strip().str.upper()

    # ------------------------------------------------------------
    # 1. Sync polling metadata from generated polling averages.
    # ------------------------------------------------------------
    polling = read_csv(POLLING_AVERAGES)

    if not polling.empty:
        polling = polling.copy()
        polling["state"] = polling["state"].astype(str).str.strip().str.upper()

        polling_cols = [
            "state",
            "polling_margin_dem",
            "poll_count",
            "latest_poll_end_date",
            "avg_poll_age_days",
            "total_poll_weight",
        ]

        polling_cols = [c for c in polling_cols if c in polling.columns]
        polling = polling[polling_cols]

        races = races.merge(
            polling,
            on="state",
            how="left",
            suffixes=("", "_from_generated_polling"),
        )

        for col in [
            "polling_margin_dem",
            "poll_count",
            "latest_poll_end_date",
            "avg_poll_age_days",
            "total_poll_weight",
        ]:
            generated = f"{col}_from_generated_polling"
            if generated in races.columns:
                races[col] = races[generated].combine_first(races.get(col))
                races = races.drop(columns=[generated])

    # Ensure no-polled races are clean.
    if "poll_count" in races.columns:
        races["poll_count"] = pd.to_numeric(races["poll_count"], errors="coerce").fillna(0)

    if "polling_active" in races.columns and "poll_count" in races.columns:
        races["polling_active"] = races["poll_count"] > 0

    # ------------------------------------------------------------
    # 2. Sync capped Bayesian fields from generated Bayesian output.
    # ------------------------------------------------------------
    bayes = read_csv(BAYESIAN_GENERATED)

    if not bayes.empty:
        bayes = bayes.copy()
        bayes["state"] = bayes["state"].astype(str).str.strip().str.upper()

        # Prefer capped fields where available.
        rename_map = {}

        if "bayesian_polling_weight_capped" in bayes.columns:
            rename_map["bayesian_polling_weight_capped"] = "bayesian_polling_weight"
        elif "bayesian_polling_weight" in bayes.columns:
            rename_map["bayesian_polling_weight"] = "bayesian_polling_weight"

        if "bayesian_model_margin_dem_capped" in bayes.columns:
            rename_map["bayesian_model_margin_dem_capped"] = "bayesian_model_margin_dem"
        elif "posterior_margin_dem" in bayes.columns:
            rename_map["posterior_margin_dem"] = "bayesian_model_margin_dem"

        if "bayesian_posterior_sd_calibrated" in bayes.columns:
            rename_map["bayesian_posterior_sd_calibrated"] = "bayesian_posterior_sd"
        elif "posterior_sd" in bayes.columns:
            rename_map["posterior_sd"] = "bayesian_posterior_sd"

        keep = ["state"]

        for old, new in rename_map.items():
            keep.append(old)

        # Also carry useful audit fields.
        for c in [
            "poll_count",
            "polling_margin_used",
            "original_bayesian_polling_weight",
            "poll_count_weight_multiplier",
            "cycle_max_polling_weight",
            "bayesian_fundamentals_weight_capped",
        ]:
            if c in bayes.columns:
                keep.append(c)

        bayes_keep = bayes[keep].rename(columns=rename_map)

        races = races.merge(
            bayes_keep,
            on="state",
            how="left",
            suffixes=("", "_from_generated_bayes"),
        )

        for col in [
            "bayesian_polling_weight",
            "bayesian_model_margin_dem",
            "bayesian_posterior_sd",
            "poll_count",
            "polling_margin_used",
            "original_bayesian_polling_weight",
            "poll_count_weight_multiplier",
            "cycle_max_polling_weight",
            "bayesian_fundamentals_weight_capped",
        ]:
            generated = f"{col}_from_generated_bayes"
            if generated in races.columns:
                races[col] = races[generated].combine_first(races.get(col))
                races = races.drop(columns=[generated])

    # ------------------------------------------------------------
    # 3. Fix final candidate-quality adjustment from component fields.
    # ------------------------------------------------------------
    for col, default in [
        ("objective_candidate_quality_adjustment_dem", 0.0),
        ("manual_candidate_quality_adjustment_dem", 0.0),
        ("candidate_quality_gate", 1.0),
    ]:
        if col not in races.columns:
            races[col] = default

    objective = to_num(races["objective_candidate_quality_adjustment_dem"], 0.0)
    manual = to_num(races["manual_candidate_quality_adjustment_dem"], 0.0)
    gate = to_num(races["candidate_quality_gate"], 1.0).clip(lower=0.0, upper=1.0)

    races["candidate_quality_adjustment_dem"] = (objective + manual) * gate

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------
    races.to_csv(RACE_INPUTS, index=False)

    print("Synced Senate generated fields into inputs/race_inputs.csv")

    if "state" in races.columns:
        me = races[races["state"] == "ME"]
        if not me.empty:
            row = me.iloc[0]
            print()
            print("Maine check")
            print("-----------")
            for col in [
                "polling_margin_dem",
                "poll_count",
                "bayesian_polling_weight",
                "bayesian_model_margin_dem",
                "objective_candidate_quality_adjustment_dem",
                "manual_candidate_quality_adjustment_dem",
                "candidate_quality_gate",
                "candidate_quality_adjustment_dem",
                "fundamentals_margin_dem",
            ]:
                if col in races.columns:
                    print(f"{col}: {row.get(col)}")


if __name__ == "__main__":
    main()
