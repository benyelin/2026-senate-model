from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

INPUT_DIR = Path("inputs")
ELECTION_DAY = date(2026, 11, 3)


def compute_days_out():
    return max(0, (ELECTION_DAY - date.today()).days)


def max_polling_weight_for_days_out(days_out):
    if days_out > 180:
        return 0.12
    if days_out > 120:
        return 0.18
    if days_out > 60:
        return 0.35
    if days_out > 30:
        return 0.50
    return 0.70


def poll_count_multiplier(poll_count):
    try:
        poll_count = float(poll_count)
    except Exception:
        poll_count = 0

    if poll_count <= 0:
        return 0.0
    if poll_count == 1:
        return 0.30
    if poll_count == 2:
        return 0.55
    if poll_count == 3:
        return 0.75
    return 1.0


def minimum_uncertainty_for_days_out(days_out):
    if days_out > 180:
        return 7.5
    if days_out > 120:
        return 8.0
    if days_out > 60:
        return 5.5
    if days_out > 30:
        return 4.5
    return 3.5


def load_race_inputs():
    race_path = INPUT_DIR / "race_inputs.csv"

    if not race_path.exists():
        raise FileNotFoundError("Could not find inputs/race_inputs.csv")

    races = pd.read_csv(race_path)

    if "state" not in races.columns:
        raise ValueError("race_inputs.csv missing state column")

    races["state"] = races["state"].astype(str).str.strip().str.upper()

    return races


def main():
    days_out = compute_days_out()
    max_cycle_cap = max_polling_weight_for_days_out(days_out)
    min_sd = minimum_uncertainty_for_days_out(days_out)

    bayes_path = INPUT_DIR / "bayesian_update_generated.csv"

    if not bayes_path.exists():
        raise FileNotFoundError("Could not find inputs/bayesian_update_generated.csv")

    df = pd.read_csv(bayes_path)
    races = load_race_inputs()

    # Repair missing state column using race_inputs row order.
    if "state" not in df.columns:
        if len(df) != len(races):
            raise ValueError(
                "bayesian_update_generated.csv has no state column, and row count "
                f"does not match race_inputs.csv. bayes rows={len(df)}, race rows={len(races)}"
            )

        df.insert(0, "state", races["state"].values)
        print("Reconstructed missing state column from race_inputs.csv row order.")

    df["state"] = df["state"].astype(str).str.strip().str.upper()

    required_cols = [
        "prior_margin_dem",
        "polling_margin_used",
        "posterior_margin_dem",
        "posterior_sd",
        "bayesian_prior_weight",
        "bayesian_polling_weight",
        "poll_count",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"bayesian_update_generated.csv missing required columns: {missing}")

    print("\nCalibration settings:")
    print(f"days_out: {days_out}")
    print(f"cycle max polling cap: {max_cycle_cap}")
    print(f"minimum posterior SD: {min_sd}")

    df["prior_margin_dem"] = pd.to_numeric(
        df["prior_margin_dem"],
        errors="coerce"
    )

    df["polling_margin_used"] = pd.to_numeric(
        df["polling_margin_used"],
        errors="coerce"
    )

    df["posterior_margin_dem"] = pd.to_numeric(
        df["posterior_margin_dem"],
        errors="coerce"
    )

    df["posterior_sd"] = pd.to_numeric(
        df["posterior_sd"],
        errors="coerce"
    )

    df["bayesian_polling_weight"] = pd.to_numeric(
        df["bayesian_polling_weight"],
        errors="coerce"
    ).fillna(0.0)

    df["poll_count"] = pd.to_numeric(
        df["poll_count"],
        errors="coerce"
    ).fillna(0.0)

    df["original_bayesian_polling_weight"] = df["bayesian_polling_weight"]
    df["original_bayesian_posterior_sd"] = df["posterior_sd"]

    df["poll_count_weight_multiplier"] = df["poll_count"].apply(
        poll_count_multiplier
    )

    df["cycle_max_polling_weight"] = max_cycle_cap

    df["bayesian_polling_weight_capped"] = np.minimum(
        df["original_bayesian_polling_weight"],
        df["cycle_max_polling_weight"] * df["poll_count_weight_multiplier"]
    )

    df["bayesian_polling_weight_capped"] = (
        df["bayesian_polling_weight_capped"]
        .clip(lower=0.0, upper=1.0)
    )

    # If polling is missing or there are zero polls, polling should not affect the race.
    df.loc[
        df["polling_margin_used"].isna() | (df["poll_count"] <= 0),
        "bayesian_polling_weight_capped"
    ] = 0.0

    df["bayesian_fundamentals_weight_capped"] = (
        1.0 - df["bayesian_polling_weight_capped"]
    )

    df["bayesian_model_margin_dem_capped"] = (
        df["prior_margin_dem"] * df["bayesian_fundamentals_weight_capped"]
        + df["polling_margin_used"] * df["bayesian_polling_weight_capped"]
    )

    # Calibrated uncertainty floor.
    df["bayesian_posterior_sd_calibrated"] = (
        df["posterior_sd"]
        .fillna(0.0)
        .clip(lower=min_sd)
    )

    # Add a little extra uncertainty for RCV/top-four races using race_inputs mapping.
    if "election_system" in races.columns:
        election_map = dict(
            zip(
                races["state"],
                races["election_system"].fillna("plurality").astype(str).str.lower()
            )
        )

        df["election_system"] = df["state"].map(election_map).fillna("plurality")

        is_rcv = df["election_system"].str.contains("rcv", na=False)

        df.loc[
            is_rcv,
            "bayesian_posterior_sd_calibrated"
        ] = df.loc[
            is_rcv,
            "bayesian_posterior_sd_calibrated"
        ] + 0.75
    else:
        df["election_system"] = "plurality"

    # Replace operational columns.
    df["bayesian_polling_weight"] = df["bayesian_polling_weight_capped"]
    df["bayesian_prior_weight"] = df["bayesian_fundamentals_weight_capped"]
    df["posterior_margin_dem"] = df["bayesian_model_margin_dem_capped"]
    df["posterior_sd"] = df["bayesian_posterior_sd_calibrated"]

    df.to_csv(bayes_path, index=False)

    print(f"\nApplied early-cycle calibration to {bayes_path}")

    # Patch generated Bayesian fields into race_inputs without touching core fundamentals.
    margin_map = dict(zip(df["state"], df["posterior_margin_dem"]))
    weight_map = dict(zip(df["state"], df["bayesian_polling_weight"]))
    sd_map = dict(zip(df["state"], df["posterior_sd"]))

    races["bayesian_model_margin_dem"] = races["state"].map(margin_map)
    races["bayesian_polling_weight"] = races["state"].map(weight_map)
    races["bayesian_posterior_sd"] = races["state"].map(sd_map)

    race_path = INPUT_DIR / "race_inputs.csv"
    races.to_csv(race_path, index=False)

    print(f"Updated calibrated Bayesian fields in {race_path}")


if __name__ == "__main__":
    main()
