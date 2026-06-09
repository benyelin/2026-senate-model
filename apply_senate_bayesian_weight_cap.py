from pathlib import Path
import argparse
import pandas as pd
import numpy as np

BAYES_PATH = Path("inputs/bayesian_update_generated.csv")


def dynamic_polling_cap(days_out: float) -> float:
    """
    Maximum race-specific Bayesian polling weight by days until Election Day.

    Conservative early-cycle curve:
    - 150+ days out: about 0.18
    - 120 days out: about 0.25
    - 90 days out: about 0.35
    - 60 days out: about 0.50
    - 30 days out: about 0.68
    - 14 days out: about 0.80
    - Final week: about 0.90
    """

    anchors = [
        (180, 0.15),
        (150, 0.18),
        (120, 0.25),
        (90, 0.35),
        (60, 0.50),
        (30, 0.68),
        (14, 0.80),
        (7, 0.88),
        (0, 0.92),
    ]

    days_out = max(0.0, float(days_out))

    # If farther out than the largest anchor, use the most conservative cap.
    if days_out >= anchors[0][0]:
        return anchors[0][1]

    # Interpolate between anchors.
    for i in range(len(anchors) - 1):
        high_day, high_cap = anchors[i]
        low_day, low_cap = anchors[i + 1]

        if high_day >= days_out >= low_day:
            span = high_day - low_day
            if span == 0:
                return low_cap

            # As days_out falls, cap rises.
            t = (high_day - days_out) / span
            return high_cap + t * (low_cap - high_cap)

    return anchors[-1][1]


def infer_days_out_from_config() -> float:
    """
    Best-effort fallback if --days-out is not supplied.
    Keeps current behavior safe if the script is run manually.
    """

    config_candidates = [
        Path("inputs/model_config.csv"),
        Path("inputs/config.csv"),
        Path("inputs/election_config.csv"),
    ]

    for path in config_candidates:
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        # Common setting/value format.
        if {"setting", "value"}.issubset(df.columns):
            mask = df["setting"].astype(str).str.lower().str.strip().eq("days_out")
            if mask.any():
                try:
                    return float(df.loc[mask, "value"].iloc[0])
                except Exception:
                    pass

        # Common key/value format.
        if {"key", "value"}.issubset(df.columns):
            mask = df["key"].astype(str).str.lower().str.strip().eq("days_out")
            if mask.any():
                try:
                    return float(df.loc[mask, "value"].iloc[0])
                except Exception:
                    pass

    # Safe fallback for current stage of the cycle.
    return 148.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-out", type=float, default=None)
    args = parser.parse_args()

    days_out = args.days_out if args.days_out is not None else infer_days_out_from_config()
    cycle_max_polling_weight = dynamic_polling_cap(days_out)

    if not BAYES_PATH.exists():
        raise FileNotFoundError("inputs/bayesian_update_generated.csv not found.")

    bayes = pd.read_csv(BAYES_PATH)

    required = [
        "state",
        "prior_margin_dem",
        "polling_margin_used",
        "bayesian_polling_weight",
    ]

    missing = [c for c in required if c not in bayes.columns]
    if missing:
        raise ValueError(f"Missing required Bayesian columns: {missing}")

    bayes["state"] = bayes["state"].astype(str).str.strip().str.upper()

    bayes["original_bayesian_polling_weight"] = pd.to_numeric(
        bayes["bayesian_polling_weight"],
        errors="coerce",
    )

    bayes["bayesian_polling_weight_capped"] = bayes[
        "original_bayesian_polling_weight"
    ].clip(lower=0, upper=cycle_max_polling_weight)

    bayes["bayesian_fundamentals_weight_capped"] = (
        1.0 - bayes["bayesian_polling_weight_capped"]
    )

    prior = pd.to_numeric(bayes["prior_margin_dem"], errors="coerce")
    poll = pd.to_numeric(bayes["polling_margin_used"], errors="coerce")
    w = pd.to_numeric(bayes["bayesian_polling_weight_capped"], errors="coerce").fillna(0)

    bayes["bayesian_model_margin_dem_uncapped"] = pd.to_numeric(
        bayes.get("posterior_margin_dem", np.nan),
        errors="coerce",
    )

    bayes["bayesian_model_margin_dem_capped"] = poll * w + prior * (1.0 - w)
    bayes["posterior_margin_dem_capped"] = bayes["bayesian_model_margin_dem_capped"]
    bayes["cycle_max_polling_weight"] = cycle_max_polling_weight
    bayes["days_out_for_polling_cap"] = days_out
    bayes["bayesian_cap_applied"] = (
        bayes["original_bayesian_polling_weight"] > cycle_max_polling_weight
    )

    bayes.to_csv(BAYES_PATH, index=False)

    print("Applied Senate dynamic Bayesian polling cap")
    print("------------------------------------------")
    print(f"Days out: {days_out:.1f}")
    print(f"Cycle max polling weight: {cycle_max_polling_weight:.3f}")
    print()

    show_cols = [
        "state",
        "prior_margin_dem",
        "polling_margin_used",
        "original_bayesian_polling_weight",
        "bayesian_polling_weight_capped",
        "bayesian_model_margin_dem_uncapped",
        "bayesian_model_margin_dem_capped",
        "bayesian_cap_applied",
    ]
    show_cols = [c for c in show_cols if c in bayes.columns]
    print(bayes[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
