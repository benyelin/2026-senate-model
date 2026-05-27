from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR = Path("inputs")
MAX_POLLING_WEIGHT = 0.25


def pick_col(df, possible_names, fallback_index=None):
    """
    Find a column by name, or fall back to position.
    """
    for name in possible_names:
        if name in df.columns:
            return name

    if fallback_index is not None and fallback_index < len(df.columns):
        return df.columns[fallback_index]

    return None


def main():
    path = INPUT_DIR / "bayesian_update_generated.csv"

    if not path.exists():
        print(f"No {path} found.")
        return

    df = pd.read_csv(path)

    print("Detected Bayesian columns:")
    print(list(df.columns))

    # Your current file appears to be structured roughly like:
    # state, fundamentals/prior margin, polling margin, model/posterior margin, ...
    state_col = pick_col(df, ["state"], fallback_index=0)
    fundamentals_col = pick_col(
        df,
        [
            "fundamentals_margin_dem",
            "prior_margin_dem",
            "pre_bayes_model_margin_dem",
            "baseline_margin_dem",
        ],
        fallback_index=1,
    )
    polling_col = pick_col(
        df,
        [
            "polling_margin_dem",
            "manual_polling_margin_dem",
            "bayesian_polling_margin_dem",
        ],
        fallback_index=2,
    )
    model_col = pick_col(
        df,
        [
            "bayesian_model_margin_dem",
            "posterior_margin_dem",
            "model_margin_dem",
            "final_margin_dem",
        ],
        fallback_index=3,
    )

    # In your file, the old/crazy polling weight appears to be the 9th column:
    # e.g. 0.9113968023379431
    polling_weight_col = pick_col(
        df,
        [
            "bayesian_polling_weight",
            "polling_weight",
            "poll_weight",
        ],
        fallback_index=8,
    )

    fundamentals_weight_col = pick_col(
        df,
        [
            "fundamentals_weight",
            "prior_weight",
        ],
        fallback_index=7,
    )

    if state_col is None or fundamentals_col is None or polling_col is None or model_col is None:
        raise ValueError(
            "Could not identify required Bayesian columns. "
            f"state={state_col}, fundamentals={fundamentals_col}, polling={polling_col}, model={model_col}"
        )

    print("\nUsing columns:")
    print(f"state: {state_col}")
    print(f"fundamentals/prior margin: {fundamentals_col}")
    print(f"polling margin: {polling_col}")
    print(f"model/posterior margin: {model_col}")
    print(f"polling weight: {polling_weight_col}")
    print(f"fundamentals weight: {fundamentals_weight_col}")

    df[fundamentals_col] = pd.to_numeric(df[fundamentals_col], errors="coerce")
    df[polling_col] = pd.to_numeric(df[polling_col], errors="coerce")

    if polling_weight_col is not None:
        df[polling_weight_col] = pd.to_numeric(
            df[polling_weight_col],
            errors="coerce"
        ).fillna(0.0)

        df["original_bayesian_polling_weight"] = df[polling_weight_col]
    else:
        df["original_bayesian_polling_weight"] = 0.0
        polling_weight_col = "bayesian_polling_weight"

    df["bayesian_polling_weight_capped"] = (
        df["original_bayesian_polling_weight"]
        .clip(lower=0.0, upper=MAX_POLLING_WEIGHT)
    )

    # If no polling margin, polling should have zero effect.
    df.loc[
        df[polling_col].isna(),
        "bayesian_polling_weight_capped"
    ] = 0.0

    df["bayesian_fundamentals_weight_capped"] = (
        1.0 - df["bayesian_polling_weight_capped"]
    )

    df["bayesian_model_margin_dem_capped"] = (
        df[fundamentals_col] * df["bayesian_fundamentals_weight_capped"]
        + df[polling_col] * df["bayesian_polling_weight_capped"]
    )

    # Replace the model/posterior margin with the capped blend.
    df[model_col] = df["bayesian_model_margin_dem_capped"]

    # Replace visible weight columns.
    if polling_weight_col in df.columns:
        df[polling_weight_col] = df["bayesian_polling_weight_capped"]

    if fundamentals_weight_col in df.columns:
        df[fundamentals_weight_col] = df["bayesian_fundamentals_weight_capped"]

    # If named columns exist, update them too.
    for col in [
        "bayesian_polling_weight",
        "polling_weight",
        "poll_weight",
    ]:
        if col in df.columns:
            df[col] = df["bayesian_polling_weight_capped"]

    for col in [
        "fundamentals_weight",
        "prior_weight",
    ]:
        if col in df.columns:
            df[col] = df["bayesian_fundamentals_weight_capped"]

    for col in [
        "bayesian_model_margin_dem",
        "posterior_margin_dem",
        "model_margin_dem",
        "final_margin_dem",
    ]:
        if col in df.columns:
            df[col] = df["bayesian_model_margin_dem_capped"]

    df.to_csv(path, index=False)

    print(f"\nApplied Bayesian polling weight cap of {MAX_POLLING_WEIGHT:.2f}")
    print(f"Wrote {path}")

    # Also patch race_inputs.csv if it contains generated Bayesian columns.
    race_path = INPUT_DIR / "race_inputs.csv"

    if race_path.exists():
        races = pd.read_csv(race_path)

        if "state" in races.columns and state_col in df.columns:
            races["state"] = races["state"].astype(str).str.strip().str.upper()
            df[state_col] = df[state_col].astype(str).str.strip().str.upper()

            update = df[
                [
                    state_col,
                    "bayesian_polling_weight_capped",
                    "bayesian_model_margin_dem_capped",
                ]
            ].copy()

            update = update.rename(
                columns={
                    state_col: "state",
                    "bayesian_polling_weight_capped": "bayesian_polling_weight",
                    "bayesian_model_margin_dem_capped": "bayesian_model_margin_dem",
                }
            )

            for col in ["bayesian_polling_weight", "bayesian_model_margin_dem"]:
                if col in races.columns:
                    races = races.drop(columns=[col])

            races = races.merge(update, on="state", how="left")
            races.to_csv(race_path, index=False)

            print(f"Updated capped Bayesian fields in {race_path}")


if __name__ == "__main__":
    main()
