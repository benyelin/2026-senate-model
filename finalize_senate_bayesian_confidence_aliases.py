from pathlib import Path
import pandas as pd

INPUTS = Path("inputs")
FILES = [
    INPUTS / "race_inputs.csv",
    INPUTS / "bayesian_update_generated.csv",
]

def sync_file(path):
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return

    df = pd.read_csv(path)

    changed = False

    # If polling-confidence accelerator has run, the capped fields are the authoritative values.
    if "polling_confidence_boost" in df.columns:
        if "bayesian_polling_weight_capped" in df.columns:
            df["bayesian_polling_weight"] = df["bayesian_polling_weight_capped"]
            changed = True

        if "bayesian_model_margin_dem_capped" in df.columns:
            df["bayesian_model_margin_dem"] = df["bayesian_model_margin_dem_capped"]
            df["posterior_margin_dem"] = df["bayesian_model_margin_dem_capped"]
            df["posterior_margin_dem_capped"] = df["bayesian_model_margin_dem_capped"]
            changed = True

        if "bayesian_polling_weight_capped" in df.columns:
            df["bayesian_prior_weight"] = 1.0 - pd.to_numeric(
                df["bayesian_polling_weight_capped"], errors="coerce"
            ).fillna(0.0)
            df["bayesian_fundamentals_weight_capped"] = df["bayesian_prior_weight"]
            changed = True

    if changed:
        df.to_csv(path, index=False)
        print(f"Synced Bayesian aliases in {path}")
    else:
        print(f"No polling-confidence fields found to sync in {path}")

for path in FILES:
    sync_file(path)
