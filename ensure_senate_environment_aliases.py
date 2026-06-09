from pathlib import Path
import pandas as pd

path = Path("inputs/race_inputs.csv")

if not path.exists():
    raise FileNotFoundError("inputs/race_inputs.csv not found.")

df = pd.read_csv(path)

if "state_environment_adjustment_dem" not in df.columns:
    print("state_environment_adjustment_dem not found; no aliases added.")
else:
    # This column is not a discretionary/manual state adjustment.
    # It is the national environment applied through each state's elasticity.
    df["national_environment_adjustment_dem"] = df["state_environment_adjustment_dem"]
    df["state_scaled_national_environment_adjustment_dem"] = df["state_environment_adjustment_dem"]

    df.to_csv(path, index=False)

    print("Added Senate environment aliases:")
    print("- national_environment_adjustment_dem")
    print("- state_scaled_national_environment_adjustment_dem")
    print()
    show_cols = [
        "state",
        "state_partisan_baseline_dem",
        "state_environment_adjustment_dem",
        "national_environment_adjustment_dem",
        "state_scaled_national_environment_adjustment_dem",
        "fundamentals_margin_dem",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string(index=False))
