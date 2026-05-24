
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from senate_model.engine import ModelConfig, run_forecast


def run_scenarios(input_dir="inputs", output_dir="outputs", swings=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    swings = swings if swings is not None else [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    rows = []

    for swing in swings:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            shutil.copytree(input_dir, tmp / "inputs")
            national_path = tmp / "inputs" / "national_environment.csv"
            national = pd.read_csv(national_path)

            if "manual_adjustment" in national["parameter"].values:
                national.loc[national["parameter"] == "manual_adjustment", "value"] = swing
            else:
                national = pd.concat(
                    [national, pd.DataFrame([{"parameter": "manual_adjustment", "value": swing}])],
                    ignore_index=True,
                )

            national.to_csv(national_path, index=False)

            result = run_forecast(tmp / "inputs", tmp / "outputs", ModelConfig(n_sims=10000, seed=20260522))
            summary = result["summary"]
            rows.append({
                "manual_national_swing_dem": swing,
                "expected_dem_seats": summary["expected_dem_seats"],
                "dem_control_probability": summary["dem_control_probability"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "scenario_summary.csv", index=False)
    return out


if __name__ == "__main__":
    print(run_scenarios().to_string(index=False))
