from pathlib import Path
import pandas as pd

RACE_INPUTS = Path("inputs/race_inputs.csv")


def to_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def main():
    if not RACE_INPUTS.exists():
        raise FileNotFoundError("inputs/race_inputs.csv not found.")

    df = pd.read_csv(RACE_INPUTS)

    for col, default in [
        ("objective_candidate_quality_adjustment_dem", 0.0),
        ("manual_candidate_quality_adjustment_dem", 0.0),
        ("candidate_quality_gate", 1.0),
    ]:
        if col not in df.columns:
            df[col] = default

    objective = to_num(df["objective_candidate_quality_adjustment_dem"], 0.0)
    manual = to_num(df["manual_candidate_quality_adjustment_dem"], 0.0)
    gate = to_num(df["candidate_quality_gate"], 1.0).clip(lower=0.0, upper=1.0)

    df["candidate_quality_adjustment_dem"] = (objective + manual) * gate

    df.to_csv(RACE_INPUTS, index=False)

    print("Synced candidate_quality_adjustment_dem from objective/manual/gate fields.")

    if "state" in df.columns:
        me = df[df["state"].astype(str).str.upper() == "ME"]
        if not me.empty:
            row = me.iloc[0]
            print()
            print("Maine candidate quality check:")
            for col in [
                "objective_candidate_quality_adjustment_dem",
                "manual_candidate_quality_adjustment_dem",
                "candidate_quality_gate",
                "candidate_quality_adjustment_dem",
            ]:
                if col in df.columns:
                    print(f"{col}: {row.get(col)}")


if __name__ == "__main__":
    main()
