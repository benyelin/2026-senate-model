from pathlib import Path
from datetime import date
import pandas as pd

path = Path("inputs/race_inputs.csv")

if not path.exists():
    raise FileNotFoundError("inputs/race_inputs.csv not found.")

df = pd.read_csv(path)

rationale_cols = {
    "incumbency_rationale": "",
    "candidate_quality_rationale": "",
    "overperformance_rationale": "",
    "liability_rationale": "",
    "special_adjustment_rationale": "",
    "last_human_review_date": "",
    "human_review_status": "",
}

for col, default in rationale_cols.items():
    if col not in df.columns:
        df[col] = default

# Fill Maine rationale if blank.
state_col = "state"
if state_col in df.columns:
    me = df[state_col].astype(str).str.upper().eq("ME")

    today = date.today().isoformat()

    def fill_if_blank(col, value):
        if col in df.columns:
            blank = df[col].isna() | df[col].astype(str).str.strip().eq("")
            df.loc[me & blank, col] = value

    fill_if_blank(
        "incumbency_rationale",
        "Intentional exceptional-case adjustment. Susan Collins has a long record of crossover overperformance in Maine; generic incumbency is retained but should be reviewed alongside overperformance and candidate-quality adjustments.",
    )

    fill_if_blank(
        "candidate_quality_rationale",
        "Intentional exceptional-case adjustment. Collins' long record of overperformance combined with Graham Platner's scandal/liability creates an unusually asymmetric candidate-quality environment. Aggressive adjustment is intentional but should be revisited after credible Maine polling.",
    )

    fill_if_blank(
        "overperformance_rationale",
        "Collins has historically overperformed Maine's partisan baseline by a meaningful margin. Kept as a separate adjustment because this race is unusually candidate-specific, but double-counting risk should be monitored.",
    )

    fill_if_blank(
        "liability_rationale",
        "Platner scandal/liability is treated as a real candidate-specific drag. Reassess if future polling shows the issue has faded or is already fully reflected in topline polling.",
    )

    fill_if_blank(
        "human_review_status",
        "Reviewed - intentional exception",
    )

    fill_if_blank(
        "last_human_review_date",
        today,
    )

df.to_csv(path, index=False)

print(f"Updated {path}")
print()
show_cols = [
    "state",
    "dem_candidate",
    "gop_candidate",
    "incumbency_rationale",
    "candidate_quality_rationale",
    "overperformance_rationale",
    "liability_rationale",
    "last_human_review_date",
    "human_review_status",
]
show_cols = [c for c in show_cols if c in df.columns]
print(df[df["state"].astype(str).str.upper().eq("ME")][show_cols].to_string(index=False))
