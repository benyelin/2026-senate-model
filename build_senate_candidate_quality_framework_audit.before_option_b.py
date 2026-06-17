from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

RACE_INPUTS = INPUTS / "race_inputs.csv"
WAR_AUDIT = OUTPUTS / "senate_candidate_war_audit.csv"
OUT = OUTPUTS / "senate_candidate_quality_framework_audit.csv"

# Framework v1 settings.
#
# Important: proposed_war_adjustment_dem in senate_candidate_war_audit.csv
# already applies Senate-specific shrinkage/transfer logic from the WAR audit script.
# This application multiplier is an additional conservatism layer before using it
# as a live candidate-quality adjustment.
WAR_APPLICATION_MULTIPLIER = 0.50
FRAMEWORK_CAP = 1.50

# Mechanical previous statewide winner bonus.
# Positive means Dem advantage; negative means GOP advantage.
STATEWIDE_WIN_BONUS = 0.75

# Scandal/liability is the only intended manual adjustment bucket.
# For now, it defaults to 0 unless race_inputs.csv later contains an explicit field.
DEFAULT_SCANDAL_ADJUSTMENT_DEM = 0.0


def safe_num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def clean_text(series):
    return series.fillna("").astype(str).replace({"nan": "", "None": "", "NaN": ""})


def safe_bool(series, default=False):
    return (
        series.fillna(default)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def pick_col(df, candidates, default=None):
    for col in candidates:
        if col in df.columns:
            return col
    return default


def classify_recommendation(row):
    current = row["current_model_candidate_adjustment_dem"]
    framework = row["framework_candidate_quality_adjustment_dem"]
    gap = row["framework_minus_current_adjustment_dem"]
    priority = str(row.get("war_review_priority", "")).lower()
    match_status = str(row.get("war_match_status", ""))

    if abs(gap) < 0.50:
        return "Keep current adjustment; framework difference is small."

    if match_status == "Neither matched":
        return "Do not use WAR mechanically; no candidate WAR match."

    if "high" in priority and abs(gap) >= 1.50:
        return "High-priority human review; current adjustment materially differs from framework."

    if gap < -0.50:
        return "Consider reducing current Dem candidate-quality adjustment."

    if gap > 0.50:
        return "Consider increasing current Dem candidate-quality adjustment."

    return "Review."


def main():
    if not RACE_INPUTS.exists():
        raise FileNotFoundError(f"Missing {RACE_INPUTS}")

    # Build/refresh WAR audit first if needed.
    if not WAR_AUDIT.exists():
        raise FileNotFoundError(
            f"Missing {WAR_AUDIT}. Run: python3 build_senate_candidate_war_audit.py"
        )

    races = pd.read_csv(RACE_INPUTS)
    war = pd.read_csv(WAR_AUDIT)

    for df in [races, war]:
        if "state" not in df.columns:
            raise ValueError("Both race_inputs.csv and senate_candidate_war_audit.csv must contain state.")

        df["state"] = df["state"].astype(str).str.upper().str.strip()

    # Current candidate adjustment column names vary across versions.
    current_col = pick_col(
        war,
        [
            "current_model_candidate_adjustment_dem",
            "current_candidate_quality_adjustment_dem",
            "candidate_quality_adjustment_dem",
        ],
    )

    if current_col is None:
        # Fall back to race_inputs if audit doesn't have the current value.
        current_col_races = pick_col(
            races,
            [
                "candidate_quality_adjustment_dem",
                "candidate_adjustment_dem",
                "candidate_quality_dem",
            ],
        )

        if current_col_races is None:
            races["current_model_candidate_adjustment_dem"] = 0.0
        else:
            races["current_model_candidate_adjustment_dem"] = safe_num(races[current_col_races], 0.0)

        current_for_merge = races[["state", "current_model_candidate_adjustment_dem"]].copy()
    else:
        war["current_model_candidate_adjustment_dem"] = safe_num(war[current_col], 0.0)
        current_for_merge = war[["state", "current_model_candidate_adjustment_dem"]].copy()

    # WAR proposed adjustment.
    if "proposed_war_adjustment_dem" not in war.columns:
        raise ValueError("senate_candidate_war_audit.csv must contain proposed_war_adjustment_dem.")

    war["proposed_war_adjustment_dem"] = safe_num(war["proposed_war_adjustment_dem"], 0.0)

    keep_war_cols = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "war_match_status",
        "proposed_war_adjustment_dem",
        "war_review_priority",
        "dem_war_name",
        "gop_war_name",
        "dem_candidate_war",
        "gop_candidate_war",
        "candidate_war_net_dem",
        "dem_war_cycles",
        "gop_war_cycles",
    ]

    keep_war_cols = [c for c in keep_war_cols if c in war.columns]

    out = war[keep_war_cols].copy()

    out = out.merge(
        current_for_merge.drop_duplicates("state"),
        on="state",
        how="left",
    )

    out["current_model_candidate_adjustment_dem"] = safe_num(
        out["current_model_candidate_adjustment_dem"],
        0.0,
    )

    # Mechanical framework v2.
    # Formula:
    # candidate_quality_adjustment_dem =
    #   candidate_war_adjustment_dem
    # + statewide_win_bonus_dem
    # + candidate_scandal_adjustment_dem
    out["candidate_war_adjustment_dem_raw"] = out["proposed_war_adjustment_dem"]
    out["candidate_war_application_multiplier"] = WAR_APPLICATION_MULTIPLIER
    out["candidate_war_adjustment_dem"] = (
        out["candidate_war_adjustment_dem_raw"] * WAR_APPLICATION_MULTIPLIER
    ).clip(lower=-FRAMEWORK_CAP, upper=FRAMEWORK_CAP)

    # Merge previous statewide winner fields from race_inputs.csv.
    statewide_cols = [
        "state",
        "dem_previous_statewide_winner",
        "gop_previous_statewide_winner",
        "previous_statewide_winner_notes",
        "candidate_scandal_adjustment_dem",
        "candidate_scandal_rationale",
    ]
    statewide_cols = [c for c in statewide_cols if c in races.columns]

    if len(statewide_cols) > 1:
        out = out.merge(
            races[statewide_cols].drop_duplicates("state"),
            on="state",
            how="left",
        )

    if "dem_previous_statewide_winner" not in out.columns:
        out["dem_previous_statewide_winner"] = False
    if "gop_previous_statewide_winner" not in out.columns:
        out["gop_previous_statewide_winner"] = False

    out["dem_previous_statewide_winner"] = safe_bool(out["dem_previous_statewide_winner"], False)
    out["gop_previous_statewide_winner"] = safe_bool(out["gop_previous_statewide_winner"], False)

    out["statewide_win_bonus_dem"] = 0.0
    out.loc[
        out["dem_previous_statewide_winner"] & ~out["gop_previous_statewide_winner"],
        "statewide_win_bonus_dem",
    ] = STATEWIDE_WIN_BONUS
    out.loc[
        out["gop_previous_statewide_winner"] & ~out["dem_previous_statewide_winner"],
        "statewide_win_bonus_dem",
    ] = -STATEWIDE_WIN_BONUS

    if "candidate_scandal_adjustment_dem" in out.columns:
        out["candidate_scandal_adjustment_dem"] = safe_num(out["candidate_scandal_adjustment_dem"], 0.0)
    else:
        out["candidate_scandal_adjustment_dem"] = DEFAULT_SCANDAL_ADJUSTMENT_DEM

    if "candidate_scandal_rationale" not in out.columns:
        out["candidate_scandal_rationale"] = ""

    if "previous_statewide_winner_notes" not in out.columns:
        out["previous_statewide_winner_notes"] = ""

    out["framework_candidate_quality_adjustment_dem"] = (
        out["candidate_war_adjustment_dem"]
        + out["statewide_win_bonus_dem"]
        + out["candidate_scandal_adjustment_dem"]
    ).clip(lower=-FRAMEWORK_CAP, upper=FRAMEWORK_CAP)

    out["framework_minus_current_adjustment_dem"] = (
        out["framework_candidate_quality_adjustment_dem"]
        - out["current_model_candidate_adjustment_dem"]
    )

    out["abs_framework_gap"] = out["framework_minus_current_adjustment_dem"].abs()

    out["framework_candidate_quality_method"] = (
        "Audit-only v2: candidate_quality_adjustment_dem = 50% of Senate WAR-audit proposed "
        "adjustment, capped at +/-1.5, plus +/-0.75 previous statewide winner bonus, plus "
        "manual scandal/liability adjustment if explicitly documented."
    )

    out["framework_review_recommendation"] = out.apply(classify_recommendation, axis=1)

    # Bring in existing rationale/status if available.
    rationale_cols = [
        "state",
        "candidate_quality_rationale",
        "candidate_scandal_rationale",
        "previous_statewide_winner_notes",
        "human_review_status",
        "last_human_review_date",
    ]

    rationale_cols = [c for c in rationale_cols if c in races.columns]

    if len(rationale_cols) > 1:
        out = out.merge(
            races[rationale_cols].drop_duplicates("state"),
            on="state",
            how="left",
        )

    for col in [
        "dem_candidate",
        "gop_candidate",
        "war_match_status",
        "war_review_priority",
        "dem_war_name",
        "gop_war_name",
        "dem_war_cycles",
        "gop_war_cycles",
        "candidate_quality_rationale",
        "candidate_scandal_rationale",
        "previous_statewide_winner_notes",
        "human_review_status",
        "last_human_review_date",
    ]:
        if col in out.columns:
            out[col] = clean_text(out[col])

    sort_cols = ["abs_framework_gap"]
    ascending = [False]

    if "war_review_priority" in out.columns:
        # Put High/Medium ahead in ties.
        priority_order = {"High": 3, "Medium": 2, "Low": 1}
        out["war_review_priority_rank"] = out["war_review_priority"].map(priority_order).fillna(0)
        sort_cols = ["war_review_priority_rank", "abs_framework_gap"]
        ascending = [False, False]

    out = out.sort_values(sort_cols, ascending=ascending)

    output_cols = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "war_match_status",
        "war_review_priority",
        "current_model_candidate_adjustment_dem",
        "proposed_war_adjustment_dem",
        "candidate_war_adjustment_dem",
        "dem_previous_statewide_winner",
        "gop_previous_statewide_winner",
        "statewide_win_bonus_dem",
        "candidate_scandal_adjustment_dem",
        "candidate_scandal_rationale",
        "previous_statewide_winner_notes",
        "framework_candidate_quality_adjustment_dem",
        "framework_minus_current_adjustment_dem",
        "abs_framework_gap",
        "framework_review_recommendation",
        "dem_candidate_war",
        "gop_candidate_war",
        "candidate_war_net_dem",
        "dem_war_name",
        "gop_war_name",
        "dem_war_cycles",
        "gop_war_cycles",
        "candidate_quality_rationale",
        "candidate_scandal_rationale",
        "previous_statewide_winner_notes",
        "human_review_status",
        "last_human_review_date",
        "framework_candidate_quality_method",
    ]

    output_cols = [c for c in output_cols if c in out.columns]

    out[output_cols].to_csv(OUT, index=False)

    print(f"Wrote {OUT}")
    print(f"Rows: {len(out)}")
    print()
    print("Largest framework/current gaps:")
    print(out[output_cols].head(20).to_string(index=False))

    print()
    print("Georgia row:")
    ga = out[out["state"].eq("GA")]
    if ga.empty:
        print("No GA row found.")
    else:
        print(ga[output_cols].to_string(index=False))


if __name__ == "__main__":
    main()
