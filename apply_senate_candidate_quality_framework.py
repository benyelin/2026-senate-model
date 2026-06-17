from pathlib import Path
import argparse
import pandas as pd

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

RACE_INPUTS = INPUTS / "race_inputs.csv"
FRAMEWORK_AUDIT = OUTPUTS / "senate_candidate_quality_framework_audit.csv"
DRY_RUN_OUT = OUTPUTS / "senate_candidate_quality_framework_apply_dry_run.csv"

LEGACY_CANDIDATE_COLS = [
    "overperformance_adjustment_dem",
    "candidate_liability_adjustment_dem",
    "objective_candidate_quality_adjustment_dem",
    "manual_candidate_quality_adjustment_dem",
    "special_adjustment_dem",
]


def safe_num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def clean_text(series):
    return series.fillna("").astype(str).replace({"nan": "", "None": "", "NaN": ""})


def ensure_col(df, col, default):
    if col not in df.columns:
        df[col] = default
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually update inputs/race_inputs.csv. Default is dry-run only.",
    )
    parser.add_argument(
        "--keep-legacy-candidate-cols",
        action="store_true",
        help="When applying, keep legacy candidate-side columns. Default is to zero them.",
    )
    args = parser.parse_args()

    if not RACE_INPUTS.exists():
        raise FileNotFoundError(f"Missing {RACE_INPUTS}")

    if not FRAMEWORK_AUDIT.exists():
        raise FileNotFoundError(
            f"Missing {FRAMEWORK_AUDIT}. Run build_senate_candidate_quality_framework_audit.py first."
        )

    races = pd.read_csv(RACE_INPUTS)
    audit = pd.read_csv(FRAMEWORK_AUDIT)

    races["state"] = races["state"].astype(str).str.upper().str.strip()
    audit["state"] = audit["state"].astype(str).str.upper().str.strip()

    needed = [
        "state",
        "candidate_war_adjustment_dem",
        "dem_previous_statewide_winner",
        "gop_previous_statewide_winner",
        "statewide_win_bonus_dem",
        "mechanical_candidate_adjustment_dem",
        "candidate_scandal_adjustment_dem",
        "candidate_scandal_rationale",
        "framework_candidate_quality_adjustment_dem",
        "framework_candidate_quality_method",
    ]

    missing = [c for c in needed if c not in audit.columns]
    if missing:
        raise ValueError(f"Framework audit is missing required columns: {missing}")

    # race_inputs scandal fields are the manual source of truth.
    ensure_col(races, "candidate_scandal_adjustment_dem", 0.0)
    ensure_col(races, "candidate_scandal_rationale", "")

    races["candidate_scandal_adjustment_dem"] = safe_num(
        races["candidate_scandal_adjustment_dem"], 0.0
    ).clip(lower=-3.0, upper=3.0)
    races["candidate_scandal_rationale"] = clean_text(races["candidate_scandal_rationale"])

    audit_for_merge = audit[needed].copy()

    # Replace audit scandal values with manual race_inputs values and recompute final.
    manual_scandal = races[
        ["state", "candidate_scandal_adjustment_dem", "candidate_scandal_rationale"]
    ].drop_duplicates("state")

    audit_for_merge = audit_for_merge.drop(
        columns=["candidate_scandal_adjustment_dem", "candidate_scandal_rationale"],
        errors="ignore",
    ).merge(manual_scandal, on="state", how="left")

    audit_for_merge["candidate_scandal_adjustment_dem"] = safe_num(
        audit_for_merge["candidate_scandal_adjustment_dem"], 0.0
    ).clip(lower=-3.0, upper=3.0)
    audit_for_merge["candidate_scandal_rationale"] = clean_text(
        audit_for_merge["candidate_scandal_rationale"]
    )

    audit_for_merge["mechanical_candidate_adjustment_dem"] = safe_num(
        audit_for_merge["mechanical_candidate_adjustment_dem"], 0.0
    ).clip(lower=-1.5, upper=1.5)

    audit_for_merge["proposed_candidate_quality_adjustment_dem"] = (
        audit_for_merge["mechanical_candidate_adjustment_dem"]
        + audit_for_merge["candidate_scandal_adjustment_dem"]
    ).clip(lower=-4.0, upper=4.0)

    audit_for_merge["candidate_quality_method"] = (
        "Framework v3 / Option B: mechanical_candidate_adjustment_dem = "
        "candidate_war_adjustment_dem + statewide_win_bonus_dem, capped at +/-1.5; "
        "candidate_scandal_adjustment_dem manual, capped at +/-3.0; "
        "final candidate_quality_adjustment_dem capped at +/-4.0."
    )
    audit_for_merge["candidate_quality_framework_version"] = (
        "senate_candidate_quality_framework_v3_option_b"
    )

    ensure_col(races, "candidate_quality_adjustment_dem", 0.0)
    races["candidate_quality_adjustment_dem"] = safe_num(
        races["candidate_quality_adjustment_dem"], 0.0
    )

    review = races.merge(
        audit_for_merge,
        on="state",
        how="left",
        suffixes=("", "_framework"),
    )

    review["current_candidate_quality_adjustment_dem"] = safe_num(
        review["candidate_quality_adjustment_dem"], 0.0
    )

    review["current_full_candidate_side_adjustment_dem"] = review[
        "current_candidate_quality_adjustment_dem"
    ]

    for col in LEGACY_CANDIDATE_COLS:
        if col in review.columns:
            review["current_full_candidate_side_adjustment_dem"] += safe_num(review[col], 0.0)

    review["proposed_candidate_quality_adjustment_dem"] = safe_num(
        review["proposed_candidate_quality_adjustment_dem"], 0.0
    )

    review["candidate_quality_change_dem"] = (
        review["proposed_candidate_quality_adjustment_dem"]
        - review["current_candidate_quality_adjustment_dem"]
    )
    review["full_candidate_side_change_dem"] = (
        review["proposed_candidate_quality_adjustment_dem"]
        - review["current_full_candidate_side_adjustment_dem"]
    )
    review["abs_candidate_quality_change"] = review["full_candidate_side_change_dem"].abs()

    def action_label(row):
        change = row["full_candidate_side_change_dem"]
        if abs(change) < 0.25:
            return "Minimal full-system change"
        if change > 0:
            return "Framework increases Dem candidate-side adjustment"
        return "Framework reduces Dem candidate-side adjustment"

    review["framework_apply_action"] = review.apply(action_label, axis=1)

    review_cols = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "current_candidate_quality_adjustment_dem",
        "current_full_candidate_side_adjustment_dem",
        "candidate_war_adjustment_dem",
        "statewide_win_bonus_dem",
        "mechanical_candidate_adjustment_dem",
        "candidate_scandal_adjustment_dem",
        "proposed_candidate_quality_adjustment_dem",
        "candidate_quality_change_dem",
        "full_candidate_side_change_dem",
        "abs_candidate_quality_change",
        "framework_apply_action",
        "candidate_scandal_rationale",
        "candidate_quality_method",
        "candidate_quality_framework_version",
    ]
    review_cols = [c for c in review_cols if c in review.columns]

    review = review.sort_values("abs_candidate_quality_change", ascending=False)
    review[review_cols].to_csv(DRY_RUN_OUT, index=False)

    print(f"Wrote dry-run review to {DRY_RUN_OUT}")
    print()
    print("Largest proposed full candidate-side changes:")
    print(review[review_cols].head(25).to_string(index=False))

    print()
    print("Georgia:")
    ga = review[review["state"].eq("GA")]
    print(ga[review_cols].to_string(index=False) if not ga.empty else "No GA row found.")

    print()
    print("Maine:")
    me = review[review["state"].eq("ME")]
    print(me[review_cols].to_string(index=False) if not me.empty else "No ME row found.")

    if not args.apply:
        print()
        print("DRY RUN ONLY. No changes made to inputs/race_inputs.csv.")
        print("To apply: python3 apply_senate_candidate_quality_framework.py --apply")
        print(
            "Default apply replaces the old candidate-side system by zeroing legacy candidate columns."
        )
        print(
            "To apply but keep legacy candidate columns: "
            "python3 apply_senate_candidate_quality_framework.py --apply --keep-legacy-candidate-cols"
        )
        return

    backup = INPUTS / "race_inputs.before_candidate_quality_framework_option_b_apply.csv"
    races.to_csv(backup, index=False)
    print()
    print(f"Backup written to {backup}")

    update_cols = [
        "candidate_war_adjustment_dem",
        "dem_previous_statewide_winner",
        "gop_previous_statewide_winner",
        "statewide_win_bonus_dem",
        "mechanical_candidate_adjustment_dem",
        "candidate_scandal_adjustment_dem",
        "candidate_scandal_rationale",
        "proposed_candidate_quality_adjustment_dem",
        "candidate_quality_method",
        "candidate_quality_framework_version",
    ]

    updates = audit_for_merge[["state"] + update_cols].copy()

    updated = races.merge(updates, on="state", how="left", suffixes=("", "_new"))

    destination_cols = [
        "candidate_war_adjustment_dem",
        "dem_previous_statewide_winner",
        "gop_previous_statewide_winner",
        "statewide_win_bonus_dem",
        "mechanical_candidate_adjustment_dem",
        "candidate_scandal_adjustment_dem",
        "candidate_scandal_rationale",
        "candidate_quality_adjustment_dem",
        "candidate_quality_method",
        "candidate_quality_framework_version",
    ]

    for col in destination_cols:
        if col not in updated.columns:
            updated[col] = ""

    for col in [
        "candidate_war_adjustment_dem",
        "dem_previous_statewide_winner",
        "gop_previous_statewide_winner",
        "statewide_win_bonus_dem",
        "mechanical_candidate_adjustment_dem",
        "candidate_scandal_adjustment_dem",
        "candidate_scandal_rationale",
        "candidate_quality_method",
        "candidate_quality_framework_version",
    ]:
        new_col = f"{col}_new"
        if new_col in updated.columns:
            updated[col] = updated[new_col].combine_first(updated[col])

    if "proposed_candidate_quality_adjustment_dem_new" in updated.columns:
        updated["candidate_quality_adjustment_dem"] = updated[
            "proposed_candidate_quality_adjustment_dem_new"
        ].combine_first(updated["candidate_quality_adjustment_dem"])

    if not args.keep_legacy_candidate_cols:
        for col in LEGACY_CANDIDATE_COLS:
            if col in updated.columns:
                updated[col] = 0.0

    drop_cols = [c for c in updated.columns if c.endswith("_new")]
    drop_cols += ["proposed_candidate_quality_adjustment_dem"]
    updated = updated.drop(columns=[c for c in drop_cols if c in updated.columns], errors="ignore")

    updated.to_csv(RACE_INPUTS, index=False)

    print(f"Applied framework candidate-quality fields to {RACE_INPUTS}")
    if not args.keep_legacy_candidate_cols:
        print(f"Zeroed legacy candidate columns where present: {LEGACY_CANDIDATE_COLS}")
    else:
        print("Kept legacy candidate columns because --keep-legacy-candidate-cols was used.")


if __name__ == "__main__":
    main()
