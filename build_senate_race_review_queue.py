from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

RACE_INPUTS = INPUTS / "race_inputs.csv"
MODEL_RESULTS = OUTPUTS / "senate_model_results.csv"
MANUAL_POLLS = INPUTS / "manual_polls.csv"
ADJUSTED_POLLS = INPUTS / "manual_polls_adjusted.csv"
BAYESIAN = INPUTS / "bayesian_update_generated.csv"

OUT = OUTPUTS / "senate_race_review_queue.csv"


def read_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def norm_state(df):
    if not df.empty and "state" in df.columns:
        df["state"] = df["state"].astype(str).str.upper().str.strip()
    return df


def safe_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def add_reason(row, condition, points, reason):
    if condition:
        row["review_score"] += points
        row["review_reasons"].append(reason)


def main():
    races = norm_state(read_csv(RACE_INPUTS))
    results = norm_state(read_csv(MODEL_RESULTS))
    manual_polls = norm_state(read_csv(MANUAL_POLLS))
    adjusted_polls = norm_state(read_csv(ADJUSTED_POLLS))
    bayes = norm_state(read_csv(BAYESIAN))

    if races.empty:
        raise FileNotFoundError(f"Missing or empty {RACE_INPUTS}")

    queue = races.copy()

    # Merge model results if available.
    if not results.empty and "state" in results.columns:
        result_cols = ["state"] + [
            c for c in results.columns
            if c in [
                "dem_win_prob",
                "gop_win_prob",
                "rep_win_prob",
                "margin_dem",
                "mean_margin_dem",
                "median_margin_dem",
                "model_margin_dem",
                "prob_dem_win",
                "prob_rep_win",
            ]
        ]
        queue = queue.merge(results[result_cols].drop_duplicates("state"), on="state", how="left")

    # Merge Bayesian diagnostics if available.
    if not bayes.empty and "state" in bayes.columns:
        bayes_cols = ["state"] + [
            c for c in bayes.columns
            if c in [
                "bayesian_polling_weight",
                "bayesian_polling_weight_capped",
                "cycle_max_polling_weight",
                "bayesian_cap_applied",
                "bayesian_model_margin_dem",
                "bayesian_model_margin_dem_capped",
                "posterior_margin_dem_capped",
            ]
        ]
        queue = queue.merge(bayes[bayes_cols].drop_duplicates("state"), on="state", how="left")

    # Poll counts.
    if not manual_polls.empty and "state" in manual_polls.columns:
        poll_counts = manual_polls.groupby("state").size().reset_index(name="manual_poll_count")
        queue = queue.merge(poll_counts, on="state", how="left")
    else:
        queue["manual_poll_count"] = 0

    queue["manual_poll_count"] = safe_num(queue["manual_poll_count"], 0)

    # Partisan/internal poll counts.
    if not manual_polls.empty and "state" in manual_polls.columns:
        mp = manual_polls.copy()

        for col in [
            "poll_sponsor_type",
            "partisan_sponsor_party",
            "pollster_partisan_affiliation",
            "is_internal_poll",
        ]:
            if col not in mp.columns:
                mp[col] = ""

        mp["is_internal_poll_bool"] = (
            mp["is_internal_poll"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

        mp["has_partisan_metadata"] = (
            mp["partisan_sponsor_party"].fillna("").astype(str).str.upper().isin(["D", "R"])
            | mp["pollster_partisan_affiliation"].fillna("").astype(str).str.upper().isin(["D", "R"])
            | mp["is_internal_poll_bool"]
            | mp["poll_sponsor_type"].fillna("").astype(str).str.lower().isin(["party", "campaign", "super pac"])
        )

        partisan_counts = (
            mp.groupby("state")["has_partisan_metadata"]
            .sum()
            .reset_index(name="partisan_or_internal_poll_count")
        )
        queue = queue.merge(partisan_counts, on="state", how="left")
    else:
        queue["partisan_or_internal_poll_count"] = 0

    queue["partisan_or_internal_poll_count"] = safe_num(queue["partisan_or_internal_poll_count"], 0)

    # Independent poll signal.
    if not manual_polls.empty and "state" in manual_polls.columns:
        mp = manual_polls.copy()
        for col in ["ind_pct", "other_pct"]:
            if col not in mp.columns:
                mp[col] = 0.0
            mp[col] = safe_num(mp[col], 0.0)

        mp["third_party_poll_share"] = mp["ind_pct"] + mp["other_pct"]

        third_party = (
            mp.groupby("state")
            .agg(
                max_third_party_poll_share=("third_party_poll_share", "max"),
                avg_third_party_poll_share=("third_party_poll_share", "mean"),
            )
            .reset_index()
        )
        queue = queue.merge(third_party, on="state", how="left")
    else:
        queue["max_third_party_poll_share"] = 0.0
        queue["avg_third_party_poll_share"] = 0.0

    queue["max_third_party_poll_share"] = safe_num(queue["max_third_party_poll_share"], 0)
    queue["avg_third_party_poll_share"] = safe_num(queue["avg_third_party_poll_share"], 0)

    # Ensure independent fields exist.
    for col, default in {
        "independent_candidate_present": False,
        "independent_candidate_name": "",
        "independent_candidate_party_lean": "",
        "independent_vote_share_estimate": 0.0,
        "independent_asymmetry_adjustment_dem": 0.0,
        "independent_adjustment_rationale": "",
    }.items():
        if col not in queue.columns:
            queue[col] = default

    queue["independent_candidate_present_bool"] = (
        queue["independent_candidate_present"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    queue["independent_vote_share_estimate"] = safe_num(queue["independent_vote_share_estimate"], 0)
    queue["independent_asymmetry_adjustment_dem"] = safe_num(queue["independent_asymmetry_adjustment_dem"], 0)

    # Try to identify competitiveness.
    prob_cols = [c for c in ["dem_win_prob", "prob_dem_win"] if c in queue.columns]
    if prob_cols:
        pcol = prob_cols[0]
        queue[pcol] = safe_num(queue[pcol], np.nan)
        queue["competitiveness"] = 1 - (queue[pcol] - 0.5).abs() * 2
    else:
        margin_cols = [
            c for c in [
                "margin_dem",
                "mean_margin_dem",
                "median_margin_dem",
                "model_margin_dem",
                "bayesian_model_margin_dem_capped",
                "posterior_margin_dem_capped",
            ]
            if c in queue.columns
        ]

        if margin_cols:
            mcol = margin_cols[0]
            queue[mcol] = safe_num(queue[mcol], 0)
            queue["competitiveness"] = (1 - (queue[mcol].abs() / 20)).clip(lower=0, upper=1)
        else:
            queue["competitiveness"] = 0

    # Rationale fields.
    rationale_cols = [
        "incumbency_rationale",
        "candidate_quality_rationale",
        "overperformance_rationale",
        "liability_rationale",
        "special_adjustment_rationale",
        "independent_adjustment_rationale",
    ]

    for col in rationale_cols:
        if col not in queue.columns:
            queue[col] = ""

    # Score each race.
    review_rows = []

    for _, source_row in queue.iterrows():
        row = source_row.copy()
        row["review_score"] = 0
        row["review_reasons"] = []

        competitiveness = float(row.get("competitiveness", 0) or 0)
        poll_count = float(row.get("manual_poll_count", 0) or 0)
        partisan_poll_count = float(row.get("partisan_or_internal_poll_count", 0) or 0)

        independent_present = bool(row.get("independent_candidate_present_bool", False))
        independent_adjustment = float(row.get("independent_asymmetry_adjustment_dem", 0) or 0)
        independent_vote_share = float(row.get("independent_vote_share_estimate", 0) or 0)
        max_third_party_poll_share = float(row.get("max_third_party_poll_share", 0) or 0)

        add_reason(
            row,
            competitiveness >= 0.75,
            4,
            "Highly competitive race",
        )

        add_reason(
            row,
            competitiveness >= 0.50 and poll_count == 0,
            3,
            "Competitive race with no manual polls",
        )

        add_reason(
            row,
            poll_count > 0,
            1,
            "Manual polling is present",
        )

        add_reason(
            row,
            partisan_poll_count > 0,
            3,
            "Partisan/internal poll metadata present",
        )

        add_reason(
            row,
            independent_present,
            4,
            "Independent/third-party candidate flagged",
        )

        add_reason(
            row,
            independent_present and independent_adjustment == 0 and (
                independent_vote_share > 0 or max_third_party_poll_share >= 3
            ),
            3,
            "Independent candidate present but asymmetry adjustment remains zero",
        )

        add_reason(
            row,
            max_third_party_poll_share >= 5,
            3,
            "Polls show notable independent/third-party vote share",
        )

        add_reason(
            row,
            abs(independent_adjustment) >= 1.5,
            2,
            "Large independent asymmetry adjustment",
        )

        # Missing rationales for adjusted or special-case races.
        special_adjustment_cols = [
            c for c in queue.columns
            if "adjustment" in c.lower() and c not in [
                "independent_asymmetry_adjustment_dem",
            ]
        ]

        has_large_adjustment = False
        for col in special_adjustment_cols:
            try:
                if abs(float(row.get(col, 0) or 0)) >= 1.5:
                    has_large_adjustment = True
                    break
            except Exception:
                pass

        missing_any_rationale = any(
            str(row.get(col, "")).strip() in ["", "nan", "None"]
            for col in rationale_cols
        )

        add_reason(
            row,
            has_large_adjustment and missing_any_rationale,
            2,
            "Large adjustment with incomplete rationale fields",
        )

        add_reason(
            row,
            str(row.get("human_review_status", "")).strip().lower() in ["needs review", "review", "pending"],
            3,
            "Human review status indicates review needed",
        )

        if row["review_score"] > 0:
            row["review_reasons"] = "; ".join(row["review_reasons"])
            review_rows.append(row)

    if review_rows:
        review_df = pd.DataFrame(review_rows)
    else:
        review_df = queue.head(0).copy()
        review_df["review_score"] = []
        review_df["review_reasons"] = []

    output_cols = [
        "state",
        "race",
        "review_score",
        "review_reasons",
        "competitiveness",
        "manual_poll_count",
        "partisan_or_internal_poll_count",
        "independent_candidate_present",
        "independent_candidate_name",
        "independent_candidate_party_lean",
        "independent_vote_share_estimate",
        "independent_asymmetry_adjustment_dem",
        "max_third_party_poll_share",
        "avg_third_party_poll_share",
        "human_review_status",
        "last_human_review_date",
    ]

    output_cols = [c for c in output_cols if c in review_df.columns]

    review_df = review_df.sort_values(
        ["review_score", "competitiveness"],
        ascending=[False, False],
    )

    review_df[output_cols].to_csv(OUT, index=False)

    print(f"Built Senate race review queue: {OUT}")
    print(f"Rows: {len(review_df)}")
    if len(review_df):
        print(review_df[output_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
