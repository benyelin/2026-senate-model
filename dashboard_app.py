from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
import subprocess

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
INPUTS = ROOT / "inputs"

st.set_page_config(page_title="2026 Senate Forecast", page_icon="🗳️", layout="wide")


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pct(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "—"


def num(x, digits=1):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"


summary = read_csv_safe(OUTPUTS / "forecast_summary.csv")
race_stats = read_csv_safe(OUTPUTS / "race_stats.csv")
seat_dist = read_csv_safe(OUTPUTS / "seat_distribution.csv")
scenarios = read_csv_safe(OUTPUTS / "scenario_summary.csv")
bayes = read_csv_safe(INPUTS / "bayesian_update_generated.csv")
poll_avgs = read_csv_safe(INPUTS / "polling_averages_generated.csv")

st.title("2026 Senate Forecast Dashboard")
st.caption("Polling ingestion → Bayesian update → correlated simulation → scenario diagnostics.")

if summary.empty:
    st.error("No forecast outputs found. Run `python run_full_pipeline.py` first.")
    st.stop()

s = summary.iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Dem Control Odds", pct(s.get("dem_control_probability")))
c2.metric("Expected Dem Seats", num(s.get("expected_dem_seats"), 2))
c3.metric("Median Dem Seats", num(s.get("median_dem_seats"), 0))
c4.metric("Polling Weight", pct(s.get("polling_weight")))
c5.metric("Implied Correlation", pct(s.get("implied_correlation")))

c6, c7, c8, c9 = st.columns(4)
c6.metric("Days Out", num(s.get("days_out"), 0))
c7.metric("Total Error SD", num(s.get("total_error_sd"), 2))
c8.metric("National Error SD", num(s.get("national_error_sd"), 2))
c9.metric("Race Error SD", num(s.get("race_error_sd"), 2))

st.divider()

tab_overview, tab_races, tab_scenarios, tab_polls, tab_method = st.tabs(
    ["Overview", "Race Table", "Scenarios", "Polling & Bayes", "Method"]
)

with tab_overview:
    left, right = st.columns([1.25, 1])

    with left:
        st.subheader("Seat Distribution")
        if seat_dist.empty:
            st.info("No seat distribution file found.")
        else:
            fig = px.bar(
                seat_dist,
                x="dem_seats",
                y="probability",
                labels={"dem_seats": "Democratic Seats", "probability": "Probability"},
                text=seat_dist["probability"].map(lambda v: f"{v:.1%}"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Model Settings")
        st.write(f"Control threshold: **{int(s.get('dem_control_threshold', 51))} seats**")
        st.write(f"Baseline Democratic seats: **{int(s.get('dem_baseline_seats', 47))}**")
        st.write(f"National environment margin: **{num(s.get('national_environment_margin'), 2)}**")

        if not race_stats.empty and "tipping_share_of_control_sims" in race_stats.columns:
            tipping = race_stats.sort_values("tipping_share_of_control_sims", ascending=False).head(8)
            show = tipping[["state", "tipping_share_of_control_sims", "simulated_dem_win_prob", "model_margin_dem"]].copy()
            show["tipping_share_of_control_sims"] = show["tipping_share_of_control_sims"].map(lambda v: f"{v:.1%}")
            show["simulated_dem_win_prob"] = show["simulated_dem_win_prob"].map(lambda v: f"{v:.1%}")
            show["model_margin_dem"] = show["model_margin_dem"].map(lambda v: f"{v:.1f}")
            st.subheader("Most Common Tipping Races")
            st.dataframe(show.rename(columns={
                "state": "State",
                "tipping_share_of_control_sims": "Tipping Share",
                "simulated_dem_win_prob": "Dem Win Prob",
                "model_margin_dem": "Model Margin",
            }), use_container_width=True, hide_index=True)

with tab_races:
    st.subheader("Race-Level Forecast")
    if race_stats.empty:
        st.info("No race stats found.")
    else:
        display = race_stats.copy()
        display["Dem Win Prob"] = display["simulated_dem_win_prob"].map(lambda v: f"{v:.1%}")
        display["Pre-Sim Prob"] = display["pre_sim_dem_win_prob"].map(lambda v: f"{v:.1%}")
        display["Model Margin"] = display["model_margin_dem"].map(lambda v: f"{v:.1f}")
        display["Avg Sim Margin"] = display["avg_simulated_margin_dem"].map(lambda v: f"{v:.1f}")
        cols = [
            "state", "race_tier", "dem_candidate", "gop_candidate", "current_holder",
            "Model Margin", "Dem Win Prob", "Avg Sim Margin",
            "elasticity", "dem_win_counts_for_seat_change"
        ]
        st.dataframe(display[cols].rename(columns={
            "state": "State",
            "race_tier": "Tier",
            "dem_candidate": "Dem Candidate",
            "gop_candidate": "GOP Candidate",
            "current_holder": "Current Holder",
            "elasticity": "Elasticity",
            "dem_win_counts_for_seat_change": "Seat Gain If Dem Wins",
        }), use_container_width=True, hide_index=True)

        fig = px.bar(
            race_stats.sort_values("simulated_dem_win_prob"),
            x="simulated_dem_win_prob",
            y="state",
            orientation="h",
            labels={"simulated_dem_win_prob": "Democratic Win Probability", "state": "State"},
        )
        fig.update_layout(xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

with tab_scenarios:
    st.subheader("National Swing Sensitivity")
    if scenarios.empty:
        st.info("No scenario outputs found. Run `python scenario_runner.py`.")
    else:
        fig = px.line(
            scenarios,
            x="national_environment_margin_dem",
            y="current_dem_control_probability",
            markers=True,
            labels={
                "national_environment_margin_dem": "National Swing Toward Democrats",
                "current_dem_control_probability": "Democratic Control Probability",
            },
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(scenarios, use_container_width=True, hide_index=True)

with tab_polls:
    st.subheader("Weighted Polling Averages")
    if poll_avgs.empty:
        st.info("No generated polling averages found.")
    else:
        st.dataframe(poll_avgs, use_container_width=True, hide_index=True)

    st.subheader("Bayesian Update Audit")
    if bayes.empty:
        st.info("No Bayesian update file found.")
    else:
        st.dataframe(bayes, use_container_width=True, hide_index=True)

with tab_method:
    st.subheader("Methodology")
    st.markdown("""
    **Pipeline**

    1. Enter/export polls into `inputs/polls_raw.csv`.
    2. `ingest_polls.py` creates weighted polling averages.
    3. `bayesian_update.py` combines fundamentals and polling.
    4. `run_model.py` simulates race margins using shared national and race-specific errors.
    5. `scenario_runner.py` tests national swing sensitivity.
    6. This dashboard reads the output CSVs.

    **Key modeling choices**

    - Outcomes are simulated from margins.
    - Shared national error creates correlated outcomes.
    - Race-specific error captures local uncertainty.
    - Time-to-election calibration controls uncertainty and polling weight.
    - Candidate metadata is separate from numerical assumptions.
    """)

    st.code("""
python run_full_pipeline.py --as-of 2026-05-22 --days-out 165 --sims 50000
streamlit run dashboard_app.py
    """.strip(), language="bash")


# Manual poll entry form
st.subheader("Add Manual Poll")

manual_raw_path = "inputs/manual_polls.csv"

with st.form("manual_poll_entry_form"):
    race = st.text_input("Race", value="ME Senate")
    state = st.text_input("State abbreviation", value="ME")
    chamber = st.selectbox("Chamber", ["Senate", "House", "Governor"], index=0)
    pollster = st.text_input("Pollster")
    
    pollster_grade = st.selectbox(
        "Pollster grade",
        ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "Unknown"],
        index=10
    )
    
    house_effect_dem = st.number_input(
    "House effect, Dem points",
    min_value=-10.0,
    max_value=10.0,
    value=0.0,
    step=0.1,
    help="Placeholder for now. Positive means the pollster tends to favor Democrats; negative means it tends to favor Republicans."
)


    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")
    
    sample_size = st.number_input("Sample size", min_value=1, value=800, step=1)
    sample_type = st.selectbox("Sample type", ["LV", "RV", "A", "Other"], index=0)

    dem_candidate = st.text_input("Democratic candidate")
    rep_candidate = st.text_input("Republican candidate")
    ind_candidate = st.text_input("Independent candidate")
    other_candidate = st.text_input("Other candidate")

    dem_pct = st.number_input("Dem %", min_value=0.0, max_value=100.0, value=45.0, step=0.1)
    rep_pct = st.number_input("Rep %", min_value=0.0, max_value=100.0, value=45.0, step=0.1)
    ind_pct = st.number_input("Independent %", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    other_pct = st.number_input("Other %", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    undecided_pct = st.number_input("Undecided %", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    notes = st.text_area("Notes")

    submitted = st.form_submit_button("Add poll")

    if submitted:

        new_poll = pd.DataFrame([{
            "race": race,
            "state": state,
            "chamber": chamber,
            "pollster": pollster,
            "pollster_grade": pollster_grade,
            "house_effect_dem": house_effect_dem,
            "start_date": start_date,
            "end_date": end_date,
            "sample_size": sample_size,
            "sample_type": sample_type,
            "dem_candidate": dem_candidate,
            "rep_candidate": rep_candidate,
            "ind_candidate": ind_candidate,
            "other_candidate": other_candidate,
            "dem_pct": dem_pct,
            "rep_pct": rep_pct,
            "ind_pct": ind_pct,
            "other_pct": other_pct,
            "undecided_pct": undecided_pct,
            "notes": notes
        }])

        if Path(manual_raw_path).exists():
            existing_polls = pd.read_csv(manual_raw_path)
            updated_polls = pd.concat([existing_polls, new_poll], ignore_index=True)
        else:
            updated_polls = new_poll

        updated_polls.to_csv(manual_raw_path, index=False)

        subprocess.run(["python3", "validate_manual_polls.py"])

        subprocess.run(["python3", "run_full_pipeline.py"])

        st.success("Poll added, validated, and full model pipeline refreshed successfully.")

        if Path(manual_raw_path).exists():
            existing_polls = pd.read_csv(manual_raw_path)
            updated_polls = pd.concat([existing_polls, new_poll], ignore_index=True)
        else:
            updated_polls = new_poll

        updated_polls.to_csv(manual_raw_path, index=False)

        subprocess.run(["python3", "validate_manual_polls.py"])

        subprocess.run(["python3", "run_full_pipeline.py"])

        st.success("Poll added, validated, and full model pipeline refreshed successfully.")
# Editable manual poll management table
st.subheader("Manage Manual Polls")

manual_raw_path = "inputs/manual_polls.csv"

if Path(manual_raw_path).exists():
    manual_polls_editor = pd.read_csv(manual_raw_path)

    if not manual_polls_editor.empty:
        st.caption("Edit cells directly. To delete a poll, select the row using the left checkbox column and click Save Changes.")

        edited_manual_polls = st.data_editor(
            manual_polls_editor,
            use_container_width=True,
            num_rows="dynamic",
            key="manual_polls_editor"
        )

        if st.button("Save Manual Poll Changes"):
            edited_manual_polls.to_csv(manual_raw_path, index=False)

            subprocess.run(["python3", "validate_manual_polls.py"])
            subprocess.run(["python3", "run_full_pipeline.py"])

            st.success("Manual poll changes saved, validated, and model refreshed.")
    else:
        st.info("No manual polls available to edit yet.")
else:
    st.info("No manual_polls.csv file found yet.")

# Recent manual polls table
st.subheader("Recent Manual Polls")

manual_polls_path = "outputs/manual_polls_clean.csv"

if Path(manual_polls_path).exists():
    manual_polls_display = pd.read_csv(manual_polls_path)

    if not manual_polls_display.empty:
        manual_polls_display["end_date"] = pd.to_datetime(
            manual_polls_display["end_date"],
            errors="coerce"
        )

        recent_manual_polls = (
            manual_polls_display
            .sort_values("end_date", ascending=False)
            .head(30)
        )

        display_cols = [
            "race",
            "pollster",
            "end_date",
            "sample_size",
            "sample_type",
            "dem_pct",
            "rep_pct",
            "ind_pct",
            "leader",
            "leader_margin",
            "poll_weight",
            "is_three_way_race",
            "notes"
        ]

        display_cols = [
            col for col in display_cols
            if col in recent_manual_polls.columns
        ]

        st.dataframe(
            recent_manual_polls[display_cols],
            use_container_width=True
        )
    else:
        st.info("No manual polls found yet.")
else:
    st.info("No manual poll file found yet. Run validate_manual_polls.py first.")
    # Model Inputs Audit Panel
st.subheader("Model Inputs Audit")

STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
}


def normalize_state_for_audit(x):
    x = str(x).strip().upper()
    return STATE_NAME_TO_ABBR.get(x, x)


race_inputs_path = INPUTS / "race_inputs.csv"
polling_avgs_path = INPUTS / "polling_averages_generated.csv"
bayesian_path = INPUTS / "bayesian_update_generated.csv"

if race_inputs_path.exists():
    race_inputs_audit = pd.read_csv(race_inputs_path)
else:
    race_inputs_audit = pd.DataFrame()

if polling_avgs_path.exists():
    polling_audit = pd.read_csv(polling_avgs_path)
else:
    polling_audit = pd.DataFrame()

if bayesian_path.exists():
    bayesian_audit = pd.read_csv(bayesian_path)
else:
    bayesian_audit = pd.DataFrame()

if race_inputs_audit.empty:
    st.info("No race_inputs.csv file found for audit.")
else:
    audit_df = race_inputs_audit.copy()

    # Normalize state identifiers across all audit inputs
    if "state" in audit_df.columns:
        audit_df["state"] = audit_df["state"].apply(normalize_state_for_audit)

    if not polling_audit.empty and "state" in polling_audit.columns:
        polling_audit["state"] = polling_audit["state"].apply(normalize_state_for_audit)

    if not bayesian_audit.empty and "state" in bayesian_audit.columns:
        bayesian_audit["state"] = bayesian_audit["state"].apply(normalize_state_for_audit)

    # Merge manual-only polling averages
    if not polling_audit.empty and "state" in polling_audit.columns:
        poll_cols = [
            col for col in [
                "state",
                "polling_margin_dem",
                "poll_count",
                "latest_poll_end_date",
                "avg_poll_age_days",
                "total_poll_weight",
            ]
            if col in polling_audit.columns
        ]

        polling_for_merge = polling_audit[poll_cols].copy()

        polling_for_merge = polling_for_merge.rename(
            columns={
                "polling_margin_dem": "manual_polling_margin_dem",
                "poll_count": "manual_poll_count",
                "latest_poll_end_date": "manual_latest_poll_end_date",
                "avg_poll_age_days": "manual_avg_poll_age_days",
                "total_poll_weight": "manual_total_poll_weight",
            }
        )

        audit_df = audit_df.merge(
            polling_for_merge,
            on="state",
            how="left"
        )

    # Merge Bayesian/final blended model values
    if not bayesian_audit.empty and "state" in bayesian_audit.columns:
        bayes_cols = [
            col for col in [
                "state",
                "fundamentals_margin_dem",
                "polling_margin_dem",
                "posterior_margin_dem",
                "posterior_sd",
                "poll_weight",
                "fundamentals_weight",
                "polling_weight",
                "final_margin_dem",
                "model_margin_dem",
            ]
            if col in bayesian_audit.columns
        ]

        bayes_for_merge = bayesian_audit[bayes_cols].copy()

        if "polling_margin_dem" in bayes_for_merge.columns:
            bayes_for_merge = bayes_for_merge.rename(
                columns={"polling_margin_dem": "bayesian_polling_margin_dem"}
            )

        audit_df = audit_df.merge(
            bayes_for_merge,
            on="state",
            how="left"
        )

    # Polling status
    if "manual_poll_count" in audit_df.columns:
        audit_df["manual_poll_count"] = pd.to_numeric(
            audit_df["manual_poll_count"],
            errors="coerce"
        ).fillna(0)

        audit_df["polling_status"] = audit_df["manual_poll_count"].apply(
            lambda x: "Manual polling used" if x > 0 else "Fundamentals only"
        )
    else:
        audit_df["manual_poll_count"] = 0
        audit_df["polling_status"] = "Fundamentals only"

    # Polling effect diagnostic
    if "manual_polling_margin_dem" in audit_df.columns:
        audit_df["manual_polling_margin_dem"] = pd.to_numeric(
            audit_df["manual_polling_margin_dem"],
            errors="coerce"
        )

    if "fundamentals_margin_dem" in audit_df.columns:
        audit_df["fundamentals_margin_dem"] = pd.to_numeric(
            audit_df["fundamentals_margin_dem"],
            errors="coerce"
        )

    if (
        "manual_polling_margin_dem" in audit_df.columns
        and "fundamentals_margin_dem" in audit_df.columns
    ):
        audit_df["poll_vs_fundamentals_gap"] = (
            audit_df["manual_polling_margin_dem"]
            - audit_df["fundamentals_margin_dem"]
        )

    if (
        "posterior_margin_dem" in audit_df.columns
        and "fundamentals_margin_dem" in audit_df.columns
    ):
        audit_df["model_movement_from_fundamentals"] = (
            audit_df["posterior_margin_dem"]
            - audit_df["fundamentals_margin_dem"]
        )

    # RCV/election-system defaults
    if "election_system" not in audit_df.columns:
        audit_df["election_system"] = "plurality"

    if "rcv_enabled" not in audit_df.columns:
        audit_df["rcv_enabled"] = False

    audit_df["rcv_enabled"] = audit_df["rcv_enabled"].astype(str)

    # Choose display columns that exist
    preferred_cols = [
        "state",
        "race",
        "dem_candidate",
        "rep_candidate",
        "ind_candidate",
        "incumbent_party",
        "incumbent_running",
        "open_seat",
        "election_system",
        "rcv_enabled",
        "polling_status",
        "fundamentals_margin_dem",
        "manual_polling_margin_dem",
        "bayesian_polling_margin_dem",
        "posterior_margin_dem",
        "final_margin_dem",
        "model_margin_dem",
        "posterior_sd",
        "poll_vs_fundamentals_gap",
        "model_movement_from_fundamentals",
        "manual_poll_count",
        "manual_latest_poll_end_date",
        "manual_avg_poll_age_days",
        "manual_total_poll_weight",
        "poll_weight",
        "polling_weight",
        "fundamentals_weight",
    ]

    display_cols = [
        col for col in preferred_cols
        if col in audit_df.columns
    ]

    st.dataframe(
        audit_df[display_cols],
        use_container_width=True
    )

    # Summary check
    manual_states = audit_df.loc[
        audit_df["polling_status"] == "Manual polling used",
        "state"
    ].tolist()

    if manual_states:
        st.caption(
            "Manual polling currently detected for: "
            + ", ".join(sorted(manual_states))
        )
    else:
        st.caption("No manual polling detected in the audit table.") 
