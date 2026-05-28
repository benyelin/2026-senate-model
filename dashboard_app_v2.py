from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

st.set_page_config(
    page_title="2026 Senate Forecast Dashboard V2",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
def read_csv_safe(path):
    try:
        if Path(path).exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
    return pd.DataFrame()


def as_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fmt_margin(x):
    x = as_float(x)
    if pd.isna(x):
        return "—"
    if x > 0:
        return f"D+{x:.1f}"
    if x < 0:
        return f"R+{abs(x):.1f}"
    return "Even"


def fmt_pct(x):
    x = as_float(x)
    if pd.isna(x):
        return "—"
    return f"{x:.1%}"


def fmt_num(x, digits=2):
    x = as_float(x)
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def normalize_state(df):
    if not df.empty and "state" in df.columns:
        df = df.copy()
        df["state"] = df["state"].astype(str).str.strip().str.upper()
    return df


def race_rating_from_prob(p):
    p = as_float(p)
    if pd.isna(p):
        return "Unknown"

    if p >= 0.95:
        return "Safe D"
    if p >= 0.85:
        return "Likely D"
    if p >= 0.65:
        return "Lean D"
    if p >= 0.55:
        return "Tilt D"
    if p > 0.45:
        return "Toss-up"
    if p > 0.35:
        return "Tilt R"
    if p > 0.15:
        return "Lean R"
    if p > 0.05:
        return "Likely R"
    return "Safe R"


def load_data():
    data = {
        "summary": read_csv_safe(OUTPUTS / "forecast_summary.csv"),
        "race_stats": normalize_state(read_csv_safe(OUTPUTS / "race_stats.csv")),
        "seat_distribution": read_csv_safe(OUTPUTS / "seat_distribution.csv"),
        "scenarios": read_csv_safe(OUTPUTS / "scenario_summary.csv"),
        "race_inputs": normalize_state(read_csv_safe(INPUTS / "race_inputs.csv")),
        "polling": normalize_state(read_csv_safe(INPUTS / "polling_averages_generated.csv")),
        "bayes": normalize_state(read_csv_safe(INPUTS / "bayesian_update_generated.csv")),
        "national_env": read_csv_safe(INPUTS / "national_environment.csv"),
    }
    return data


data = load_data()

summary = data["summary"]
race_stats = data["race_stats"]
seat_distribution = data["seat_distribution"]
scenarios = data["scenarios"]
race_inputs = data["race_inputs"]
polling = data["polling"]
bayes = data["bayes"]
national_env = data["national_env"]

# -----------------------------
# Header
# -----------------------------
st.title("2026 Senate Forecast Dashboard")
st.caption("Clean view: forecast overview, race ratings, model drivers, scenarios, and diagnostics.")

if summary.empty:
    st.error("No forecast summary found. Run `python3 run_full_pipeline.py` first.")
    st.stop()

summary_row = summary.iloc[-1].to_dict()

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_races, tab_drivers, tab_scenarios, tab_diagnostics = st.tabs(
    [
        "Overview",
        "Race Ratings",
        "Model Drivers",
        "Scenarios",
        "Diagnostics",
    ]
)

# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    st.subheader("Topline Forecast")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Dem Control Odds", fmt_pct(summary_row.get("dem_control_probability")))
    c2.metric("Expected Dem Seats", fmt_num(summary_row.get("expected_dem_seats"), 2))
    c3.metric("Median Dem Seats", fmt_num(summary_row.get("median_dem_seats"), 0))
    c4.metric("National Environment", fmt_margin(summary_row.get("national_environment_margin")))
    c5.metric("Days Out", fmt_num(summary_row.get("days_out"), 0))

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Seat Distribution")

        if seat_distribution.empty:
            st.info("No seat distribution file found.")
        else:
            sd = seat_distribution.copy()

            # Try common column names
            x_col = None
            y_col = None

            for col in ["dem_seats", "seats", "Democratic seats"]:
                if col in sd.columns:
                    x_col = col
                    break

            for col in ["probability", "share", "frequency"]:
                if col in sd.columns:
                    y_col = col
                    break

            if x_col and y_col:
                fig = px.bar(
                    sd,
                    x=x_col,
                    y=y_col,
                    labels={
                        x_col: "Democratic seats",
                        y_col: "Probability",
                    },
                    title="Simulated Democratic Seat Distribution",
                )
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(sd, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Most Important Races")

        if race_stats.empty:
            st.info("No race stats found.")
        else:
            rs = race_stats.copy()

            if "tipping_share_of_control_sims" in rs.columns:
                rs["tipping_share_of_control_sims"] = pd.to_numeric(
                    rs["tipping_share_of_control_sims"],
                    errors="coerce"
                )
                top = rs.sort_values("tipping_share_of_control_sims", ascending=False).head(10)
            elif "simulated_dem_win_prob" in rs.columns:
                rs["distance_to_50"] = (
                    pd.to_numeric(rs["simulated_dem_win_prob"], errors="coerce") - 0.5
                ).abs()
                top = rs.sort_values("distance_to_50").head(10)
            else:
                top = rs.head(10)

            rows = []
            for _, row in top.iterrows():
                rows.append(
                    {
                        "State": row.get("state", ""),
                        "Race": f"{row.get('dem_candidate', '')} vs. {row.get('gop_candidate', '')}",
                        "Dem odds": fmt_pct(row.get("simulated_dem_win_prob")),
                        "Model margin": fmt_margin(row.get("model_margin_dem")),
                        "Tipping share": fmt_pct(row.get("tipping_share_of_control_sims")),
                    }
                )

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Competitive Race Snapshot")

    if race_stats.empty:
        st.info("No race stats found.")
    else:
        comp = race_stats.copy()
        comp["simulated_dem_win_prob"] = pd.to_numeric(
            comp.get("simulated_dem_win_prob"),
            errors="coerce"
        )
        comp["competitiveness"] = (comp["simulated_dem_win_prob"] - 0.5).abs()
        comp = comp.sort_values("competitiveness").head(12)

        display = []
        for _, row in comp.iterrows():
            display.append(
                {
                    "State": row.get("state", ""),
                    "Rating": race_rating_from_prob(row.get("simulated_dem_win_prob")),
                    "Dem candidate": row.get("dem_candidate", ""),
                    "GOP candidate": row.get("gop_candidate", ""),
                    "Dem odds": fmt_pct(row.get("simulated_dem_win_prob")),
                    "Model margin": fmt_margin(row.get("model_margin_dem")),
                    "Avg sim margin": fmt_margin(row.get("avg_simulated_margin_dem")),
                }
            )

        st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)

# -----------------------------
# Race Ratings
# -----------------------------
with tab_races:
    st.subheader("Race Ratings")

    if race_stats.empty:
        st.info("No race stats found.")
    else:
        rs = race_stats.copy()

        rs["simulated_dem_win_prob_num"] = pd.to_numeric(
            rs.get("simulated_dem_win_prob"),
            errors="coerce"
        )
        rs["model_margin_dem_num"] = pd.to_numeric(
            rs.get("model_margin_dem"),
            errors="coerce"
        )
        rs["rating"] = rs["simulated_dem_win_prob_num"].apply(race_rating_from_prob)
        rs["competitiveness"] = (rs["simulated_dem_win_prob_num"] - 0.5).abs()

        view_mode = st.radio(
            "Sort races by",
            ["Competitiveness", "Dem win probability", "State"],
            horizontal=True,
        )

        if view_mode == "Competitiveness":
            rs = rs.sort_values("competitiveness")
        elif view_mode == "Dem win probability":
            rs = rs.sort_values("simulated_dem_win_prob_num", ascending=False)
        else:
            rs = rs.sort_values("state")

        chart_df = rs.copy()
        chart_df["Dem win probability"] = chart_df["simulated_dem_win_prob_num"]
        chart_df["State"] = chart_df["state"]

        fig = px.bar(
            chart_df,
            x="Dem win probability",
            y="State",
            orientation="h",
            hover_data=[
                "dem_candidate",
                "gop_candidate",
                "rating",
                "model_margin_dem_num",
            ],
            title="Democratic Win Probability by Race",
        )
        fig.update_layout(xaxis_tickformat=".0%", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        table_rows = []
        for _, row in rs.iterrows():
            table_rows.append(
                {
                    "State": row.get("state", ""),
                    "Rating": row.get("rating", ""),
                    "Dem candidate": row.get("dem_candidate", ""),
                    "GOP candidate": row.get("gop_candidate", ""),
                    "Holder": row.get("current_holder", ""),
                    "Dem odds": fmt_pct(row.get("simulated_dem_win_prob")),
                    "Model margin": fmt_margin(row.get("model_margin_dem")),
                    "Fundamentals": fmt_margin(row.get("fundamentals_margin_dem")),
                    "Polling": fmt_margin(row.get("polling_margin_dem")),
                    "Tipping share": fmt_pct(row.get("tipping_share_of_control_sims")),
                }
            )

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# -----------------------------
# Model Drivers
# -----------------------------
with tab_drivers:
    st.subheader("National Environment")

    if national_env.empty:
        st.info("No national environment file found.")
    else:
        env = national_env.iloc[-1]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Generic Ballot", fmt_margin(env.get("generic_ballot_margin_dem")))
        c2.metric("Pres. Approval", f"{fmt_num(env.get('presidential_approval'), 1)}%")
        c3.metric("Pres. Disapproval", f"{fmt_num(env.get('presidential_disapproval'), 1)}%")
        c4.metric("Net Approval", fmt_num(env.get("presidential_net_approval"), 1))
        c5.metric("National Environment", fmt_margin(env.get("national_environment_margin_dem")))

        st.caption(
            "Current formula: 0.85 × generic ballot + 0.50 × approval adjustment + 0.50 × midterm adjustment."
        )

        env_display_cols = [
            "as_of_date",
            "generic_ballot_margin_dem",
            "presidential_approval",
            "presidential_disapproval",
            "presidential_net_approval",
            "approval_adjustment_dem",
            "midterm_adjustment_dem",
            "national_environment_margin_dem",
            "source_notes",
        ]
        env_display_cols = [c for c in env_display_cols if c in national_env.columns]

        with st.expander("Raw national environment inputs", expanded=False):
            st.dataframe(national_env[env_display_cols].tail(1), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Fundamentals and Polling Audit")

    if race_inputs.empty:
        st.info("No race inputs found.")
    else:
        audit = race_inputs.copy()

        if not race_stats.empty:
            stats_cols = [
                c for c in [
                    "state",
                    "model_margin_dem",
                    "simulated_dem_win_prob",
                    "avg_simulated_margin_dem",
                    "pre_sim_dem_win_prob",
                ]
                if c in race_stats.columns
            ]
            audit = audit.merge(race_stats[stats_cols], on="state", how="left")

        if not polling.empty:
            polling_cols = [
                c for c in [
                    "state",
                    "polling_margin_dem",
                    "poll_count",
                    "latest_poll_end_date",
                    "avg_poll_age_days",
                    "total_poll_weight",
                ]
                if c in polling.columns
            ]
            poll_view = polling[polling_cols].rename(
                columns={
                    "polling_margin_dem": "manual_polling_margin_dem",
                    "poll_count": "manual_poll_count",
                }
            )
            audit = audit.merge(poll_view, on="state", how="left")

        key_default = ["AK", "FL", "GA", "ME", "NC", "OH", "TX"]
        all_states = sorted(audit["state"].dropna().unique().tolist())
        selected = st.multiselect(
            "States to show",
            all_states,
            default=[s for s in key_default if s in all_states],
        )

        audit = audit[audit["state"].isin(selected)].copy()

        rows = []
        for _, row in audit.iterrows():
            rows.append(
                {
                    "State": row.get("state", ""),
                    "Baseline source": row.get("baseline_source", ""),
                    "Pres. baseline": fmt_margin(row.get("state_partisan_baseline_dem")),
                    "Nat'l env. effect": fmt_margin(row.get("state_environment_adjustment_dem")),
                    "Incumbency": fmt_margin(row.get("incumbency_adjustment_dem")),
                    "Cand. quality": fmt_margin(row.get("candidate_quality_adjustment_dem")),
                    "Special": fmt_margin(row.get("special_adjustment_dem")),
                    "Fundamentals": fmt_margin(row.get("fundamentals_margin_dem")),
                    "Manual polling": fmt_margin(row.get("manual_polling_margin_dem")),
                    "Poll count": fmt_num(row.get("manual_poll_count"), 0),
                    "Bayes margin": fmt_margin(row.get("bayesian_model_margin_dem")),
                    "Poll weight": fmt_pct(row.get("bayesian_polling_weight")),
                    "Posterior SD": fmt_num(row.get("bayesian_posterior_sd"), 2),
                    "Final margin": fmt_margin(row.get("model_margin_dem")),
                    "Dem odds": fmt_pct(row.get("simulated_dem_win_prob")),
                    "Notes": row.get("fundamentals_notes", ""),
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("Full race input table", expanded=False):
            st.dataframe(race_inputs, use_container_width=True, hide_index=True)

# -----------------------------
# Scenarios
# -----------------------------
with tab_scenarios:
    st.subheader("National Environment Scenario Sensitivity")

    if scenarios.empty:
        st.info("No scenario summary found. Run `python3 scenario_runner.py`.")
    else:
        sc = scenarios.copy()
        sc["national_environment_margin_dem_num"] = pd.to_numeric(
            sc.get("national_environment_margin_dem"),
            errors="coerce"
        )
        sc["dem_control_probability_num"] = pd.to_numeric(
            sc.get("dem_control_probability"),
            errors="coerce"
        )
        sc["expected_dem_seats_num"] = pd.to_numeric(
            sc.get("expected_dem_seats"),
            errors="coerce"
        )
        sc = sc.sort_values("national_environment_margin_dem_num")

        fig = px.line(
            sc,
            x="national_environment_margin_dem_num",
            y="dem_control_probability_num",
            markers=True,
            hover_data=["scenario", "expected_dem_seats_num"],
            labels={
                "national_environment_margin_dem_num": "National environment margin",
                "dem_control_probability_num": "Democratic control probability",
            },
            title="Control Probability by National Environment",
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        scen_rows = []
        for _, row in sc.iterrows():
            scen_rows.append(
                {
                    "Scenario": row.get("scenario", ""),
                    "National env.": fmt_margin(row.get("national_environment_margin_dem")),
                    "Shift from base": fmt_margin(row.get("environment_shift_from_base")),
                    "Dem control odds": fmt_pct(row.get("dem_control_probability")),
                    "Expected Dem seats": fmt_num(row.get("expected_dem_seats"), 2),
                    "Median Dem seats": fmt_num(row.get("median_dem_seats"), 0),
                }
            )
        st.dataframe(pd.DataFrame(scen_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("National Environment Formula Tests")

    formula_summary = read_csv_safe(OUTPUTS / "national_environment_formula_readable_summary.csv")

    if formula_summary.empty:
        st.info("No readable formula test found. Run `python3 summarize_formula_test.py`.")
    else:
        st.dataframe(formula_summary, use_container_width=True, hide_index=True)

# -----------------------------
# Diagnostics
# -----------------------------
with tab_diagnostics:
    st.subheader("Model Diagnostics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Error SD", fmt_num(summary_row.get("total_error_sd"), 2))
    c2.metric("National Error SD", fmt_num(summary_row.get("national_error_sd"), 2))
    c3.metric("Race Error SD", fmt_num(summary_row.get("race_error_sd"), 2))
    c4.metric("Implied Correlation", fmt_pct(summary_row.get("implied_correlation")))

    st.divider()

    st.subheader("File Status")

    files = [
        INPUTS / "race_inputs.csv",
        INPUTS / "national_environment.csv",
        INPUTS / "manual_polls.csv",
        OUTPUTS / "manual_polls_clean.csv",
        INPUTS / "polling_averages_generated.csv",
        INPUTS / "bayesian_update_generated.csv",
        OUTPUTS / "race_stats.csv",
        OUTPUTS / "forecast_summary.csv",
        OUTPUTS / "scenario_summary.csv",
    ]

    status_rows = []
    for f in files:
        status_rows.append(
            {
                "File": str(f),
                "Exists": f.exists(),
                "Size KB": fmt_num(f.stat().st_size / 1024, 1) if f.exists() else "—",
            }
        )

    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Raw Outputs")

    with st.expander("forecast_summary.csv"):
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("race_stats.csv"):
        st.dataframe(race_stats, use_container_width=True, hide_index=True)

    with st.expander("bayesian_update_generated.csv"):
        st.dataframe(bayes, use_container_width=True, hide_index=True)
