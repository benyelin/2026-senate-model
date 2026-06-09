from pathlib import Path
import pandas as pd
import streamlit as st

POLL_PATH = Path("inputs/manual_polls.csv")

st.set_page_config(page_title="Senate Poll Metadata Editor", layout="wide")

st.title("Senate Poll Metadata Editor")
st.caption(
    "Use this page to mark whether Senate polls are partisan, sponsored, or internal. "
    "These fields feed the partisan pollster adjustment script."
)

if not POLL_PATH.exists():
    st.error(f"Could not find {POLL_PATH}")
    st.stop()

df = pd.read_csv(POLL_PATH)

metadata_defaults = {
    "poll_sponsor_type": "",
    "partisan_sponsor_party": "",
    "is_internal_poll": False,
    "pollster_partisan_affiliation": "",
    "partisan_pollster_review_notes": "",
}

for col, default in metadata_defaults.items():
    if col not in df.columns:
        df[col] = default

for col in [
    "poll_sponsor_type",
    "partisan_sponsor_party",
    "pollster_partisan_affiliation",
    "partisan_pollster_review_notes",
]:
    df[col] = (
        df[col]
        .fillna("")
        .astype(str)
        .replace({"nan": "", "None": "", "NaN": ""})
    )

df["is_internal_poll"] = (
    df["is_internal_poll"]
    .fillna(False)
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes", "y"])
)

df["_row_id"] = range(len(df))

display_cols = [
    "_row_id",
    "race",
    "state",
    "pollster",
    "sponsor",
    "start_date",
    "end_date",
    "poll_sponsor_type",
    "partisan_sponsor_party",
    "is_internal_poll",
    "pollster_partisan_affiliation",
    "partisan_pollster_review_notes",
]

display_cols = [c for c in display_cols if c in df.columns]

editor_df = df[display_cols].copy()

edited = st.data_editor(
    editor_df,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    disabled=[
        c for c in [
            "_row_id",
            "race",
            "state",
            "pollster",
            "sponsor",
            "start_date",
            "end_date",
        ]
        if c in editor_df.columns
    ],
    column_config={
        "poll_sponsor_type": st.column_config.SelectboxColumn(
            "Sponsor type",
            options=[
                "",
                "independent",
                "media",
                "university",
                "party",
                "campaign",
                "super PAC",
                "other",
            ],
        ),
        "partisan_sponsor_party": st.column_config.SelectboxColumn(
            "Sponsor party",
            options=["", "D", "R", "none", "unknown"],
        ),
        "is_internal_poll": st.column_config.CheckboxColumn(
            "Internal/campaign poll",
            default=False,
        ),
        "pollster_partisan_affiliation": st.column_config.SelectboxColumn(
            "Pollster partisan affiliation",
            options=["", "D", "R", "none", "unknown"],
        ),
        "partisan_pollster_review_notes": st.column_config.TextColumn(
            "Partisan poll notes",
        ),
    },
    key="senate_poll_metadata_editor",
)

st.divider()

if st.button("Save Poll Metadata", type="primary"):
    updated = df.copy()

    for _, row in edited.iterrows():
        idx = int(row["_row_id"])

        for col in metadata_defaults:
            if col in row.index:
                updated.loc[idx, col] = row[col]

    updated = updated.drop(columns=["_row_id"], errors="ignore")
    updated.to_csv(POLL_PATH, index=False)

    st.success(f"Saved poll metadata to {POLL_PATH}")
    st.info(
        "Next run the Senate full pipeline so the partisan pollster adjustment file is regenerated."
    )

st.markdown("### How to use these fields")

st.write(
    """
    - Public/media poll: leave sponsor party blank or use `none`; internal unchecked.
    - Democratic internal: sponsor type `campaign`, sponsor party `D`, internal checked.
    - Republican internal: sponsor type `campaign`, sponsor party `R`, internal checked.
    - Known partisan pollster but not sponsored: use pollster partisan affiliation `D` or `R`; sponsor party can stay blank or `none`.
    """
)
