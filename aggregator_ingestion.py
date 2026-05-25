
"""
P1 Aggregator ingestion tools.

This is deliberately semi-automated and transparent:
- Pull public pages when accessible.
- Save raw HTML snapshots for audit.
- Parse what can be parsed reliably.
- Write generated CSVs that feed the existing pipeline.

Important: aggregator sites often change markup and some may block automated access.
If parsing fails, use the generated audit files and update CSVs manually.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}


def fetch_url(url: str, timeout: int = 20) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def save_snapshot(text: str, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    path = output_dir / f"{safe}.html"
    path.write_text(text, encoding="utf-8")
    return path


def parse_first_margin_text(text: str) -> Optional[float]:
    """
    Parse strings like:
    - Democrats +6
    - Democratic +6.8
    - D+6.8
    - Republicans +2
    - R+2.5

    Returns Democratic margin, so Republican leads are negative.
    """
    patterns = [
        r"Democrats?\s*\+?\s*([0-9]+(?:\.[0-9]+)?)",
        r"Democratic\s*\+?\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bD\s*\+?\s*([0-9]+(?:\.[0-9]+)?)",
        r"Republicans?\s*\+?\s*([0-9]+(?:\.[0-9]+)?)",
        r"Republican\s*\+?\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bR\s*\+?\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for i, pat in enumerate(patterns):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = float(m.group(1))
            # Republican patterns are last three
            if i >= 3:
                return -val
            return val
    return None


def parse_generic_ballot_from_page(html: str) -> Optional[float]:
    # First try visible text, which works for pages that display "Democrats +6".
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    return parse_first_margin_text(text)


def update_national_environment_from_sources(
    input_dir: str | Path = "inputs",
    source_preference: str = "auto",
    presidential_approval: Optional[float] = None,
    approval_slope: Optional[float] = None,
    midterm_effect_dem: Optional[float] = None,
) -> pd.DataFrame:
    """
    Updates inputs/national_environment.csv where possible.

    Generic ballot is pulled from enabled national_generic_ballot rows
    in inputs/aggregator_sources.csv. If multiple work, the first enabled
    source wins unless source_preference is a source_name substring.

    Presidential approval is not scraped by default because approval pages vary
    more and are often paywalled/JS-heavy. Pass --presidential-approval to update it.
    """
    input_dir = Path(input_dir)
    sources_path = input_dir / "aggregator_sources.csv"
    env_path = input_dir / "national_environment.csv"

    sources = pd.read_csv(sources_path)
    env = pd.read_csv(env_path)

    audit_rows = []
    html_dir = input_dir / "aggregator_snapshots"

    candidates = sources[(sources["source_type"] == "national_generic_ballot") & (sources["enabled"].astype(int) == 1)]
    if source_preference and source_preference != "auto":
        preferred = candidates[candidates["source_name"].str.contains(source_preference, case=False, na=False)]
        if not preferred.empty:
            candidates = preferred

    generic_margin = None
    for _, row in candidates.iterrows():
        source_name = row["source_name"]
        url = row["url"]
        try:
            html = fetch_url(url)
            save_snapshot(html, html_dir, source_name)
            parsed = parse_generic_ballot_from_page(html)
            audit_rows.append({
                "kind": "generic_ballot",
                "source_name": source_name,
                "url": url,
                "status": "parsed" if parsed is not None else "fetched_not_parsed",
                "value": parsed,
            })
            if parsed is not None:
                generic_margin = parsed
                break
        except Exception as e:
            audit_rows.append({
                "kind": "generic_ballot",
                "source_name": source_name,
                "url": url,
                "status": f"error: {type(e).__name__}: {e}",
                "value": None,
            })

    def set_param(param: str, value):
        nonlocal env
        if value is None:
            return
        mask = env["parameter"] == param
        if mask.any():
            env.loc[mask, "value"] = value
        else:
            env = pd.concat([
                env,
                pd.DataFrame([{"parameter": param, "value": value, "description": "Generated by aggregator ingestion"}])
            ], ignore_index=True)

    set_param("generic_ballot_dem_margin", generic_margin)
    set_param("presidential_approval", presidential_approval)
    set_param("approval_slope", approval_slope)
    set_param("midterm_effect_dem", midterm_effect_dem)

    env.to_csv(env_path, index=False)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(input_dir / "national_environment_import_audit.csv", index=False)
    return audit


def parse_rcp_average_from_tables(html: str, state: str) -> Optional[dict]:
    """
    Best-effort parser for RCP/RealClearPolling pages.
    Looks for a row containing 'RCP Average' and a spread like 'Ossoff +14.0'.
    """
    try:
        tables = pd.read_html(html)
    except Exception:
        tables = []

    for table in tables:
        flat = table.astype(str)
        joined_rows = flat.apply(lambda r: " | ".join(r.values), axis=1)
        for row_text in joined_rows:
            if "RCP" in row_text and "Average" in row_text:
                # Find a spread at the end-ish, e.g. "Ossoff +14.0" or "Republican +2"
                margin = parse_first_margin_text(row_text)
                if margin is not None:
                    return {
                        "state": state,
                        "polling_margin_dem": margin,
                        "raw_row": row_text,
                        "parser": "pandas_read_html_rcp_average",
                    }
                # Candidate-name spread parser. If the D candidate name is unknown, flag for manual review.
                m = re.search(r"([A-Za-z .'-]+)\s+\+([0-9]+(?:\.[0-9]+)?)", row_text)
                if m:
                    return {
                        "state": state,
                        "polling_margin_dem": None,
                        "raw_row": row_text,
                        "parser": "needs_candidate_party_mapping",
                    }
    return None


def update_polling_from_aggregators(input_dir: str | Path = "inputs") -> pd.DataFrame:
    """
    Best-effort Senate race polling importer.

    For each enabled senate_race source, tries to parse an RCP average.
    Output:
    - inputs/aggregator_race_polling_generated.csv
    - updates inputs/polls_raw.csv with a synthetic aggregator poll row when margin is parseable

    Because Senate matchup pages often have candidate-specific tables, this is intentionally conservative.
    """
    input_dir = Path(input_dir)
    sources = pd.read_csv(input_dir / "aggregator_sources.csv")
    race_sources = sources[(sources["source_type"] == "senate_race") & (sources["enabled"].astype(int) == 1)]

    html_dir = input_dir / "aggregator_snapshots"
    out_rows = []

    for _, row in race_sources.iterrows():
        state = row.get("state", "")
        source_name = row["source_name"]
        url = row["url"]
        if not isinstance(state, str) or not state.strip():
            continue

        try:
            html = fetch_url(url)
            save_snapshot(html, html_dir, source_name)
            parsed = parse_rcp_average_from_tables(html, state.strip())
            if parsed is None:
                out_rows.append({
                    "state": state,
                    "source_name": source_name,
                    "url": url,
                    "status": "fetched_not_parsed",
                    "polling_margin_dem": None,
                    "raw_row": "",
                })
            else:
                out_rows.append({
                    "state": state,
                    "source_name": source_name,
                    "url": url,
                    "status": "parsed" if parsed["polling_margin_dem"] is not None else "needs_manual_party_mapping",
                    "polling_margin_dem": parsed["polling_margin_dem"],
                    "raw_row": parsed["raw_row"],
                })
        except Exception as e:
            out_rows.append({
                "state": state,
                "source_name": source_name,
                "url": url,
                "status": f"error: {type(e).__name__}: {e}",
                "polling_margin_dem": None,
                "raw_row": "",
            })

    out = pd.DataFrame(out_rows)
    out.to_csv(input_dir / "aggregator_race_polling_generated.csv", index=False)

    # Do not write aggregator polling into polls_raw.csv.
    # Manual polls are now the only polling source.
    return out

if Path(manual_polls_path).exists():
    manual_polls = pd.read_csv(manual_polls_path)
    polls = pd.concat([polls, manual_polls], ignore_index=True)
    synthetic_rows = []
    for _, r in out.dropna(subset=["polling_margin_dem"]).iterrows():
        margin = float(r["polling_margin_dem"])
        # Use 50 +/- margin/2 as synthetic two-party shares.
        dem = 50 + margin / 2
        gop = 50 - margin / 2
        synthetic_rows.append({
            "state": r["state"],
            "pollster": r["source_name"],
            "start_date": pd.Timestamp.today().date().isoformat(),
            "end_date": pd.Timestamp.today().date().isoformat(),
            "dem_candidate": "Aggregator average",
            "gop_candidate": "Aggregator average",
            "dem_share": dem,
            "gop_share": gop,
            "sample_size": "",
            "pollster_rating": "B",
            "mode": "aggregator",
            "sponsor": "",
            "notes": f"Generated from {r['url']}",
        })

    if synthetic_rows:
        polls = pd.concat([polls, pd.DataFrame(synthetic_rows)], ignore_index=True)
        polls.to_csv(polls_path, index=False)

    return out
