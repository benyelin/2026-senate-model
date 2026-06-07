from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

RACE_INPUTS_PATH = INPUTS / "race_inputs.csv"
WAR_PATH = INPUTS / "candidate_war.csv"
SETTINGS_PATH = INPUTS / "senate_calibration_settings.csv"

AUDIT_OUTPUT = OUTPUTS / "senate_candidate_war_audit.csv"

DEFAULT_MIN_NAME_SCORE = 0.90
DEFAULT_SENATE_WAR_SHRINKAGE = 0.35
DEFAULT_SENATE_WAR_CAP = 3.0
DEFAULT_ONE_SIDED_MULTIPLIER = 0.50
DEFAULT_HOUSE_TO_SENATE_TRANSFER_MULTIPLIER = 0.50

CYCLE_WEIGHTS = {
    2024: 1.00,
    2022: 0.65,
    2020: 0.40,
    2018: 0.25,
    2016: 0.15,
    2014: 0.10,
}


def read_settings():
    if not SETTINGS_PATH.exists():
        return {}

    try:
        df = pd.read_csv(SETTINGS_PATH)
    except Exception:
        return {}

    if df.empty or "setting" not in df.columns or "value" not in df.columns:
        return {}

    out = {}

    for _, row in df.iterrows():
        key = str(row.get("setting", "")).strip()
        try:
            out[key] = float(row.get("value"))
        except Exception:
            continue

    return out


def setting(settings, key, default):
    return float(settings.get(key, default))


def normalize_name(name):
    if pd.isna(name):
        return ""

    s = str(name).strip()

    # Convert "Last, First Middle" to "First Middle Last".
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.upper()

    nickname_map = {
        "MIKE": "MICHAEL",
        "DAVE": "DAVID",
        "DAN": "DANIEL",
        "BOB": "ROBERT",
        "ROB": "ROBERT",
        "BILL": "WILLIAM",
        "WILL": "WILLIAM",
        "CHUCK": "CHARLES",
        "TOM": "THOMAS",
        "JIM": "JAMES",
        "JIMMY": "JAMES",
        "JOE": "JOSEPH",
        "PAT": "PATRICK",
        "CHRIS": "CHRISTOPHER",
        "MATT": "MATTHEW",
        "BEN": "BENJAMIN",
        "SAM": "SAMUEL",
        "LIZ": "ELIZABETH",
        "KATE": "KATHERINE",
        "KATIE": "KATHERINE",
        "KATHY": "KATHERINE",
        "KIM": "KIMBERLY",
    }

    s = re.sub(r"[^A-Z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = [
        t for t in s.split()
        if t not in {"JR", "SR", "II", "III", "IV", "V"}
    ]

    tokens = [nickname_map.get(t, t) for t in tokens]

    return " ".join(tokens)


def name_parts(norm):
    if not norm:
        return []

    return [
        p for p in norm.split()
        if p not in {"THE", "DE", "DA", "DEL", "VAN", "VON"}
    ]


def name_similarity(a, b):
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)

    if not a_norm or not b_norm:
        return 0.0

    if a_norm == b_norm:
        return 1.0

    a_parts = name_parts(a_norm)
    b_parts = name_parts(b_norm)

    if not a_parts or not b_parts:
        return 0.0

    a_first, a_last = a_parts[0], a_parts[-1]
    b_first, b_last = b_parts[0], b_parts[-1]

    if a_first == b_first and a_last == b_last:
        return 0.97

    if a_first[:1] == b_first[:1] and a_last == b_last:
        return 0.90

    if a_last == b_last and len(set(a_parts).intersection(set(b_parts))) >= 2:
        return 0.88

    if a_last == b_last:
        return 0.75

    return 0.0


def standardize_party(x):
    s = str(x).strip().upper()

    if s in {"D", "DEM", "DEMOCRAT", "DEMOCRATIC", "DFL"}:
        return "D"

    if s in {"R", "REP", "REPUBLICAN", "GOP"}:
        return "R"

    return s


def read_csv_with_fallback(path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]

    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"Read {path} using encoding={encoding}")
            return df
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")


def parse_war_margin_to_dem_net(value, sortable=None):
    """
    Convert race-level WAR notation into Democratic net overperformance.

    Examples:
      D+1.3 -> +1.3
      R+0.2 -> -0.2

    In the House export you used earlier, Sortable appeared opposite-signed:
      D+1.3 -> -1.3
      R+0.2 -> +0.2
    """
    s = str(value).strip().upper()

    if s and s not in {"NAN", "NONE", ""}:
        m = re.match(r"^([DR])\s*\+\s*([0-9.]+)", s)
        if m:
            side = m.group(1)
            amount = float(m.group(2))
            return amount if side == "D" else -amount

        m = re.match(r"^([DR])\s*-\s*([0-9.]+)", s)
        if m:
            side = m.group(1)
            amount = float(m.group(2))
            return -amount if side == "D" else amount

        try:
            return float(s)
        except Exception:
            pass

    try:
        return -float(sortable)
    except Exception:
        return 0.0


def find_col(columns, options, required=False):
    lower_map = {str(c).strip().lower(): c for c in columns}

    for opt in options:
        key = opt.lower()
        if key in lower_map:
            return lower_map[key]

    for c in columns:
        cl = str(c).strip().lower()
        for opt in options:
            if opt.lower() in cl:
                return c

    if required:
        raise ValueError(
            f"Could not find required column. Tried: {options}. Available: {list(columns)}"
        )

    return None


def load_wide_war_export(war):
    """
    Handles your current export format:
      Year, Chamber, Geography, Democrat, Republican, WAR, Sortable

    Converts race-level WAR to candidate-level rows.
    """
    required = {"Year", "Chamber", "Geography", "Democrat", "Republican", "WAR"}

    if not required.issubset(set(war.columns)):
        return None

    sortable_col = "Sortable" if "Sortable" in war.columns else None

    rows = []

    for _, row in war.iterrows():
        chamber = str(row.get("Chamber", "")).strip().upper()

        year = pd.to_numeric(row.get("Year"), errors="coerce")
        if pd.isna(year):
            continue

        geography = str(row.get("Geography", "")).strip().upper()
        state = geography[:2] if len(geography) >= 2 else ""

        dem_name = row.get("Democrat", "")
        rep_name = row.get("Republican", "")

        sortable = row.get(sortable_col) if sortable_col else None
        dem_net = parse_war_margin_to_dem_net(row.get("WAR"), sortable)

        # Split race-level net performance symmetrically.
        dem_score = dem_net / 2.0
        rep_score = -dem_net / 2.0

        if str(dem_name).strip():
            rows.append(
                {
                    "war_candidate_name": dem_name,
                    "war_candidate_norm": normalize_name(dem_name),
                    "war_party": "D",
                    "war_cycle": int(year),
                    "war_score_raw": dem_score,
                    "war_chamber": chamber,
                    "war_geography": geography,
                    "war_state": state,
                }
            )

        if str(rep_name).strip():
            rows.append(
                {
                    "war_candidate_name": rep_name,
                    "war_candidate_norm": normalize_name(rep_name),
                    "war_party": "R",
                    "war_cycle": int(year),
                    "war_score_raw": rep_score,
                    "war_chamber": chamber,
                    "war_geography": geography,
                    "war_state": state,
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["war_cycle_weight"] = out["war_cycle"].map(CYCLE_WEIGHTS).fillna(0.05)
    out["war_weighted_score"] = out["war_score_raw"] * out["war_cycle_weight"]

    return out


def load_generic_candidate_war(war):
    candidate_col = find_col(
        war.columns,
        ["candidate_name", "candidate", "name", "person", "cand"],
        required=True,
    )

    party_col = find_col(
        war.columns,
        ["party", "candidate_party", "party_code"],
        required=True,
    )

    cycle_col = find_col(
        war.columns,
        ["cycle", "year", "election_year"],
        required=True,
    )

    war_col = find_col(
        war.columns,
        ["war_margin", "war", "wins_above_replacement", "wins above replacement", "candidate_war", "value"],
        required=True,
    )

    chamber_col = find_col(war.columns, ["chamber", "office"], required=False)
    state_col = find_col(war.columns, ["state", "state_code"], required=False)
    geo_col = find_col(war.columns, ["geography", "district_id", "district", "race"], required=False)

    out = pd.DataFrame()
    out["war_candidate_name"] = war[candidate_col]
    out["war_candidate_norm"] = war[candidate_col].apply(normalize_name)
    out["war_party"] = war[party_col].apply(standardize_party)
    out["war_cycle"] = pd.to_numeric(war[cycle_col], errors="coerce")
    out["war_score_raw"] = pd.to_numeric(war[war_col], errors="coerce")

    out["war_chamber"] = war[chamber_col].astype(str).str.strip().str.upper() if chamber_col else ""
    out["war_geography"] = war[geo_col].astype(str).str.strip().str.upper() if geo_col else ""

    if state_col:
        out["war_state"] = war[state_col].fillna("").astype(str).str.strip().str.upper()
    else:
        out["war_state"] = out["war_geography"].str.extract(r"^([A-Z]{2})", expand=False).fillna("")

    out = out.dropna(subset=["war_cycle", "war_score_raw"])
    out["war_cycle"] = out["war_cycle"].astype(int)

    out["war_cycle_weight"] = out["war_cycle"].map(CYCLE_WEIGHTS).fillna(0.05)
    out["war_weighted_score"] = out["war_score_raw"] * out["war_cycle_weight"]

    return out


def load_war():
    if not WAR_PATH.exists():
        raise FileNotFoundError("inputs/candidate_war.csv not found.")

    war = read_csv_with_fallback(WAR_PATH)

    print(f"Using WAR file: {WAR_PATH}")
    print(f"WAR columns: {list(war.columns)}")

    wide = load_wide_war_export(war)

    if wide is not None:
        print("Detected wide WAR export format.")
        print(f"Converted to {len(wide)} candidate-level WAR rows.")
        return wide

    return load_generic_candidate_war(war)


def aggregate_candidate_war(war):
    grouped = (
        war.groupby(["war_candidate_norm", "war_party"], as_index=False)
        .agg(
            war_candidate_name=("war_candidate_name", "last"),
            war_score_weighted_sum=("war_weighted_score", "sum"),
            war_weight_sum=("war_cycle_weight", "sum"),
            war_cycles=("war_cycle", lambda x: ",".join(str(int(v)) for v in sorted(set(x), reverse=True))),
            war_observations=("war_score_raw", "count"),
            war_latest_cycle=("war_cycle", "max"),
            war_chambers=("war_chamber", lambda x: ",".join(sorted(set(str(v) for v in x if str(v).strip())))),
            war_states=("war_state", lambda x: ",".join(sorted(set(str(v) for v in x if str(v).strip())))),
        )
    )

    grouped["candidate_war_recency_weighted"] = (
        grouped["war_score_weighted_sum"] / grouped["war_weight_sum"].replace(0, np.nan)
    )

    grouped["candidate_war_recency_weighted"] = grouped["candidate_war_recency_weighted"].fillna(0.0)

    return grouped


def match_candidate(candidate_name, party, state, war_agg, min_score):
    candidate_norm = normalize_name(candidate_name)

    if not candidate_norm:
        return None

    pool = war_agg[war_agg["war_party"].eq(party)].copy()

    if pool.empty:
        return None

    state = str(state).strip().upper()

    pool["name_match_score"] = pool["war_candidate_norm"].apply(
        lambda x: name_similarity(candidate_norm, x)
    )

    if "war_states" in pool.columns:
        pool["same_state"] = pool["war_states"].astype(str).apply(
            lambda states: state in {s.strip().upper() for s in states.split(",") if s.strip()}
        )
    else:
        pool["same_state"] = False

    # Conservative rule:
    # - exact/near-exact full-name match is usable
    # - first-initial/last-name match only if same state
    pool["usable_match"] = (
        (pool["name_match_score"] >= 0.97)
        | ((pool["same_state"]) & (pool["name_match_score"] >= min_score))
    )

    pool = pool[pool["usable_match"]].copy()

    if pool.empty:
        return None

    pool = pool.sort_values(
        ["same_state", "name_match_score", "war_latest_cycle", "war_observations"],
        ascending=[False, False, False, False],
    )

    return pool.iloc[0]


def numeric_value(row, col):
    if col not in row.index:
        return 0.0

    val = pd.to_numeric(row.get(col), errors="coerce")

    if pd.isna(val):
        return 0.0

    return float(val)


def current_model_candidate_adjustment(row):
    """
    Total current Senate candidate/race-specific adjustment, excluding baseline and environment.

    This lets us compare WAR-implied candidate quality against what the model already says.
    """
    cols = [
        "overperformance_adjustment_dem",
        "candidate_liability_adjustment_dem",
        "objective_candidate_quality_adjustment_dem",
        "manual_candidate_quality_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]

    return sum(numeric_value(row, col) for col in cols)


def main():
    settings = read_settings()

    min_score = setting(settings, "senate_candidate_war_min_name_score", DEFAULT_MIN_NAME_SCORE)
    shrinkage = setting(settings, "senate_candidate_war_shrinkage", DEFAULT_SENATE_WAR_SHRINKAGE)
    cap = setting(settings, "senate_candidate_war_cap", DEFAULT_SENATE_WAR_CAP)
    one_sided_multiplier = setting(settings, "senate_candidate_war_one_sided_multiplier", DEFAULT_ONE_SIDED_MULTIPLIER)
    house_to_senate_multiplier = setting(
        settings,
        "senate_house_war_transfer_multiplier",
        DEFAULT_HOUSE_TO_SENATE_TRANSFER_MULTIPLIER,
    )

    if not RACE_INPUTS_PATH.exists():
        raise FileNotFoundError("inputs/race_inputs.csv not found.")

    races = pd.read_csv(RACE_INPUTS_PATH)

    if "state" not in races.columns:
        raise ValueError("race_inputs.csv must contain state.")

    for col in ["dem_candidate", "gop_candidate"]:
        if col not in races.columns:
            races[col] = ""

    war = load_war()
    war_agg = aggregate_candidate_war(war)

    rows = []

    for _, row in races.iterrows():
        state = str(row.get("state", "")).strip().upper()
        dem_candidate = row.get("dem_candidate", "")
        gop_candidate = row.get("gop_candidate", "")

        dem_match = match_candidate(dem_candidate, "D", state, war_agg, min_score)
        gop_match = match_candidate(gop_candidate, "R", state, war_agg, min_score)

        dem_war = float(dem_match["candidate_war_recency_weighted"]) if dem_match is not None else 0.0
        gop_war = float(gop_match["candidate_war_recency_weighted"]) if gop_match is not None else 0.0

        dem_chambers = str(dem_match["war_chambers"]) if dem_match is not None else ""
        gop_chambers = str(gop_match["war_chambers"]) if gop_match is not None else ""

        # House WAR transfers imperfectly to statewide Senate.
        dem_transfer = house_to_senate_multiplier if "HOUSE" in dem_chambers and "SENATE" not in dem_chambers else 1.0
        gop_transfer = house_to_senate_multiplier if "HOUSE" in gop_chambers and "SENATE" not in gop_chambers else 1.0

        dem_effective_war = dem_war * dem_transfer
        gop_effective_war = gop_war * gop_transfer

        raw_net_dem = dem_effective_war - gop_effective_war
        shrunk = raw_net_dem * shrinkage
        capped_before_match_quality = float(np.clip(shrunk, -cap, cap))

        if dem_match is not None and gop_match is not None:
            match_status = "Both matched"
            match_quality_multiplier = 1.0
        elif dem_match is not None:
            match_status = "Only D matched"
            match_quality_multiplier = one_sided_multiplier
        elif gop_match is not None:
            match_status = "Only R matched"
            match_quality_multiplier = one_sided_multiplier
        else:
            match_status = "Neither matched"
            match_quality_multiplier = 0.0

        proposed_war_adjustment_dem = capped_before_match_quality * match_quality_multiplier

        current_adjustment = current_model_candidate_adjustment(row)
        gap = proposed_war_adjustment_dem - current_adjustment

        if abs(gap) >= 2.0:
            review_priority = "High"
        elif abs(gap) >= 1.0:
            review_priority = "Medium"
        else:
            review_priority = "Low"

        rows.append(
            {
                "state": state,
                "dem_candidate": dem_candidate,
                "gop_candidate": gop_candidate,
                "war_match_status": match_status,
                "dem_war_name": dem_match["war_candidate_name"] if dem_match is not None else "",
                "gop_war_name": gop_match["war_candidate_name"] if gop_match is not None else "",
                "dem_name_match_score": float(dem_match["name_match_score"]) if dem_match is not None else 0.0,
                "gop_name_match_score": float(gop_match["name_match_score"]) if gop_match is not None else 0.0,
                "dem_candidate_war": dem_war,
                "gop_candidate_war": gop_war,
                "dem_effective_war": dem_effective_war,
                "gop_effective_war": gop_effective_war,
                "dem_war_cycles": dem_match["war_cycles"] if dem_match is not None else "",
                "gop_war_cycles": gop_match["war_cycles"] if gop_match is not None else "",
                "dem_war_chambers": dem_chambers,
                "gop_war_chambers": gop_chambers,
                "candidate_war_net_dem": raw_net_dem,
                "senate_candidate_war_shrinkage": shrinkage,
                "senate_candidate_war_cap": cap,
                "senate_candidate_war_one_sided_multiplier": one_sided_multiplier,
                "senate_house_war_transfer_multiplier": house_to_senate_multiplier,
                "candidate_war_adjustment_dem_before_match_quality": capped_before_match_quality,
                "candidate_war_match_quality_multiplier": match_quality_multiplier,
                "proposed_war_adjustment_dem": proposed_war_adjustment_dem,
                "current_model_candidate_adjustment_dem": current_adjustment,
                "war_minus_current_adjustment_dem": gap,
                "war_review_priority": review_priority,
                "overperformance_adjustment_dem": row.get("overperformance_adjustment_dem", np.nan),
                "candidate_liability_adjustment_dem": row.get("candidate_liability_adjustment_dem", np.nan),
                "candidate_quality_adjustment_dem": row.get("candidate_quality_adjustment_dem", np.nan),
                "special_adjustment_dem": row.get("special_adjustment_dem", np.nan),
                "fundamentals_margin_dem": row.get("fundamentals_margin_dem", np.nan),
                "model_margin_dem": row.get("model_margin_dem", np.nan),
                "simulated_dem_win_prob": row.get("simulated_dem_win_prob", np.nan),
            }
        )

    audit = pd.DataFrame(rows)

    OUTPUTS.mkdir(exist_ok=True)
    audit.to_csv(AUDIT_OUTPUT, index=False)

    print(f"Wrote {AUDIT_OUTPUT}")

    print()
    print("Senate WAR match status")
    print("-----------------------")
    print(audit["war_match_status"].value_counts(dropna=False).to_string())

    print()
    print("Review priority")
    print("---------------")
    print(audit["war_review_priority"].value_counts(dropna=False).to_string())

    print()
    print("Largest WAR/current adjustment gaps")
    print("-----------------------------------")

    show_cols = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "war_match_status",
        "proposed_war_adjustment_dem",
        "current_model_candidate_adjustment_dem",
        "war_minus_current_adjustment_dem",
        "war_review_priority",
        "dem_war_name",
        "gop_war_name",
        "dem_war_cycles",
        "gop_war_cycles",
    ]

    print(
        audit.reindex(audit["war_minus_current_adjustment_dem"].abs().sort_values(ascending=False).index)
        [show_cols]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
