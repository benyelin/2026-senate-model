from pathlib import Path
import subprocess
import sys
import pandas as pd
import numpy as np

RACE_INPUTS = Path("inputs/race_inputs.csv")


HELP_TEXT = """
Fields this tool can update:

1. incumbency_adjustment_dem
   Positive helps Democrats. Negative helps Republicans.
   Suggested:
     Democratic incumbent running: +2.0
     Republican incumbent running: -2.0
     Open seat / incumbent lost primary: 0.0

2. dem_prior_overperformance / gop_prior_overperformance
   Enter candidate overperformance as a positive or negative number.
   This is evidence of candidate strength relative to partisan baseline.

   Example:
     Democrat won by D+2 in a state baseline of R+3 => +5.0
     Republican won by R+10 in a state baseline of R+4 => +6.0 for GOP

   Recommended standard:
     Use most recent relevant statewide performance if available.
     Otherwise use a recent comparable House/district race.
     Do not use an unweighted career average unless races are comparable.

3. dem_candidate_liability / gop_candidate_liability
   Use positive numbers only.
   Dem liability hurts Democrats.
   GOP liability helps Democrats.

   Liability scale:
     0.0 = no known liability
     0.5 = mild weakness
     1.0 = clear electoral weakness
     1.5 = major documented liability
     2.0 = severe nominee-quality problem

4. manual_candidate_quality_adjustment_dem
   Manual override. Positive helps Democrats, negative helps Republicans.
   Use sparingly for factors not captured elsewhere.

5. nominee confirmed flags
   Candidate quality is gated:
     both nominees confirmed = 100%
     one nominee confirmed = 50%
     neither confirmed = 0%
"""


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def as_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def parse_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ["true", "1", "yes", "y", "confirmed"]


def bool_to_str(x):
    return "True" if parse_bool(x) else "False"


def ensure_column(df, col, default):
    if col not in df.columns:
        df[col] = default
    return df


def display_races(df):
    print()
    print("Senate races")
    print("------------")

    for i, row in df.iterrows():
        state = clean_text(row.get("state", ""))
        dem = clean_text(row.get("dem_candidate", ""))
        gop = clean_text(row.get("gop_candidate", ""))
        inc = as_float(row.get("incumbency_adjustment_dem", 0.0))
        dem_over = as_float(row.get("dem_prior_overperformance", 0.0))
        gop_over = as_float(row.get("gop_prior_overperformance", 0.0))
        dem_liab = as_float(row.get("dem_candidate_liability", 0.0))
        gop_liab = as_float(row.get("gop_candidate_liability", 0.0))
        cq = as_float(row.get("candidate_quality_adjustment_dem", 0.0))

        print(
            f"{i + 1:>2}. {state:<3} "
            f"Dem: {dem or '(blank)'} | GOP: {gop or '(blank)'} | "
            f"Inc {inc:+.1f} | "
            f"Over D {dem_over:+.1f}/R {gop_over:+.1f} | "
            f"Liab D {dem_liab:.1f}/R {gop_liab:.1f} | "
            f"CQ {cq:+.2f}"
        )


def select_race(df):
    while True:
        choice = input("\nSelect race by number or state abbreviation: ").strip()

        if choice == "":
            print("No selection entered.")
            continue

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(df):
                return idx
            print(f"Number must be between 1 and {len(df)}.")
            continue

        state = choice.upper()
        matches = df[df["state"].astype(str).str.upper() == state]

        if len(matches) == 1:
            return int(matches.index[0])

        if len(matches) > 1:
            print(f"Multiple matches for {state}; select by number instead.")
            continue

        print(f"No race found for '{choice}'.")


def prompt_float(label, current, allow_negative=True, blank_keeps=True):
    while True:
        raw = input(f"{label} [{current}]: ").strip()

        if raw == "" and blank_keeps:
            return current

        try:
            val = float(raw)
        except Exception:
            print("Please enter a number.")
            continue

        if not allow_negative and val < 0:
            print("Use a positive number only.")
            continue

        return val


def prompt_bool(label, current):
    current_bool = parse_bool(current)
    current_display = "Y" if current_bool else "N"

    while True:
        raw = input(f"{label} Y/N [{current_display}]: ").strip().lower()

        if raw == "":
            return current_bool

        if raw in ["y", "yes", "true", "1"]:
            return True

        if raw in ["n", "no", "false", "0"]:
            return False

        print("Please enter Y or N.")


def show_current(row):
    print()
    print("Current race fundamentals")
    print("-------------------------")

    fields = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "current_holder",
        "incumbency_adjustment_dem",
        "dem_nominee_confirmed",
        "gop_nominee_confirmed",
        "dem_prior_elected_office",
        "gop_prior_elected_office",
        "dem_prior_statewide_win",
        "gop_prior_statewide_win",
        "dem_prior_overperformance",
        "gop_prior_overperformance",
        "dem_candidate_liability",
        "gop_candidate_liability",
        "manual_candidate_quality_adjustment_dem",
        "objective_candidate_quality_adjustment_dem",
        "candidate_quality_gate",
        "candidate_quality_adjustment_dem",
        "fundamentals_margin_dem",
    ]

    for f in fields:
        if f in row.index:
            print(f"{f}: {row.get(f)}")


def update_notes(df, idx, note):
    if "candidate_quality_manual_note" not in df.columns:
        df["candidate_quality_manual_note"] = ""

    existing = clean_text(df.loc[idx, "candidate_quality_manual_note"])

    if note.strip() == "":
        return

    if existing:
        df.loc[idx, "candidate_quality_manual_note"] = existing + " | " + note.strip()
    else:
        df.loc[idx, "candidate_quality_manual_note"] = note.strip()


def run_script(script_name):
    script = Path(script_name)

    if not script.exists():
        print(f"WARNING: {script_name} not found; skipped.")
        return

    print()
    print("$", sys.executable, script_name)
    proc = subprocess.run([sys.executable, script_name])

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    if not RACE_INPUTS.exists():
        raise FileNotFoundError(f"Could not find {RACE_INPUTS}. Run from the Senate project folder.")

    df = pd.read_csv(RACE_INPUTS)

    if df.empty:
        raise ValueError(f"{RACE_INPUTS} is empty.")

    if "state" not in df.columns:
        raise ValueError("race_inputs.csv must have a state column.")

    df["state"] = df["state"].astype(str).str.strip().str.upper()

    defaults = {
        "dem_candidate": "",
        "gop_candidate": "",
        "current_holder": "",
        "incumbency_adjustment_dem": 0.0,
        "dem_nominee_confirmed": False,
        "gop_nominee_confirmed": False,
        "dem_prior_elected_office": False,
        "gop_prior_elected_office": False,
        "dem_prior_statewide_win": False,
        "gop_prior_statewide_win": False,
        "dem_prior_overperformance": 0.0,
        "gop_prior_overperformance": 0.0,
        "dem_candidate_liability": 0.0,
        "gop_candidate_liability": 0.0,
        "manual_candidate_quality_adjustment_dem": 0.0,
        "candidate_quality_manual_note": "",
    }

    for col, default in defaults.items():
        df = ensure_column(df, col, default)

    numeric_cols = [
        "incumbency_adjustment_dem",
        "dem_prior_overperformance",
        "gop_prior_overperformance",
        "dem_candidate_liability",
        "gop_candidate_liability",
        "manual_candidate_quality_adjustment_dem",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    print(HELP_TEXT)

    display_races(df)

    idx = select_race(df)
    row = df.loc[idx].copy()

    show_current(row)

    state = clean_text(row.get("state", ""))
    dem = clean_text(row.get("dem_candidate", ""))
    gop = clean_text(row.get("gop_candidate", ""))

    print()
    print(f"Selected {state}: {dem or '(blank Dem)'} vs {gop or '(blank GOP)'}")
    print()
    print("Press Enter to keep the current value for any field.")
    print()

    new_values = {}

    new_values["incumbency_adjustment_dem"] = prompt_float(
        "Incumbency adjustment Dem. D incumbent +2, GOP incumbent -2, open 0",
        as_float(row.get("incumbency_adjustment_dem", 0.0)),
        allow_negative=True,
    )

    print()
    print("Nominee confirmation flags affect how much candidate quality applies.")
    new_values["dem_nominee_confirmed"] = prompt_bool(
        "Dem nominee confirmed",
        row.get("dem_nominee_confirmed", False),
    )
    new_values["gop_nominee_confirmed"] = prompt_bool(
        "GOP nominee confirmed",
        row.get("gop_nominee_confirmed", False),
    )

    print()
    print("Prior elected/statewide experience.")
    new_values["dem_prior_elected_office"] = prompt_bool(
        "Dem prior elected office",
        row.get("dem_prior_elected_office", False),
    )
    new_values["gop_prior_elected_office"] = prompt_bool(
        "GOP prior elected office",
        row.get("gop_prior_elected_office", False),
    )
    new_values["dem_prior_statewide_win"] = prompt_bool(
        "Dem prior statewide win",
        row.get("dem_prior_statewide_win", False),
    )
    new_values["gop_prior_statewide_win"] = prompt_bool(
        "GOP prior statewide win",
        row.get("gop_prior_statewide_win", False),
    )

    print()
    print("Candidate overperformance.")
    print("Use most recent relevant statewide result where possible, not a raw career average.")
    new_values["dem_prior_overperformance"] = prompt_float(
        "Dem prior overperformance",
        as_float(row.get("dem_prior_overperformance", 0.0)),
        allow_negative=True,
    )
    new_values["gop_prior_overperformance"] = prompt_float(
        "GOP prior overperformance",
        as_float(row.get("gop_prior_overperformance", 0.0)),
        allow_negative=True,
    )

    print()
    print("Candidate liability. Positive numbers only.")
    new_values["dem_candidate_liability"] = prompt_float(
        "Dem candidate liability",
        as_float(row.get("dem_candidate_liability", 0.0)),
        allow_negative=False,
    )
    new_values["gop_candidate_liability"] = prompt_float(
        "GOP candidate liability",
        as_float(row.get("gop_candidate_liability", 0.0)),
        allow_negative=False,
    )

    print()
    print("Manual candidate-quality override. Positive helps Democrats; negative helps Republicans.")
    new_values["manual_candidate_quality_adjustment_dem"] = prompt_float(
        "Manual CQ adjustment Dem",
        as_float(row.get("manual_candidate_quality_adjustment_dem", 0.0)),
        allow_negative=True,
    )

    print()
    print("Proposed changes")
    print("----------------")

    changed = []

    for field, new in new_values.items():
        old = row.get(field)

        if isinstance(new, bool):
            old_norm = parse_bool(old)
            changed_flag = old_norm != new
            old_display = bool_to_str(old)
            new_display = bool_to_str(new)
        else:
            old_norm = as_float(old, 0.0)
            changed_flag = abs(old_norm - float(new)) > 1e-9
            old_display = f"{old_norm}"
            new_display = f"{new}"

        if changed_flag:
            changed.append((field, old_display, new_display))
            print(f"{field}: {old_display} -> {new_display}")

    if not changed:
        print("No changes.")
        return

    note = input("\nOptional note for this update: ").strip()

    confirm = input("\nSave these changes? Type YES to save: ").strip().upper()

    if confirm != "YES":
        print("No changes saved.")
        return

    for field, new in new_values.items():
        df.loc[idx, field] = new

    if note:
        update_notes(df, idx, f"{state}: {note}")

    df.to_csv(RACE_INPUTS, index=False)

    print()
    print(f"Saved changes to {RACE_INPUTS}")

    run_script("update_candidate_quality.py")
    run_script("recalculate_fundamentals.py")

    updated = pd.read_csv(RACE_INPUTS)
    updated["state"] = updated["state"].astype(str).str.strip().str.upper()
    updated_row = updated[updated["state"] == state].iloc[0]

    print()
    print("Updated race fundamentals")
    print("-------------------------")

    show_current(updated_row)

    print()
    print("Next recommended command:")
    print("  python3 run_full_pipeline.py")


if __name__ == "__main__":
    main()
