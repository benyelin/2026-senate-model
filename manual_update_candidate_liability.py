from pathlib import Path
import subprocess
import sys
import pandas as pd
import numpy as np

RACE_INPUTS = Path("inputs/race_inputs.csv")

LIABILITY_SCALE = """
Candidate liability scale:
  0.0 = no known candidate-specific liability
  0.5 = mild weakness
  1.0 = clear electoral weakness
  1.5 = major documented liability
  2.0 = severe nominee-quality problem

Use positive numbers only.
Dem liability hurts Democrats.
GOP liability helps Democrats.
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
        dem_liab = as_float(row.get("dem_candidate_liability", 0.0))
        gop_liab = as_float(row.get("gop_candidate_liability", 0.0))
        cq = as_float(row.get("candidate_quality_adjustment_dem", 0.0))

        print(
            f"{i + 1:>2}. {state:<3} "
            f"Dem: {dem or '(blank)'} [liability {dem_liab:.1f}] | "
            f"GOP: {gop or '(blank)'} [liability {gop_liab:.1f}] | "
            f"Current CQ: {cq:+.2f}"
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


def select_party(row):
    state = clean_text(row.get("state", ""))
    dem = clean_text(row.get("dem_candidate", ""))
    gop = clean_text(row.get("gop_candidate", ""))

    print()
    print(f"Selected race: {state}")
    print(f"  D: {dem or '(blank)'}")
    print(f"  R: {gop or '(blank)'}")

    while True:
        party = input("Update liability for which candidate? Enter D or R: ").strip().upper()

        if party in ["D", "DEM", "DEMOCRAT", "DEMOCRATIC"]:
            return "D"

        if party in ["R", "GOP", "REP", "REPUBLICAN"]:
            return "R"

        print("Please enter D or R.")


def prompt_liability(current):
    print(LIABILITY_SCALE)
    print(f"Current liability score: {current:.2f}")

    while True:
        value = input("Enter new liability score, usually 0.0 to 2.0: ").strip()

        if value == "":
            print("No value entered.")
            continue

        try:
            score = float(value)
        except Exception:
            print("Please enter a number, such as 0, 0.5, 1.0, 1.5, or 2.0.")
            continue

        if score < 0:
            print("Use positive numbers only. The script handles party direction.")
            continue

        if score > 4:
            confirm = input(
                f"{score:.1f} is very high. Type YES to confirm: "
            ).strip().upper()
            if confirm != "YES":
                print("Not confirmed. Enter another score.")
                continue

        return score


def prompt_note(existing_note, state, party, score):
    print()
    print("Optional note for candidate-quality audit.")
    print("This will be appended to candidate_quality_notes_source if that column exists,")
    print("or stored in candidate_quality_manual_note if not.")
    print()

    note = input("Enter note, or press Enter to skip: ").strip()

    if note == "":
        return existing_note

    prefix = f"{state} {party} liability updated to {score:.2f}: {note}"

    if existing_note:
        return existing_note + " | " + prefix

    return prefix


def run_candidate_quality_update():
    script = Path("update_candidate_quality.py")

    if not script.exists():
        print()
        print("WARNING: update_candidate_quality.py not found; saved liability but did not recalculate CQ.")
        return

    print()
    print("$", sys.executable, "update_candidate_quality.py")
    proc = subprocess.run([sys.executable, "update_candidate_quality.py"])

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

    for col, default in [
        ("dem_candidate", ""),
        ("gop_candidate", ""),
        ("dem_candidate_liability", 0.0),
        ("gop_candidate_liability", 0.0),
        ("candidate_quality_adjustment_dem", 0.0),
        ("manual_candidate_quality_adjustment_dem", 0.0),
        ("candidate_quality_manual_note", ""),
    ]:
        df = ensure_column(df, col, default)

    df["dem_candidate_liability"] = pd.to_numeric(
        df["dem_candidate_liability"],
        errors="coerce",
    ).fillna(0.0)

    df["gop_candidate_liability"] = pd.to_numeric(
        df["gop_candidate_liability"],
        errors="coerce",
    ).fillna(0.0)

    display_races(df)

    idx = select_race(df)
    row = df.loc[idx].copy()

    party = select_party(row)

    if party == "D":
        field = "dem_candidate_liability"
        candidate = clean_text(row.get("dem_candidate", ""))
    else:
        field = "gop_candidate_liability"
        candidate = clean_text(row.get("gop_candidate", ""))

    state = clean_text(row.get("state", ""))
    current = as_float(row.get(field, 0.0))

    print()
    print(f"Updating {state} {party} candidate liability: {candidate or '(blank candidate)'}")

    new_score = prompt_liability(current)

    print()
    print("Proposed update:")
    print(f"  Race: {state}")
    print(f"  Candidate side: {party}")
    print(f"  Candidate: {candidate or '(blank candidate)'}")
    print(f"  Field: {field}")
    print(f"  Old score: {current:.2f}")
    print(f"  New score: {new_score:.2f}")

    confirm = input("\nSave this update? Type YES to save: ").strip().upper()

    if confirm != "YES":
        print("No changes saved.")
        return

    df.loc[idx, field] = new_score

    note_col = "candidate_quality_manual_note"
    existing_note = clean_text(df.loc[idx, note_col])
    df.loc[idx, note_col] = prompt_note(existing_note, state, party, new_score)

    df.to_csv(RACE_INPUTS, index=False)

    print()
    print(f"Saved update to {RACE_INPUTS}")

    run_candidate_quality_update()

    updated = pd.read_csv(RACE_INPUTS)
    updated["state"] = updated["state"].astype(str).str.strip().str.upper()
    updated_row = updated[updated["state"] == state].iloc[0]

    print()
    print("Updated candidate-quality fields")
    print("--------------------------------")
    fields = [
        "state",
        "dem_candidate",
        "gop_candidate",
        "dem_nominee_confirmed",
        "gop_nominee_confirmed",
        "dem_candidate_liability",
        "gop_candidate_liability",
        "candidate_liability_adjustment_dem",
        "manual_candidate_quality_adjustment_dem",
        "objective_candidate_quality_adjustment_dem",
        "candidate_quality_gate",
        "candidate_quality_adjustment_dem",
        "candidate_quality_manual_note",
        "candidate_quality_notes",
    ]

    for f in fields:
        if f in updated.columns:
            print(f"{f}: {updated_row.get(f)}")

    print()
    print("Next recommended command:")
    print("  python3 run_full_pipeline.py")


if __name__ == "__main__":
    main()
