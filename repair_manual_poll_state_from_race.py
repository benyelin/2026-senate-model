from pathlib import Path
import pandas as pd
import re

MANUAL_POLLS = Path("inputs/manual_polls.csv")

STATE_CODES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY"
}

STATE_NAMES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY"
}


def infer_state_from_race(race):
    text = str(race).strip().upper()

    # Examples: "OH Senate", "ME Senate"
    m = re.match(r"^([A-Z]{2})\b", text)
    if m and m.group(1) in STATE_CODES:
        return m.group(1)

    # Examples: "Ohio Senate"
    for name, code in STATE_NAMES.items():
        if text.startswith(name + " "):
            return code

    return None


def main():
    if not MANUAL_POLLS.exists():
        print("No inputs/manual_polls.csv found; skipping.")
        return

    df = pd.read_csv(MANUAL_POLLS)

    if df.empty or "race" not in df.columns or "state" not in df.columns:
        print("manual_polls.csv missing race/state columns or empty; skipping.")
        return

    fixes = []

    for idx, row in df.iterrows():
        inferred = infer_state_from_race(row.get("race", ""))
        current = str(row.get("state", "")).strip().upper()

        if inferred and current != inferred:
            fixes.append((idx, row.get("race", ""), current, inferred))
            df.at[idx, "state"] = inferred

    df.to_csv(MANUAL_POLLS, index=False)

    if fixes:
        print("Fixed race/state mismatches in manual polls:")
        for _, race, old, new in fixes:
            print(f"  {race}: {old} -> {new}")
    else:
        print("No manual poll race/state mismatches found.")


if __name__ == "__main__":
    main()
