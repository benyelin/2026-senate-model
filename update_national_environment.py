from pathlib import Path
from datetime import date
import argparse
import pandas as pd

INPUTS = Path("inputs")
NATIONAL_ENV_PATH = INPUTS / "national_environment.csv"


def clamp(value, low, high):
    return max(low, min(high, value))


def approval_adjustment_for_republican_president(approval):
    """
    Converts presidential approval into a modest Democratic adjustment.

    Assumption:
      - President is Republican.
      - Approval below 45 helps Democrats.
      - Approval above 45 helps Republicans.
      - Adjustment is capped at +/- 3 points.

    Example:
      approval = 39.6
      (45 - 39.6) / 3 = +1.8 Democratic adjustment
    """
    return clamp((45.0 - approval) / 3.0, -3.0, 3.0)


def main():
    parser = argparse.ArgumentParser(
        description="Update national environment inputs for the Senate model."
    )

    parser.add_argument(
        "--as-of-date",
        default=date.today().isoformat(),
        help="Date for this national environment update, YYYY-MM-DD."
    )

    parser.add_argument(
        "--generic-ballot",
        type=float,
        required=True,
        help="Generic ballot margin, Democratic minus Republican."
    )

    parser.add_argument(
        "--approval",
        type=float,
        required=True,
        help="Presidential approval percentage."
    )

    parser.add_argument(
        "--disapproval",
        type=float,
        required=True,
        help="Presidential disapproval percentage. Net approval is calculated automatically as approval minus disapproval."
    )

    parser.add_argument(
        "--president-party",
        default="R",
        choices=["R", "D"],
        help="President's party. Current model assumes R unless changed."
    )

    parser.add_argument(
        "--midterm",
        type=float,
        default=1.0,
        help="Additional midterm adjustment in Democratic-margin terms."
    )

    parser.add_argument(
        "--approval-adjustment",
        type=float,
        default=None,
        help="Optional manual approval adjustment. If omitted, calculated automatically."
    )

    parser.add_argument(
        "--notes",
        default="Manual national environment update.",
        help="Source notes for dashboard/audit trail."
    )

    args = parser.parse_args()

    INPUTS.mkdir(parents=True, exist_ok=True)

    net_approval = args.approval - args.disapproval

    if args.approval_adjustment is not None:
        approval_adjustment_dem = args.approval_adjustment
    else:
        if args.president_party == "R":
            approval_adjustment_dem = approval_adjustment_for_republican_president(
                args.approval
            )
        else:
            # If president is Democratic, low approval would hurt Democrats.
            approval_adjustment_dem = -approval_adjustment_for_republican_president(
                args.approval
            )

    # Reduced double-count formula selected after backtesting.
    # Rationale: the generic ballot already captures much of the national mood,
    # presidential approval, and midterm environment. This formula keeps those
    # signals but avoids fully double-counting them.
    national_environment_margin_dem = (
        0.85 * args.generic_ballot
        + 0.50 * approval_adjustment_dem
        + 0.50 * args.midterm
    )

    row = {
        "as_of_date": args.as_of_date,
        "generic_ballot_margin_dem": args.generic_ballot,
        "presidential_approval": args.approval,
        "presidential_disapproval": args.disapproval,
        "presidential_net_approval": net_approval,
        "president_party": args.president_party,
        "midterm_adjustment_dem": args.midterm,
        "approval_adjustment_dem": approval_adjustment_dem,
        "national_environment_margin_dem": national_environment_margin_dem,
        "source_notes": args.notes,
    }

    df = pd.DataFrame([row])
    df.to_csv(NATIONAL_ENV_PATH, index=False)

    print(f"Updated {NATIONAL_ENV_PATH}")
    print()
    print("National environment calculation:")
    print("  Formula: 0.85*generic + 0.50*approval_adjustment + 0.50*midterm")
    print(f"  Generic ballot margin Dem:     {args.generic_ballot:+.2f} x 0.85 = {0.85 * args.generic_ballot:+.2f}")
    print(f"  Approval adjustment Dem:       {approval_adjustment_dem:+.2f} x 0.50 = {0.50 * approval_adjustment_dem:+.2f}")
    print(f"  Midterm adjustment Dem:        {args.midterm:+.2f} x 0.50 = {0.50 * args.midterm:+.2f}")
    print(f"  National environment Dem:      {national_environment_margin_dem:+.2f}")

    if net_approval is not None:
        print(f"  Presidential net approval:     {net_approval:+.2f}")


if __name__ == "__main__":
    main()
