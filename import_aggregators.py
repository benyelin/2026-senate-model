
import argparse

from senate_model.aggregator_ingestion import (
    update_national_environment_from_sources,
    update_polling_from_aggregators,
)


def main():
    parser = argparse.ArgumentParser(description="Import polling/national environment data from configured aggregator sources.")
    parser.add_argument("--input-dir", default="inputs")
    parser.add_argument("--national", action="store_true", help="Update national_environment.csv.")
    parser.add_argument("--races", action="store_true", help="Try to update polls_raw.csv from race aggregator pages.")
    parser.add_argument("--source-preference", default="auto")
    parser.add_argument("--presidential-approval", type=float, default=None)
    parser.add_argument("--approval-slope", type=float, default=None)
    parser.add_argument("--midterm-effect-dem", type=float, default=None)

    args = parser.parse_args()

    if not args.national and not args.races:
        args.national = True
        args.races = True

    if args.national:
        audit = update_national_environment_from_sources(
            input_dir=args.input_dir,
            source_preference=args.source_preference,
            presidential_approval=args.presidential_approval,
            approval_slope=args.approval_slope,
            midterm_effect_dem=args.midterm_effect_dem,
        )
        print("\nNational environment import audit:")
        print(audit.to_string(index=False) if not audit.empty else "No audit rows.")

    if args.races:
        race_audit = update_polling_from_aggregators(input_dir=args.input_dir)
        print("\nRace polling import audit:")
        print(race_audit.to_string(index=False) if not race_audit.empty else "No race rows.")


if __name__ == "__main__":
    main()
