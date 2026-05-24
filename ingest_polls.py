
import argparse
from pathlib import Path

from senate_model.poll_ingestion import update_race_inputs_from_polls


def main():
    parser = argparse.ArgumentParser(description="Ingest raw Senate polls and update race_inputs.csv.")
    parser.add_argument("--input-dir", default="inputs")
    parser.add_argument("--as-of", default=None, help="Optional YYYY-MM-DD date for poll aging.")
    parser.add_argument("--half-life-days", type=float, default=21.0)
    args = parser.parse_args()

    avgs = update_race_inputs_from_polls(
        input_dir=args.input_dir,
        as_of_date=args.as_of,
        half_life_days=args.half_life_days,
    )

    print("\nGenerated weighted polling averages:")
    print(avgs.to_string(index=False))
    print(f"\nUpdated {Path(args.input_dir) / 'race_inputs.csv'}")
    print(f"Wrote audit file {Path(args.input_dir) / 'polling_averages_generated.csv'}")


if __name__ == "__main__":
    main()
