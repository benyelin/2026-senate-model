
import argparse
from pathlib import Path

from senate_model.engine import ModelConfig, run_forecast


def main():
    parser = argparse.ArgumentParser(description="Run the 2026 Senate Python forecast model.")
    parser.add_argument("--input-dir", default="inputs")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--today", default=None)
    parser.add_argument("--baseline-seats", type=int, default=34)
    parser.add_argument("--threshold", type=int, default=51)

    args = parser.parse_args()
    config = ModelConfig(
        n_sims=args.sims,
        seed=args.seed,
        today=args.today,
        dem_baseline_seats=args.baseline_seats,
        control_threshold=args.threshold,
    )

    results = run_forecast(args.input_dir, args.output_dir, config)
    s = results["summary"]

    print("\n2026 Senate Forecast — Python M1")
    print("-" * 40)
    print(f"Simulations:              {s['n_sims']:,}")
    print(f"Days out:                 {s['days_out']}")
    print(f"Expected Dem seats:       {s['expected_dem_seats']:.2f}")
    print(f"Median Dem seats:         {s['median_dem_seats']:.0f}")
    print(f"Dem control probability:  {s['dem_control_probability']:.1%}")
    print(f"Total error SD:           {s['total_error_sd']:.2f}")
    print(f"National error SD:        {s['national_error_sd']:.2f}")
    print(f"Race error SD:            {s['race_error_sd']:.2f}")
    print(f"Implied correlation:      {s['implied_correlation']:.1%}")
    print(f"Polling weight:           {s['polling_weight']:.1%}")
    print(f"Fundamentals weight:      {s['fundamentals_weight']:.1%}")
    print(f"\nOutputs saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
