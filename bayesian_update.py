
import argparse
from datetime import date
from senate_model.bayesian_updater import run_bayesian_update


def compute_days_out_today():
    election_day = date(2026, 11, 3)
    return max(0, (election_day - date.today()).days)


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian-style polling/fundamentals updater.")
    parser.add_argument("--input-dir", default="inputs")
    parser.add_argument("--days-out", type=int, default=None, help="Optional days until Election Day. If omitted, computed automatically.")
    parser.add_argument("--prior-sd-multiplier", type=float, default=1.15)
    parser.add_argument("--min-polling-sd", type=float, default=2.0)
    parser.add_argument("--sparse-poll-penalty", type=float, default=1.25)
    args = parser.parse_args()

    out = run_bayesian_update(
        input_dir=args.input_dir,
        as_of_days_out=args.days_out if args.days_out is not None else compute_days_out_today(),
        prior_sd_multiplier=args.prior_sd_multiplier,
        min_polling_sd=args.min_polling_sd,
        sparse_poll_penalty=args.sparse_poll_penalty,
    )

    print("\nBayesian update generated:")
    print(out.to_string(index=False))
    print("\nUpdated race_inputs.csv using posterior margins.")
    print("Audit file: inputs/bayesian_update_generated.csv")


if __name__ == "__main__":
    main()
