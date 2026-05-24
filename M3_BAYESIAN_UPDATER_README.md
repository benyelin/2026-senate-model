# M3 Bayesian-Style Polling/Fundamentals Updater

M3 adds a transparent Bayesian-style update layer.

## Why this matters

Earlier versions used a fixed polling/fundamentals blend. M3 instead lets polling
gain influence when it is more informative:

- more polls
- fresher polls
- higher total poll weight
- closer to Election Day

Sparse or stale polling remains noisy and therefore gets less weight.

## Workflow

```bash
python ingest_polls.py --as-of 2026-05-22
python bayesian_update.py --days-out 165
python run_model.py --today 2026-05-22
```

## Outputs

- `inputs/polling_averages_generated.csv`
- `inputs/bayesian_update_generated.csv`
- updated `inputs/race_inputs.csv`
- normal forecast outputs in `outputs/`

## Modeling note

For compatibility with the existing simulation engine, M3 writes the posterior
margin into `fundamentals_margin_dem`. This means the simulator treats the
Bayesian posterior as the central race estimate before adding national
environment, uncertainty, and correlated errors.

This is intentionally simple and auditable. A future M4 version could make the
engine natively understand separate prior/poll/posterior objects.