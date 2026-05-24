# M2 Poll Ingestion Pipeline

This step adds a semi-automated polling ingestion layer.

## Workflow

1. Add individual polls to `inputs/polls_raw.csv`.
2. Run:

   ```bash
   python ingest_polls.py --as-of 2026-05-22
   ```

3. This creates:
   - `inputs/polling_averages_generated.csv`
   - updated `inputs/race_inputs.csv`

4. Run the forecast:

   ```bash
   python run_model.py --today 2026-05-22
   ```

## Required columns in `polls_raw.csv`

- `state`
- `pollster`
- `start_date`
- `end_date`
- `dem_candidate`
- `gop_candidate`
- `dem_share`
- `gop_share`

## Optional columns

- `sample_size`
- `pollster_rating`
- `mode`
- `sponsor`
- `notes`

## Weighting logic

The polling average uses:

- recency weighting
- sample-size weighting
- pollster quality weighting
- mode weighting
- sponsor penalty

It is deliberately transparent and simple enough to audit.