# O1: Full Race Universe Expansion

This version expands the model from key races to all 35 Senate elections scheduled for 2026:
33 regular Class II races plus the Florida and Ohio special elections.

## Seat math change

- Default Democratic baseline seats: 34
- This represents Democratic-caucus seats not up in 2026.
- Every 2026 race has `dem_win_counts_for_seat_change = 1`.
- Total Democratic seats = 34 + Democratic wins among modeled 2026 races.

## Race tiers

`race_inputs.csv` now includes:

- `race_tier`
- `tier_error_multiplier`
- `polling_active`

Tiers:
- Competitive
- Likely
- Safe
- UltraSafe

Safe and UltraSafe races are still simulated, but with large margins and reduced race-specific volatility.