from pathlib import Path
import pandas as pd
import numpy as np

BAYES_PATH = Path("inputs/bayesian_update_generated.csv")

# Matches current health-check expected cap at this point in the cycle.
# Later we can replace this with a shared dynamic cap function.
CYCLE_MAX_POLLING_WEIGHT = 0.18

if not BAYES_PATH.exists():
    raise FileNotFoundError("inputs/bayesian_update_generated.csv not found.")

bayes = pd.read_csv(BAYES_PATH)

required = [
    "state",
    "prior_margin_dem",
    "polling_margin_used",
    "bayesian_polling_weight",
]

missing = [c for c in required if c not in bayes.columns]
if missing:
    raise ValueError(f"Missing required Bayesian columns: {missing}")

bayes["state"] = bayes["state"].astype(str).str.strip().str.upper()

bayes["original_bayesian_polling_weight"] = pd.to_numeric(
    bayes["bayesian_polling_weight"],
    errors="coerce"
)

bayes["bayesian_polling_weight_capped"] = bayes["original_bayesian_polling_weight"].clip(
    lower=0,
    upper=CYCLE_MAX_POLLING_WEIGHT,
)

bayes["bayesian_fundamentals_weight_capped"] = 1.0 - bayes["bayesian_polling_weight_capped"]

prior = pd.to_numeric(bayes["prior_margin_dem"], errors="coerce")
poll = pd.to_numeric(bayes["polling_margin_used"], errors="coerce")
w = pd.to_numeric(bayes["bayesian_polling_weight_capped"], errors="coerce").fillna(0)

bayes["bayesian_model_margin_dem_uncapped"] = pd.to_numeric(
    bayes.get("posterior_margin_dem", np.nan),
    errors="coerce",
)

bayes["bayesian_model_margin_dem_capped"] = poll * w + prior * (1.0 - w)
bayes["posterior_margin_dem_capped"] = bayes["bayesian_model_margin_dem_capped"]
bayes["cycle_max_polling_weight"] = CYCLE_MAX_POLLING_WEIGHT
bayes["bayesian_cap_applied"] = bayes["original_bayesian_polling_weight"] > CYCLE_MAX_POLLING_WEIGHT

bayes.to_csv(BAYES_PATH, index=False)

print(f"Applied Senate Bayesian polling cap: {CYCLE_MAX_POLLING_WEIGHT:.3f}")
print()
show_cols = [
    "state",
    "prior_margin_dem",
    "polling_margin_used",
    "original_bayesian_polling_weight",
    "bayesian_polling_weight_capped",
    "bayesian_model_margin_dem_uncapped",
    "bayesian_model_margin_dem_capped",
    "bayesian_cap_applied",
]
print(bayes[show_cols].to_string(index=False))
