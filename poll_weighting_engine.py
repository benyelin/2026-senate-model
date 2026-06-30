from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


GRADE_WEIGHTS = {
    "A+": 1.20,
    "A": 1.20,
    "A-": 1.10,
    "B+": 1.10,
    "B": 1.00,
    "B-": 1.00,
    "C+": 0.85,
    "C": 0.85,
    "C-": 0.70,
    "D+": 0.70,
    "D": 0.70,
    "D-": 0.65,
    "F": 0.55,
}

POPULATION_WEIGHTS = {
    "LV": 1.10,
    "LIKELY VOTERS": 1.10,
    "LIKELY VOTER": 1.10,
    "RV": 1.00,
    "REGISTERED VOTERS": 1.00,
    "REGISTERED VOTER": 1.00,
    "A": 0.75,
    "ADULTS": 0.75,
    "ADULT": 0.75,
}

NEUTRAL_SPONSOR_TYPES = {
    "",
    "NEUTRAL",
    "NONPARTISAN",
    "NON-PARTISAN",
    "MEDIA",
    "UNIVERSITY",
    "ACADEMIC",
}

PARTISAN_SPONSOR_TYPES = {
    "PARTISAN",
    "PARTY",
    "PARTY-ALIGNED",
    "PARTY ALIGNED",
    "ADVOCACY",
}

INTERNAL_SPONSOR_TYPES = {
    "INTERNAL",
    "CAMPAIGN",
    "CANDIDATE",
    "CAMPAIGN INTERNAL",
    "CANDIDATE INTERNAL",
}


@dataclass(frozen=True)
class PollWeightingSettings:
    unknown_grade_weight: float = 0.80
    unknown_population_weight: float = 0.85
    unknown_sponsor_weight: float = 0.85

    partisan_sponsor_weight: float = 0.80
    internal_poll_weight: float = 0.65

    partisan_margin_adjustment: float = 1.00
    internal_margin_adjustment: float = 1.50

    sample_reference_n: float = 600.0
    sample_weight_min: float = 0.65
    sample_weight_max: float = 1.35

    environment_translation_min_age: int = 21
    environment_translation_cap: float = 3.00

    pollster_concentration_cap: float = 0.35


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_upper(value) -> str:
    return normalize_text(value).upper()


def numeric(value, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return float(parsed)


def numeric_series(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """Return a numeric Series even when the requested column is absent."""
    if column in df.columns:
        return pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(default)

    return pd.Series(
        default,
        index=df.index,
        dtype=float,
    )


def recency_half_life(days_out: float) -> float:
    if days_out > 150:
        return 45.0
    if days_out > 90:
        return 35.0
    if days_out > 45:
        return 25.0
    if days_out > 15:
        return 14.0
    return 7.0


def calculate_recency_weight(age_days: float, days_out: float) -> float:
    age = max(0.0, numeric(age_days, 0.0))
    half_life = recency_half_life(days_out)
    return float(0.5 ** (age / half_life))


def calculate_grade_weight(
    grade,
    settings: PollWeightingSettings,
) -> float:
    normalized = normalize_upper(grade)
    return GRADE_WEIGHTS.get(normalized, settings.unknown_grade_weight)


def calculate_sample_size_weight(
    sample_size,
    settings: PollWeightingSettings,
) -> float:
    n = numeric(sample_size, settings.sample_reference_n)

    if n <= 0:
        return settings.sample_weight_min

    weight = math.sqrt(n / settings.sample_reference_n)

    return float(
        min(
            settings.sample_weight_max,
            max(settings.sample_weight_min, weight),
        )
    )


def calculate_population_weight(
    sample_type,
    settings: PollWeightingSettings,
) -> float:
    normalized = normalize_upper(sample_type)
    return POPULATION_WEIGHTS.get(
        normalized,
        settings.unknown_population_weight,
    )


def classify_sponsor_type(row: pd.Series) -> str:
    sponsor_type = normalize_upper(row.get("poll_sponsor_type", ""))
    internal = normalize_upper(row.get("is_internal_poll", ""))
    sponsor_party = normalize_upper(row.get("partisan_sponsor_party", ""))

    if internal in {"TRUE", "1", "YES", "Y"}:
        return "internal"

    if sponsor_type in INTERNAL_SPONSOR_TYPES:
        return "internal"

    if sponsor_type in PARTISAN_SPONSOR_TYPES:
        return "partisan"

    if sponsor_party in {"D", "DEM", "DEMOCRATIC", "R", "REP", "REPUBLICAN"}:
        return "partisan"

    if sponsor_type in NEUTRAL_SPONSOR_TYPES:
        return "neutral"

    return "unknown"


def sponsor_party_direction(row: pd.Series) -> Optional[str]:
    party = normalize_upper(row.get("partisan_sponsor_party", ""))

    if party in {"D", "DEM", "DEMOCRATIC"}:
        return "D"

    if party in {"R", "REP", "REPUBLICAN", "GOP"}:
        return "R"

    affiliation = normalize_upper(
        row.get("pollster_partisan_affiliation", "")
    )

    if affiliation in {"D", "DEM", "DEMOCRATIC"}:
        return "D"

    if affiliation in {"R", "REP", "REPUBLICAN", "GOP"}:
        return "R"

    return None


def calculate_sponsor_weight(
    row: pd.Series,
    settings: PollWeightingSettings,
) -> float:
    classification = classify_sponsor_type(row)

    if classification == "internal":
        return settings.internal_poll_weight

    if classification == "partisan":
        return settings.partisan_sponsor_weight

    if classification == "neutral":
        return 1.0

    return settings.unknown_sponsor_weight


def calculate_sponsor_margin_adjustment_dem(
    row: pd.Series,
    settings: PollWeightingSettings,
) -> float:
    classification = classify_sponsor_type(row)
    party = sponsor_party_direction(row)

    if party is None:
        return 0.0

    magnitude = 0.0

    if classification == "internal":
        magnitude = settings.internal_margin_adjustment
    elif classification == "partisan":
        magnitude = settings.partisan_margin_adjustment

    # Correct against the sponsoring party.
    # Democratic sponsor => move margin Republicanward.
    # Republican sponsor => move margin Democraticward.
    if party == "D":
        return -magnitude

    if party == "R":
        return magnitude

    return 0.0


def calculate_environment_translation_dem(
    row: pd.Series,
    settings: PollWeightingSettings,
) -> float:
    age_days = numeric(row.get("poll_age_days", 0.0), 0.0)

    if age_days < settings.environment_translation_min_age:
        return 0.0

    current_environment = numeric(
        row.get("current_national_environment_dem", np.nan),
        np.nan,
    )
    poll_date_environment = numeric(
        row.get("national_environment_at_poll_date_dem", np.nan),
        np.nan,
    )

    if np.isnan(current_environment) or np.isnan(poll_date_environment):
        return 0.0

    elasticity = numeric(
        row.get(
            "race_environment_sensitivity",
            row.get("state_elasticity", row.get("district_elasticity", 1.0)),
        ),
        1.0,
    )

    raw_translation = (
        current_environment - poll_date_environment
    ) * elasticity

    return float(
        np.clip(
            raw_translation,
            -settings.environment_translation_cap,
            settings.environment_translation_cap,
        )
    )


def apply_pollster_concentration_cap(
    group: pd.DataFrame,
    pollster_col: str,
    weight_col: str,
    cap: float,
) -> pd.Series:
    """
    Iteratively scale pollster totals so no pollster exceeds cap of race weight.

    Returns capped weights indexed like group.
    """
    weights = pd.to_numeric(
        group[weight_col],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    pollsters = (
        group[pollster_col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    result = weights.copy()

    for _ in range(20):
        total = result.sum()

        if total <= 0:
            break

        pollster_totals = result.groupby(pollsters).sum()
        shares = pollster_totals / total
        offenders = shares[shares > cap + 1e-10]

        if offenders.empty:
            break

        changed = False

        for pollster, share in offenders.items():
            mask = pollsters.eq(pollster)
            current_total = result.loc[mask].sum()
            other_total = total - current_total

            if other_total <= 0:
                continue

            allowed_total = cap * other_total / (1.0 - cap)

            if current_total > allowed_total:
                scale = allowed_total / current_total
                result.loc[mask] *= scale
                changed = True

        if not changed:
            break

    return result


def calculate_effective_poll_count(weights: pd.Series) -> float:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    denominator = float((w ** 2).sum())

    if denominator <= 0:
        return 0.0

    return float((w.sum() ** 2) / denominator)


def deduplicate_polls(polls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove exact or near-exact duplicate poll records before weighting.

    Polls are considered duplicates when the available identifying fields
    describe the same poll. The first occurrence is retained and all removed
    rows are returned in a separate audit dataframe.
    """
    out = polls.copy()

    candidate_fields = [
        "state",
        "district_id",
        "pollster",
        "sponsor",
        "start_date",
        "end_date",
        "sample_size",
        "sample_type",
        "dem_candidate",
        "rep_candidate",
        "dem_pct",
        "rep_pct",
        "reported_margin_dem",
    ]

    key_fields = [c for c in candidate_fields if c in out.columns]

    if not key_fields:
        return out, pd.DataFrame(columns=out.columns)

    normalized = pd.DataFrame(index=out.index)

    for col in key_fields:
        if col in {"sample_size", "dem_pct", "rep_pct", "reported_margin_dem"}:
            normalized[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            ).round(3)
        else:
            normalized[col] = (
                out[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

    duplicate_mask = normalized.duplicated(
        subset=key_fields,
        keep="first",
    )

    duplicates = out.loc[duplicate_mask].copy()
    duplicates["duplicate_reason"] = "Exact duplicate poll record"

    deduplicated = out.loc[~duplicate_mask].copy()

    return deduplicated, duplicates


def weight_polls(
    polls: pd.DataFrame,
    *,
    race_col: str,
    pollster_col: str = "pollster",
    days_out: float,
    settings: Optional[PollWeightingSettings] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = settings or PollWeightingSettings()

    # Poll counts and effective sample sizes must represent distinct polls.
    out, duplicate_polls = deduplicate_polls(polls)

    if "reported_margin_dem" not in out.columns:
        if {"dem_pct", "rep_pct"}.issubset(out.columns):
            out["reported_margin_dem"] = (
                pd.to_numeric(out["dem_pct"], errors="coerce")
                - pd.to_numeric(out["rep_pct"], errors="coerce")
            )
        else:
            raise ValueError(
                "Poll data must contain reported_margin_dem or dem_pct and rep_pct."
            )

    out["poll_age_days"] = pd.to_numeric(
        out.get("poll_age_days", 0.0),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    out["recency_half_life_days"] = recency_half_life(days_out)

    out["recency_weight"] = out["poll_age_days"].apply(
        lambda age: calculate_recency_weight(age, days_out)
    )

    out["pollster_grade_weight"] = out.get(
        "pollster_grade",
        pd.Series("", index=out.index),
    ).apply(lambda grade: calculate_grade_weight(grade, settings))

    out["sample_size_weight"] = out.get(
        "sample_size",
        pd.Series(settings.sample_reference_n, index=out.index),
    ).apply(lambda n: calculate_sample_size_weight(n, settings))

    out["population_weight"] = out.get(
        "sample_type",
        pd.Series("", index=out.index),
    ).apply(lambda s: calculate_population_weight(s, settings))

    out["sponsor_classification"] = out.apply(
        classify_sponsor_type,
        axis=1,
    )

    out["sponsor_weight"] = out.apply(
        lambda row: calculate_sponsor_weight(row, settings),
        axis=1,
    )

    out["partisan_sponsor_adjustment_dem"] = out.apply(
        lambda row: calculate_sponsor_margin_adjustment_dem(row, settings),
        axis=1,
    )

    if "pollster_house_effect_dem" in out.columns:
        out["pollster_house_effect_dem"] = pd.to_numeric(
            out["pollster_house_effect_dem"],
            errors="coerce",
        ).fillna(0.0)
    else:
        out["pollster_house_effect_dem"] = pd.Series(
            0.0,
            index=out.index,
            dtype=float,
        )

    out["environment_translation_dem"] = out.apply(
        lambda row: calculate_environment_translation_dem(row, settings),
        axis=1,
    )

    out["adjusted_margin_dem"] = (
        pd.to_numeric(out["reported_margin_dem"], errors="coerce")
        + out["partisan_sponsor_adjustment_dem"]
        - out["pollster_house_effect_dem"]
        + out["environment_translation_dem"]
    )

    out["preliminary_poll_weight"] = (
        out["recency_weight"]
        * out["pollster_grade_weight"]
        * out["sample_size_weight"]
        * out["population_weight"]
        * out["sponsor_weight"]
    )

    out["final_poll_weight"] = out["preliminary_poll_weight"]

    for _, indices in out.groupby(race_col).groups.items():
        group = out.loc[indices]

        out.loc[indices, "final_poll_weight"] = (
            apply_pollster_concentration_cap(
                group,
                pollster_col=pollster_col,
                weight_col="preliminary_poll_weight",
                cap=settings.pollster_concentration_cap,
            )
        )

    out["weighted_margin_component"] = (
        out["adjusted_margin_dem"] * out["final_poll_weight"]
    )

    race_rows = []

    for race, group in out.groupby(race_col, dropna=False):
        total_weight = group["final_poll_weight"].sum()

        if total_weight > 0:
            average = (
                group["weighted_margin_component"].sum()
                / total_weight
            )
        else:
            average = np.nan

        pollster_totals = (
            group.groupby(pollster_col)["final_poll_weight"]
            .sum()
            .sort_values(ascending=False)
        )

        largest_pollster_share = (
            float(pollster_totals.iloc[0] / total_weight)
            if total_weight > 0 and not pollster_totals.empty
            else 0.0
        )

        latest_age = (
            float(group["poll_age_days"].min())
            if not group.empty
            else np.nan
        )

        weighted_average_age = (
            float(
                np.average(
                    group["poll_age_days"],
                    weights=group["final_poll_weight"],
                )
            )
            if total_weight > 0
            else np.nan
        )

        race_rows.append(
            {
                race_col: race,
                "polling_margin_dem": average,
                "poll_count": int(len(group)),
                "effective_poll_count": calculate_effective_poll_count(
                    group["final_poll_weight"]
                ),
                "total_poll_weight": float(total_weight),
                "latest_poll_age_days": latest_age,
                "weighted_avg_poll_age_days": weighted_average_age,
                "largest_pollster_weight_share": largest_pollster_share,
                "only_partisan_or_internal_polls": bool(
                    group["sponsor_classification"]
                    .isin(["partisan", "internal"])
                    .all()
                ),
                "max_absolute_environment_translation": float(
                    group["environment_translation_dem"].abs().max()
                ),
            }
        )

    race_averages = pd.DataFrame(race_rows)

    return out, race_averages
