from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dslab.models.surrogate import SurrogateResult, predict_surrogates, train_surrogates
from dslab.optimize.acquisition import (
    feasibility_scores,
    objective_scores,
    predicted_constraint_satisfaction,
)
from dslab.optimize.candidate_generator import generate_candidates
from dslab.schema.design_schema import DesignSpec
from dslab.trust.trust_score import compute_trust_scores


@dataclass
class RecommendationResult:
    recommendations: pd.DataFrame
    candidate_pool: pd.DataFrame
    surrogate: SurrogateResult


def recommend_candidates(spec: DesignSpec, df: pd.DataFrame, n: int = 5) -> RecommendationResult:
    """Train surrogates, score a candidate pool, and return top recommendations."""
    surrogate = train_surrogates(spec, df)
    candidates = generate_candidates(spec)
    scored = predict_surrogates(surrogate, candidates)
    scored["objective_score"] = objective_scores(spec, scored)
    scored["feasibility_score"], risk_text = feasibility_scores(spec, scored)
    scored["predicted_feasible"] = predicted_constraint_satisfaction(spec, scored)[
        "predicted_feasible"
    ]
    scored["trust_score"] = compute_trust_scores(spec, df, candidates)

    rec = spec.recommendation
    total_weight = rec.objective_weight + rec.feasibility_weight + rec.trust_penalty_weight
    scored["recommendation_score"] = (
        rec.objective_weight * scored["objective_score"]
        + rec.feasibility_weight * scored["feasibility_score"]
        + rec.trust_penalty_weight * scored["trust_score"]
    ).div(total_weight).clip(0.0, 1.0)
    scored["explanation"] = [
        _explain(row, risk)
        for (_, row), risk in zip(scored.iterrows(), risk_text, strict=False)
    ]

    ordered = scored.sort_values("recommendation_score", ascending=False).head(n).copy()
    ordered.insert(0, "candidate_rank", range(1, len(ordered) + 1))

    ordered = _order_recommendation_columns(spec, surrogate, ordered)
    return RecommendationResult(recommendations=ordered, candidate_pool=scored, surrogate=surrogate)


def _order_recommendation_columns(
    spec: DesignSpec,
    surrogate: SurrogateResult,
    df: pd.DataFrame,
) -> pd.DataFrame:
    base = [
        "candidate_rank",
        "recommendation_score",
        "objective_score",
        "feasibility_score",
        "predicted_feasible",
        "trust_score",
    ]
    variables = spec.variable_names
    predictions = [f"pred_{target}" for target in surrogate.target_columns]
    tail = ["explanation"]
    return df[base + variables + predictions + tail]


def _explain(row: pd.Series, risk_text: str) -> str:
    parts = []
    if row["objective_score"] >= 0.70:
        parts.append("High objective score")
    else:
        parts.append("Moderate objective trade-off")
    if row["trust_score"] >= 0.75:
        parts.append("low extrapolation risk")
    elif row["trust_score"] >= 0.50:
        parts.append("moderate extrapolation risk")
    else:
        parts.append("high extrapolation risk")
    if bool(row["predicted_feasible"]):
        parts.append("predicted feasible")
    else:
        parts.append(risk_text.rstrip("."))
    return ". ".join(parts) + "."
