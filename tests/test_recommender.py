from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dslab.audit.data_quality import compute_audit
from dslab.optimize.candidate_generator import generate_candidates
from dslab.optimize.recommender import recommend_candidates
from dslab.schema.design_schema import DesignSpec
from dslab.schema.parser import load_dataset, load_design_problem
from dslab.trust.trust_score import compute_trust_scores

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_generation_is_seed_stable() -> None:
    problem = load_design_problem(ROOT / "examples" / "vehicle_concept" / "design.yaml")

    first = generate_candidates(problem.spec, n=10, seed=123)
    second = generate_candidates(problem.spec, n=10, seed=123)

    assert first.equals(second)


def test_recommendation_returns_requested_rows() -> None:
    problem = load_design_problem(ROOT / "examples" / "vehicle_concept" / "design.yaml")
    rec = problem.spec.recommendation.model_copy(update={"candidate_pool_size": 60, "random_seed": 11})
    spec = problem.spec.model_copy(update={"recommendation": rec})
    df = load_dataset(problem).head(50)
    _, evaluated = compute_audit(spec, df)

    result = recommend_candidates(spec, evaluated, n=4)

    assert len(result.recommendations) == 4
    assert result.recommendations["candidate_rank"].tolist() == [1, 2, 3, 4]
    assert {"recommendation_score", "trust_score", "explanation"}.issubset(result.recommendations.columns)


def test_recommendation_score_is_weighted_average_and_bounded() -> None:
    problem = load_design_problem(ROOT / "examples" / "vehicle_concept" / "design.yaml")
    rec = problem.spec.recommendation.model_copy(update={"candidate_pool_size": 80, "random_seed": 17})
    spec = problem.spec.model_copy(update={"recommendation": rec})
    df = load_dataset(problem).head(60)
    _, evaluated = compute_audit(spec, df)

    result = recommend_candidates(spec, evaluated, n=5)
    rows = result.recommendations
    total_weight = rec.objective_weight + rec.feasibility_weight + rec.trust_penalty_weight
    expected = (
        rec.objective_weight * rows["objective_score"]
        + rec.feasibility_weight * rows["feasibility_score"]
        + rec.trust_penalty_weight * rows["trust_score"]
    ) / total_weight

    assert rows["recommendation_score"].between(0.0, 1.0).all()
    assert rows["trust_score"].between(0.0, 1.0).all()
    assert rows["recommendation_score"].is_monotonic_decreasing
    assert rows["predicted_feasible"].isin([True, False]).all()
    assert not rows["explanation"].str.contains(r"\.\.").any()
    np.testing.assert_allclose(rows["recommendation_score"], expected)

    infeasible_rows = rows[~rows["predicted_feasible"]]
    assert not infeasible_rows["explanation"].str.contains("predicted feasible").any()


def test_trust_scores_are_clipped_to_unit_interval() -> None:
    spec = DesignSpec.model_validate(
        {
            "project": {"name": "trust_demo"},
            "data": {"path": "samples.csv"},
            "variables": {
                "x": {"type": "continuous", "range": [0, 1]},
                "mode": {"type": "categorical", "values": ["a", "b"]},
            },
            "objectives": [{"name": "y", "direction": "maximize"}],
        }
    )
    train_df = pd.DataFrame(
        {
            "x": [0.0, 0.5, 1.0],
            "mode": ["a", "a", "b"],
            "y": [1.0, 2.0, 3.0],
        }
    )
    candidates = pd.DataFrame({"x": [0.25, 99.0], "mode": ["a", "unseen"]})

    scores = compute_trust_scores(spec, train_df, candidates)

    assert scores.between(0.0, 1.0).all()
