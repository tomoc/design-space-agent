from __future__ import annotations

import pandas as pd

from dslab.audit.constraints import add_constraint_columns, parse_constraint_expression
from dslab.optimize.acquisition import feasibility_scores, predicted_constraint_satisfaction
from dslab.schema.design_schema import ConstraintSpec, DesignSpec


def test_constraint_parser_supports_all_basic_operators() -> None:
    operators = ["<=", ">=", "<", ">", "=="]
    for op in operators:
        rule = parse_constraint_expression("c", f"value {op} 1.5")
        assert rule.column == "value"
        assert rule.operator == op
        assert rule.threshold == 1.5


def test_feasible_classification() -> None:
    df = pd.DataFrame(
        {
            "temperature": [40.0, 48.0, 42.0],
            "gradeability": [0.32, 0.31, 0.25],
        }
    )
    constraints = [
        ConstraintSpec(name="temp", expression="temperature <= 45.0"),
        ConstraintSpec(name="grade", expression="gradeability >= 0.30"),
    ]

    evaluated = add_constraint_columns(df, constraints)

    assert evaluated["constraint__temp"].tolist() == [True, False, True]
    assert evaluated["constraint__grade"].tolist() == [True, True, False]
    assert evaluated["feasible"].tolist() == [True, False, False]


def test_feasible_classification_respects_strict_and_equality_operators() -> None:
    df = pd.DataFrame(
        {
            "lt_value": [0.9, 1.0],
            "gt_value": [1.1, 1.0],
            "eq_value": [1.0, 1.1],
        }
    )
    constraints = [
        ConstraintSpec(name="lt", expression="lt_value < 1.0"),
        ConstraintSpec(name="gt", expression="gt_value > 1.0"),
        ConstraintSpec(name="eq", expression="eq_value == 1.0"),
    ]

    evaluated = add_constraint_columns(df, constraints)

    assert evaluated["constraint__lt"].tolist() == [True, False]
    assert evaluated["constraint__gt"].tolist() == [True, False]
    assert evaluated["constraint__eq"].tolist() == [True, False]
    assert evaluated["feasible"].tolist() == [True, False]


def test_predicted_feasible_matches_constraint_expressions() -> None:
    spec = DesignSpec.model_validate(
        {
            "project": {"name": "predicted_constraint_demo"},
            "data": {"path": "samples.csv"},
            "variables": {"x": {"type": "continuous", "range": [0, 1]}},
            "objectives": [{"name": "y", "direction": "maximize"}],
            "constraints": [
                {"name": "lt", "expression": "lt_value < 1.0"},
                {"name": "gt", "expression": "gt_value > 1.0"},
                {"name": "eq", "expression": "eq_value == 1.0"},
            ],
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_lt_value": [0.9, 1.0],
            "pred_gt_value": [1.1, 1.0],
            "pred_eq_value": [1.0, 1.1],
        }
    )

    satisfaction = predicted_constraint_satisfaction(spec, predictions)
    _, labels = feasibility_scores(spec, predictions)

    assert satisfaction["predicted_feasible"].tolist() == [True, False]
    assert labels[0] == "All predicted constraints are satisfied."
    assert "main predicted constraint risk" in labels[1]
