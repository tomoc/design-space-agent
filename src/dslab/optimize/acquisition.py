from __future__ import annotations

import numpy as np
import pandas as pd

from dslab.audit.constraints import ConstraintRule, parse_constraints
from dslab.schema.design_schema import DesignSpec


def objective_scores(spec: DesignSpec, predictions: pd.DataFrame) -> pd.Series:
    """Compute direction-aware normalized objective scores."""
    scores: list[pd.Series] = []
    for objective in spec.objectives:
        column = f"pred_{objective.name}"
        values = predictions[column].astype(float)
        low = float(values.min())
        high = float(values.max())
        if np.isclose(high, low):
            score = pd.Series(0.5, index=predictions.index)
        elif objective.direction == "maximize":
            score = (values - low) / (high - low)
        else:
            score = (high - values) / (high - low)
        scores.append(score.clip(0.0, 1.0))
    return pd.concat(scores, axis=1).mean(axis=1)


def feasibility_scores(spec: DesignSpec, predictions: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Compute predicted feasibility scores and the main risk constraint per row."""
    rules = parse_constraints(spec.constraints)
    if not rules:
        return pd.Series(1.0, index=predictions.index), ["No configured constraints."] * len(predictions)

    per_rule_scores: list[pd.Series] = []
    satisfaction = predicted_constraint_satisfaction(spec, predictions)
    violation_labels: list[list[tuple[str, float]]] = [[] for _ in range(len(predictions))]
    for rule in rules:
        column = f"pred_{rule.column}"
        values = predictions[column].astype(float)
        scale = max(abs(rule.threshold), 1.0)
        if rule.operator in {"<=", "<"}:
            violation = ((values - rule.threshold) / scale).clip(lower=0.0)
        elif rule.operator in {">=", ">"}:
            violation = ((rule.threshold - values) / scale).clip(lower=0.0)
        else:
            violation = (values - rule.threshold).abs() / scale

        score = (1.0 / (1.0 + 4.0 * violation)).clip(0.0, 1.0)
        per_rule_scores.append(score)
        rule_satisfied = satisfaction[f"constraint__{rule.name}"].to_numpy()
        for idx, (is_satisfied, amount) in enumerate(
            zip(rule_satisfied, violation.to_numpy(), strict=True)
        ):
            if not is_satisfied:
                violation_labels[idx].append((rule.name, float(amount)))

    feasibility = pd.concat(per_rule_scores, axis=1).mean(axis=1)
    main_risks = []
    for labels in violation_labels:
        if not labels:
            main_risks.append("All predicted constraints are satisfied.")
        else:
            labels.sort(key=lambda item: item[1], reverse=True)
            main_risks.append(f"{labels[0][0]} is the main predicted constraint risk.")
    return feasibility, main_risks


def predicted_constraint_satisfaction(spec: DesignSpec, predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate hard predicted constraint satisfaction using the configured expressions."""
    rules = parse_constraints(spec.constraints)
    result = pd.DataFrame(index=predictions.index)
    if not rules:
        result["predicted_feasible"] = True
        return result

    constraint_columns: list[str] = []
    for rule in rules:
        column_name = f"constraint__{rule.name}"
        result[column_name] = _evaluate_predicted_rule(predictions, rule)
        constraint_columns.append(column_name)
    result["predicted_feasible"] = result[constraint_columns].all(axis=1)
    return result


def _evaluate_predicted_rule(predictions: pd.DataFrame, rule: ConstraintRule) -> pd.Series:
    column = f"pred_{rule.column}"
    if column not in predictions.columns:
        raise KeyError(f"Predictions are missing required constraint column {column!r}")
    values = predictions[column].astype(float)
    if rule.operator == "<=":
        return values <= rule.threshold
    if rule.operator == "<":
        return values < rule.threshold
    if rule.operator == ">=":
        return values >= rule.threshold
    if rule.operator == ">":
        return values > rule.threshold
    return pd.Series(np.isclose(values, rule.threshold), index=predictions.index)
