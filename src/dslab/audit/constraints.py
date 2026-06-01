from __future__ import annotations

import operator
import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dslab.schema.design_schema import ConstraintSpec

CONSTRAINT_PATTERN = re.compile(
    r"^\s*(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<op><=|>=|<|>|==)\s*"
    r"(?P<threshold>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


@dataclass(frozen=True)
class ConstraintRule:
    name: str
    expression: str
    column: str
    operator: str
    threshold: float


class ConstraintParseError(ValueError):
    """Raised for unsupported constraint expressions."""


def parse_constraint_expression(name: str, expression: str) -> ConstraintRule:
    """Parse a safe numeric column comparison constraint."""
    match = CONSTRAINT_PATTERN.match(expression)
    if not match:
        msg = (
            f"Unsupported constraint expression for {name!r}: {expression!r}. "
            "Supported forms are: column <= number, column < number, "
            "column >= number, column > number, column == number."
        )
        raise ConstraintParseError(msg)
    return ConstraintRule(
        name=name,
        expression=expression,
        column=match.group("column"),
        operator=match.group("op"),
        threshold=float(match.group("threshold")),
    )


def parse_constraints(constraints: Iterable[ConstraintSpec]) -> list[ConstraintRule]:
    """Parse all constraints from a design specification."""
    return [parse_constraint_expression(item.name, item.expression) for item in constraints]


def constraint_target_columns(constraints: Iterable[ConstraintSpec]) -> list[str]:
    """Return the CSV target columns referenced by constraints."""
    columns: list[str] = []
    seen: set[str] = set()
    for rule in parse_constraints(constraints):
        if rule.column not in seen:
            seen.add(rule.column)
            columns.append(rule.column)
    return columns


def evaluate_rule(df: pd.DataFrame, rule: ConstraintRule) -> pd.Series:
    """Evaluate one parsed rule against a DataFrame."""
    if rule.column not in df.columns:
        raise KeyError(f"Constraint {rule.name!r} references missing column {rule.column!r}")
    series = df[rule.column]
    operations = {
        "<=": operator.le,
        "<": operator.lt,
        ">=": operator.ge,
        ">": operator.gt,
    }
    if rule.operator == "==":
        return pd.Series(np.isclose(series.astype(float), rule.threshold), index=df.index)
    return operations[rule.operator](series, rule.threshold).fillna(False)


def add_constraint_columns(
    df: pd.DataFrame,
    constraints: Iterable[ConstraintSpec],
    feasible_column: str = "feasible",
) -> pd.DataFrame:
    """Add one boolean column per constraint and an aggregate feasible column."""
    result = df.copy()
    rules = parse_constraints(constraints)
    if not rules:
        result[feasible_column] = True
        return result

    constraint_columns: list[str] = []
    for rule in rules:
        column_name = f"constraint__{rule.name}"
        result[column_name] = evaluate_rule(result, rule)
        constraint_columns.append(column_name)
    result[feasible_column] = result[constraint_columns].all(axis=1)
    return result
