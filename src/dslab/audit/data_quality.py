from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from dslab.audit.constraints import add_constraint_columns, parse_constraints
from dslab.schema.design_schema import DesignSpec


@dataclass(frozen=True)
class AuditResult:
    row_count: int
    variables: list[str]
    objectives: list[str]
    constraints: list[str]
    missing_values: dict[str, int]
    duplicate_rows: int
    continuous_range_violations: dict[str, int]
    categorical_unknowns: dict[str, int]
    feasible_count: int
    infeasible_count: int
    group_count: int | None


def compute_audit(spec: DesignSpec, df: pd.DataFrame) -> tuple[AuditResult, pd.DataFrame]:
    """Compute data-quality and feasibility audit information."""
    evaluated = add_constraint_columns(df, spec.constraints)
    missing_values = {column: int(count) for column, count in df.isna().sum().items() if count > 0}
    continuous_violations: dict[str, int] = {}
    categorical_unknowns: dict[str, int] = {}

    for name, variable in spec.variables.items():
        if variable.type == "continuous" and variable.range is not None:
            lower, upper = variable.range
            mask = df[name].notna() & ((df[name] < lower) | (df[name] > upper))
            continuous_violations[name] = int(mask.sum())
        if variable.type == "categorical" and variable.values is not None:
            allowed = set(variable.values)
            mask = df[name].notna() & ~df[name].isin(allowed)
            categorical_unknowns[name] = int(mask.sum())

    feasible_count = int(evaluated["feasible"].sum())
    group_count = None
    if spec.data.group_column:
        group_count = int(df[spec.data.group_column].nunique(dropna=True))

    result = AuditResult(
        row_count=len(df),
        variables=spec.variable_names,
        objectives=spec.objective_names,
        constraints=[constraint.expression for constraint in spec.constraints],
        missing_values=missing_values,
        duplicate_rows=int(df.duplicated().sum()),
        continuous_range_violations=continuous_violations,
        categorical_unknowns=categorical_unknowns,
        feasible_count=feasible_count,
        infeasible_count=int(len(df) - feasible_count),
        group_count=group_count,
    )
    return result, evaluated


def render_audit_console(audit: AuditResult, console: Console | None = None) -> None:
    """Render an audit summary using Rich."""
    console = console or Console()
    console.print("[bold]Design-space audit[/bold]")

    summary = Table(show_header=True, header_style="bold")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Rows", str(audit.row_count))
    summary.add_row("Variables", str(len(audit.variables)))
    summary.add_row("Objectives", str(len(audit.objectives)))
    summary.add_row("Constraints", str(len(audit.constraints)))
    summary.add_row("Duplicate rows", str(audit.duplicate_rows))
    summary.add_row("Feasible rows", str(audit.feasible_count))
    summary.add_row("Infeasible rows", str(audit.infeasible_count))
    if audit.group_count is not None:
        summary.add_row("Groups", str(audit.group_count))
    console.print(summary)

    quality = Table(show_header=True, header_style="bold")
    quality.add_column("Check")
    quality.add_column("Details")
    quality.add_row("Missing values", _format_dict(audit.missing_values))
    quality.add_row("Range violations", _format_dict(audit.continuous_range_violations))
    quality.add_row("Unknown categories", _format_dict(audit.categorical_unknowns))
    console.print(quality)


def audit_to_markdown(audit: AuditResult, spec: DesignSpec) -> str:
    """Convert audit results to a Markdown report."""
    parsed_constraints = parse_constraints(spec.constraints)
    lines = [
        f"# Audit: {spec.project.name}",
        "",
        spec.project.description,
        "",
        "## Summary",
        "",
        f"- Rows: {audit.row_count}",
        f"- Variables: {len(audit.variables)} ({', '.join(audit.variables)})",
        f"- Objectives: {len(audit.objectives)} ({', '.join(audit.objectives)})",
        f"- Constraints: {len(audit.constraints)}",
        f"- Duplicate rows: {audit.duplicate_rows}",
        f"- Feasible rows: {audit.feasible_count}",
        f"- Infeasible rows: {audit.infeasible_count}",
    ]
    if audit.group_count is not None:
        lines.append(f"- Groups: {audit.group_count}")
    lines.extend(
        [
            "",
            "## Constraints",
            "",
        ]
    )
    if parsed_constraints:
        lines.extend(f"- `{rule.expression}`" for rule in parsed_constraints)
    else:
        lines.append("- No constraints configured.")
    lines.extend(
        [
            "",
            "## Data Quality",
            "",
            f"- Missing values: {_format_dict(audit.missing_values)}",
            f"- Continuous range violations: {_format_dict(audit.continuous_range_violations)}",
            f"- Unknown categorical values: {_format_dict(audit.categorical_unknowns)}",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_markdown(audit: AuditResult, spec: DesignSpec, out_dir: Path) -> Path:
    """Write audit Markdown to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "audit.md"
    path.write_text(audit_to_markdown(audit, spec), encoding="utf-8")
    return path


def _format_dict(values: dict[str, int]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in values.items())
