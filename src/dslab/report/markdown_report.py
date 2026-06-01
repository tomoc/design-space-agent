from __future__ import annotations

from pathlib import Path

import pandas as pd

from dslab.audit.data_quality import AuditResult
from dslab.schema.design_schema import DesignSpec


def write_markdown_report(
    spec: DesignSpec,
    audit: AuditResult,
    pareto_df: pd.DataFrame,
    metrics: pd.DataFrame,
    recommendations: pd.DataFrame,
    plots: dict[str, Path],
    out_path: Path,
) -> Path:
    """Write a concise Markdown report."""
    lines = [
        f"# {spec.project.name} design-space report",
        "",
        spec.project.description,
        "",
        "Example datasets in this project are synthetic demo data.",
        "",
        "## Summary",
        "",
        f"- Rows: {audit.row_count}",
        f"- Feasible rows: {audit.feasible_count}",
        f"- Infeasible rows: {audit.infeasible_count}",
        f"- Pareto front rows: {len(pareto_df)}",
        "",
        "## Design variables",
        "",
    ]
    for name, variable in spec.variables.items():
        detail = variable.range if variable.type == "continuous" else variable.values
        unit = f" {variable.unit}" if variable.unit else ""
        lines.append(f"- `{name}`: {variable.type}, {detail}{unit}")

    lines.extend(["", "## Objectives", ""])
    for objective in spec.objectives:
        unit = f" ({objective.unit})" if objective.unit else ""
        lines.append(f"- `{objective.name}`: {objective.direction}{unit}")

    lines.extend(["", "## Constraints", ""])
    for constraint in spec.constraints:
        lines.append(f"- `{constraint.expression}`")

    lines.extend(
        [
            "",
            "## Data audit",
            "",
            f"- Missing values: {audit.missing_values or 'none'}",
            f"- Duplicate rows: {audit.duplicate_rows}",
            f"- Continuous range violations: {audit.continuous_range_violations}",
            f"- Unknown categories: {audit.categorical_unknowns}",
            "",
            "## Surrogate model performance",
            "",
            _dataframe_to_markdown(metrics.round(4)),
            "",
            "## Trust / extrapolation summary",
            "",
            "The MVP trust score is heuristic and is not calibrated uncertainty.",
            "",
            "## Recommended next candidates",
            "",
            "`predicted_feasible` is a hard check against the configured constraint expressions using surrogate predictions. `recommendation_score` is the configured weighted average of objective, feasibility, and trust scores.",
            "",
            _dataframe_to_markdown(recommendations.round(4)),
            "",
            "## Plots",
            "",
        ]
    )
    for label, path in plots.items():
        lines.append(f"- {label}: `{path}`")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = [str(column) for column in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return "\n".join([header, separator, *rows])
