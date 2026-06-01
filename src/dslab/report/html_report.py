from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Template

from dslab.audit.data_quality import AuditResult
from dslab.schema.design_schema import DesignSpec

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ spec.project.name }} design-space report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; line-height: 1.5; }
    main { max-width: 1180px; margin: 0 auto; }
    h1, h2, h3 { color: #102a43; }
    h1 { margin-bottom: 4px; }
    h2 { border-top: 1px solid #d9e2ec; margin-top: 28px; padding-top: 20px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }
    th, td { border: 1px solid #d9e2ec; padding: 8px; text-align: left; vertical-align: top; }
    th { background: #f0f4f8; position: sticky; top: 0; }
    tr:nth-child(even) td { background: #fbfdff; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
    .metric { border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; background: #f8fafc; }
    .metric strong { color: #334e68; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric span { display: block; font-size: 24px; margin-top: 4px; }
    .note { background: #f8fafc; border-left: 4px solid #627d98; padding: 10px 12px; }
    .table-wrap { overflow-x: auto; max-width: 100%; }
    img { max-width: 100%; border: 1px solid #d9e2ec; border-radius: 6px; }
    code { background: #f0f4f8; padding: 2px 4px; border-radius: 4px; }
  </style>
</head>
<body>
<main>
  <h1>{{ spec.project.name }}</h1>
  <p>{{ spec.project.description }}</p>
  <p><strong>Data note:</strong> Example datasets in this project are synthetic demo data.</p>

  <h2>Project summary</h2>
  <div class="grid">
    <div class="metric"><strong>Rows</strong><span>{{ audit.row_count }}</span></div>
    <div class="metric"><strong>Feasible</strong><span>{{ audit.feasible_count }}</span></div>
    <div class="metric"><strong>Infeasible</strong><span>{{ audit.infeasible_count }}</span></div>
    <div class="metric"><strong>Pareto front</strong><span>{{ pareto_count }}</span></div>
  </div>

  <h2>Design variables</h2>
  <table>
    <tr><th>Name</th><th>Type</th><th>Range / values</th><th>Unit</th></tr>
    {% for name, variable in spec.variables.items() %}
    <tr>
      <td>{{ name }}</td>
      <td>{{ variable.type }}</td>
      <td>{% if variable.type == "continuous" %}{{ variable.range }}{% else %}{{ variable.values }}{% endif %}</td>
      <td>{{ variable.unit or "" }}</td>
    </tr>
    {% endfor %}
  </table>

  <h2>Objectives</h2>
  <table>
    <tr><th>Name</th><th>Direction</th><th>Unit</th></tr>
    {% for objective in spec.objectives %}
    <tr><td>{{ objective.name }}</td><td>{{ objective.direction }}</td><td>{{ objective.unit or "" }}</td></tr>
    {% endfor %}
  </table>

  <h2>Constraints</h2>
  <ul>
    {% for constraint in spec.constraints %}
    <li><code>{{ constraint.expression }}</code></li>
    {% endfor %}
  </ul>

  <h2>Data audit</h2>
  <ul>
    <li>Missing values: {{ audit.missing_values or "none" }}</li>
    <li>Duplicate rows: {{ audit.duplicate_rows }}</li>
    <li>Continuous range violations: {{ audit.continuous_range_violations }}</li>
    <li>Unknown categories: {{ audit.categorical_unknowns }}</li>
    {% if audit.group_count is not none %}<li>Groups: {{ audit.group_count }}</li>{% endif %}
  </ul>

  <h2>Surrogate model performance</h2>
  <div class="table-wrap">{{ metrics_html }}</div>

  <h2>Trust / extrapolation summary</h2>
  <p class="note">The MVP trust score is a heuristic score from 0 to 1. It combines observed training-range checks, nearest-neighbor distance in continuous-variable space, unseen categories, and unseen category combinations. It is not calibrated uncertainty.</p>

  <h2>Recommended next candidates</h2>
  <p><code>predicted_feasible</code> is a hard check against the configured constraint expressions using surrogate predictions. <code>recommendation_score</code> is the configured weighted average of objective, feasibility, and trust scores.</p>
  <div class="table-wrap">{{ recommendations_html }}</div>

  <h2>Plots</h2>
  {% for label, path in plots.items() %}
    <h3>{{ label }}</h3>
    <img src="{{ path }}" alt="{{ label }}">
  {% endfor %}
</main>
</body>
</html>
"""


def write_html_report(
    spec: DesignSpec,
    audit: AuditResult,
    pareto_df: pd.DataFrame,
    metrics: pd.DataFrame,
    recommendations: pd.DataFrame,
    plots: dict[str, Path],
    out_path: Path,
) -> Path:
    """Write a static HTML design-space report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    relative_plots = {key.replace("_", " ").title(): _relative(path, out_path.parent) for key, path in plots.items()}
    html = Template(HTML_TEMPLATE).render(
        spec=spec,
        audit=audit,
        pareto_count=len(pareto_df),
        metrics_html=metrics.round(4).to_html(index=False, escape=True),
        recommendations_html=recommendations.round(4).to_html(index=False, escape=True),
        plots=relative_plots,
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
