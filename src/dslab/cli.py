from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from dslab.audit.data_quality import (
    AuditResult,
    compute_audit,
    render_audit_console,
    write_audit_markdown,
)
from dslab.models.surrogate import SurrogateResult, train_surrogates
from dslab.optimize.pareto import ParetoResult, extract_pareto_front
from dslab.optimize.recommender import recommend_candidates
from dslab.report.html_report import write_html_report
from dslab.report.markdown_report import write_markdown_report
from dslab.schema.design_schema import DesignSpec
from dslab.schema.parser import DesignConfigError, load_dataset, load_design_problem

app = typer.Typer(no_args_is_help=True, help="Trust-aware design-space exploration CLI.")
console = Console()

DesignYamlArg = Annotated[Path, typer.Argument(help="Path to design.yaml")]
OutDirOption = Annotated[Path, typer.Option("--out", "-o", help="Output directory")]
ReportOutOption = Annotated[Path, typer.Option("--out", "-o", help="HTML report path")]
NOption = Annotated[int, typer.Option("--n", "-n", min=1, help="Number of recommendations")]


@dataclass
class PipelineArtifacts:
    spec: DesignSpec
    data: pd.DataFrame
    audit: AuditResult
    pareto: ParetoResult
    surrogate: SurrogateResult
    recommendations: pd.DataFrame
    plots: dict[str, Path]


@app.command(name="audit")
def audit_command(
    design_yaml: DesignYamlArg,
    out: OutDirOption = Path("outputs"),
) -> None:
    """Audit YAML, CSV data quality, and feasibility."""
    _safe_run(lambda: _audit_impl(design_yaml, out))


@app.command()
def explore(
    design_yaml: DesignYamlArg,
    out: OutDirOption = Path("outputs"),
) -> None:
    """Extract Pareto front, create plots, and evaluate surrogate models."""
    _safe_run(lambda: _explore_impl(design_yaml, out))


@app.command()
def recommend(
    design_yaml: DesignYamlArg,
    n: NOption = 5,
    out: OutDirOption = Path("outputs"),
) -> None:
    """Recommend next candidate designs."""
    _safe_run(lambda: _recommend_impl(design_yaml, n, out))


@app.command()
def report(
    design_yaml: DesignYamlArg,
    out: ReportOutOption = Path("outputs/report.html"),
) -> None:
    """Generate a static HTML and Markdown report."""
    _safe_run(lambda: _report_impl(design_yaml, out))


@app.command()
def run(
    design_yaml: DesignYamlArg,
    out: OutDirOption = Path("outputs"),
    n: NOption = 5,
) -> None:
    """Run audit, exploration, recommendation, plotting, and reporting."""
    _safe_run(lambda: _run_impl(design_yaml, out, n))


def _audit_impl(design_yaml: Path, out: Path) -> None:
    spec, df = _load_spec_and_data(design_yaml)
    audit, _ = compute_audit(spec, df)
    render_audit_console(audit, console)
    path = write_audit_markdown(audit, spec, out)
    console.print(f"[green]Wrote audit:[/green] {path}")


def _explore_impl(design_yaml: Path, out: Path) -> None:
    spec, df = _load_spec_and_data(design_yaml)
    audit, evaluated = compute_audit(spec, df)
    write_audit_markdown(audit, spec, out)
    pareto = extract_pareto_front(spec, evaluated)
    if pareto.warning:
        console.print(f"[yellow]{pareto.warning}[/yellow]")
    out.mkdir(parents=True, exist_ok=True)
    pareto_path = out / "pareto_front.csv"
    pareto.pareto_front.to_csv(pareto_path, index=False)
    plots = _create_plots(spec, evaluated, pareto.pareto_front, out / "plots")
    surrogate = train_surrogates(spec, evaluated)
    (out / "surrogate_metrics.csv").write_text(surrogate.metrics.to_csv(index=False), encoding="utf-8")
    console.print(f"[green]Pareto front rows:[/green] {len(pareto.pareto_front)}")
    console.print(f"[green]Wrote Pareto front:[/green] {pareto_path}")
    _print_metrics(surrogate.metrics)
    for name, path in plots.items():
        console.print(f"[green]Wrote plot {name}:[/green] {path}")


def _recommend_impl(design_yaml: Path, n: int, out: Path) -> None:
    spec, df = _load_spec_and_data(design_yaml)
    _, evaluated = compute_audit(spec, df)
    result = recommend_candidates(spec, evaluated, n=n)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "recommendations.csv"
    result.recommendations.to_csv(path, index=False)
    (out / "surrogate_metrics.csv").write_text(result.surrogate.metrics.to_csv(index=False), encoding="utf-8")
    console.print(f"[green]Wrote recommendations:[/green] {path}")
    _print_recommendations(result.recommendations)


def _report_impl(design_yaml: Path, out: Path) -> None:
    report_path = _resolve_report_path(out)
    artifacts = _run_pipeline(design_yaml, report_path.parent, n=5)
    write_html_report(
        artifacts.spec,
        artifacts.audit,
        artifacts.pareto.pareto_front,
        artifacts.surrogate.metrics,
        artifacts.recommendations,
        artifacts.plots,
        report_path,
    )
    md_path = report_path.with_suffix(".md")
    write_markdown_report(
        artifacts.spec,
        artifacts.audit,
        artifacts.pareto.pareto_front,
        artifacts.surrogate.metrics,
        artifacts.recommendations,
        artifacts.plots,
        md_path,
    )
    console.print(f"[green]Wrote report:[/green] {report_path}")
    console.print(f"[green]Wrote Markdown report:[/green] {md_path}")


def _run_impl(design_yaml: Path, out: Path, n: int) -> None:
    artifacts = _run_pipeline(design_yaml, out, n=n)
    write_html_report(
        artifacts.spec,
        artifacts.audit,
        artifacts.pareto.pareto_front,
        artifacts.surrogate.metrics,
        artifacts.recommendations,
        artifacts.plots,
        out / "report.html",
    )
    write_markdown_report(
        artifacts.spec,
        artifacts.audit,
        artifacts.pareto.pareto_front,
        artifacts.surrogate.metrics,
        artifacts.recommendations,
        artifacts.plots,
        out / "report.md",
    )
    console.print(f"[green]Completed run:[/green] {out}")
    console.print(f"[green]Recommendations:[/green] {out / 'recommendations.csv'}")
    console.print(f"[green]Report:[/green] {out / 'report.html'}")


def _run_pipeline(design_yaml: Path, out: Path, n: int) -> PipelineArtifacts:
    spec, df = _load_spec_and_data(design_yaml)
    out.mkdir(parents=True, exist_ok=True)
    audit, evaluated = compute_audit(spec, df)
    write_audit_markdown(audit, spec, out)

    pareto = extract_pareto_front(spec, evaluated)
    if pareto.warning:
        console.print(f"[yellow]{pareto.warning}[/yellow]")
    pareto.pareto_front.to_csv(out / "pareto_front.csv", index=False)

    plots = _create_plots(spec, evaluated, pareto.pareto_front, out / "plots")
    recommendation_result = recommend_candidates(spec, evaluated, n=n)
    recommendation_result.recommendations.to_csv(out / "recommendations.csv", index=False)
    recommendation_result.surrogate.metrics.to_csv(out / "surrogate_metrics.csv", index=False)

    render_audit_console(audit, console)
    console.print(f"[green]Pareto front rows:[/green] {len(pareto.pareto_front)}")
    _print_metrics(recommendation_result.surrogate.metrics)

    return PipelineArtifacts(
        spec=spec,
        data=evaluated,
        audit=audit,
        pareto=pareto,
        surrogate=recommendation_result.surrogate,
        recommendations=recommendation_result.recommendations,
        plots=plots,
    )


def _load_spec_and_data(design_yaml: Path) -> tuple[DesignSpec, pd.DataFrame]:
    problem = load_design_problem(design_yaml)
    return problem.spec, load_dataset(problem)


def _resolve_report_path(out: Path) -> Path:
    if out.suffix.lower() == ".html":
        return out
    return out / "report.html"


def _create_plots(
    spec: DesignSpec,
    evaluated: pd.DataFrame,
    pareto_front: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    from dslab.viz.plots import create_plots

    return create_plots(spec, evaluated, pareto_front, out_dir)


def _print_metrics(metrics: pd.DataFrame) -> None:
    table = Table(title="Surrogate metrics", show_header=True, header_style="bold")
    for column in metrics.columns:
        table.add_column(column)
    for _, row in metrics.round(4).iterrows():
        table.add_row(*(str(value) for value in row.tolist()))
    console.print(table)


def _print_recommendations(recommendations: pd.DataFrame) -> None:
    table = Table(title="Top recommendations", show_header=True, header_style="bold")
    columns = [
        "candidate_rank",
        "recommendation_score",
        "objective_score",
        "feasibility_score",
        "predicted_feasible",
        "trust_score",
    ]
    for column in columns:
        table.add_column(column)
    for _, row in recommendations.round(4).iterrows():
        table.add_row(*(str(row[column]) for column in columns))
    console.print(table)


def _safe_run(fn: Callable[[], None]) -> None:
    try:
        fn()  # type: ignore[operator]
    except (DesignConfigError, ValueError, KeyError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
