from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from dslab.schema.design_schema import DesignSpec


def create_plots(spec: DesignSpec, df: pd.DataFrame, pareto_df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Create MVP static plots and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x_col, y_col = _representative_axes(spec)
    plots = {
        "feasible_map": out_dir / "feasible_map.png",
        "pareto_front": out_dir / "pareto_front.png",
        "objective_scatter": out_dir / "objective_scatter.png",
    }
    _plot_feasible_map(spec, df, x_col, y_col, plots["feasible_map"])
    _plot_pareto_front(spec, df, pareto_df, x_col, y_col, plots["pareto_front"])
    _plot_objective_scatter(spec, df, plots["objective_scatter"])
    return plots


def _representative_axes(spec: DesignSpec) -> tuple[str, str]:
    continuous = spec.continuous_variable_names
    if len(continuous) >= 2:
        return continuous[0], continuous[1]
    if len(continuous) == 1:
        return continuous[0], continuous[0]
    variables = spec.variable_names
    if len(variables) >= 2:
        return variables[0], variables[1]
    return variables[0], variables[0]


def _plot_feasible_map(
    spec: DesignSpec,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    path: Path,
) -> None:
    plt = _pyplot()
    objective = spec.objective_names[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    feasible = df["feasible"] if "feasible" in df.columns else pd.Series(True, index=df.index)
    infeasible_df = df[~feasible]
    feasible_df = df[feasible]
    if len(infeasible_df):
        ax.scatter(infeasible_df[x_col], infeasible_df[y_col], c="tab:red", label="infeasible", alpha=0.65)
    if len(feasible_df):
        scatter = ax.scatter(
            feasible_df[x_col],
            feasible_df[y_col],
            c=feasible_df[objective],
            cmap="viridis",
            label="feasible",
            alpha=0.85,
        )
        fig.colorbar(scatter, ax=ax, label=objective)
    ax.set_title("Feasible / infeasible map")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pareto_front(
    spec: DesignSpec,
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    path: Path,
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[x_col], df[y_col], c="lightgray", label="samples", alpha=0.65)
    if len(pareto_df):
        ax.scatter(
            pareto_df[x_col],
            pareto_df[y_col],
            c="tab:blue",
            label="Pareto front",
            alpha=0.95,
            edgecolors="black",
        )
    ax.set_title("Pareto front in design-variable space")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_objective_scatter(spec: DesignSpec, df: pd.DataFrame, path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    objectives = spec.objective_names
    feasible = df["feasible"] if "feasible" in df.columns else pd.Series(True, index=df.index)
    colors = feasible.map({True: "tab:blue", False: "tab:red"})
    if len(objectives) >= 2:
        ax.scatter(df[objectives[0]], df[objectives[1]], c=colors, alpha=0.75)
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_title("Objective pair scatter")
    else:
        x_col = spec.continuous_variable_names[0] if spec.continuous_variable_names else spec.variable_names[0]
        ax.scatter(df[x_col], df[objectives[0]], c=colors, alpha=0.75)
        ax.set_xlabel(x_col)
        ax.set_ylabel(objectives[0])
        ax.set_title("Objective scatter")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pyplot() -> Any:
    _mpl_config_dir = Path(tempfile.gettempdir()) / "dslab-matplotlib"
    _mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
