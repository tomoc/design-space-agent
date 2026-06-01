from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dslab.schema.design_schema import DesignSpec


@dataclass(frozen=True)
class ParetoResult:
    pareto_front: pd.DataFrame
    used_feasible_rows: bool
    warning: str | None = None


def pareto_mask(values: np.ndarray, directions: list[str]) -> np.ndarray:
    """Return a boolean mask for nondominated rows."""
    if values.ndim != 2:
        raise ValueError("Pareto values must be a 2D array.")
    if values.shape[0] == 0:
        return np.array([], dtype=bool)

    costs = values.astype(float).copy()
    for idx, direction in enumerate(directions):
        if direction == "maximize":
            costs[:, idx] *= -1
        elif direction != "minimize":
            raise ValueError(f"Unsupported objective direction: {direction}")

    efficient = np.ones(costs.shape[0], dtype=bool)
    for i, candidate in enumerate(costs):
        if not efficient[i]:
            continue
        dominated_by_any = np.all(costs <= candidate, axis=1) & np.any(costs < candidate, axis=1)
        dominated_by_any[i] = False
        if dominated_by_any.any():
            efficient[i] = False
    return efficient


def extract_pareto_front(spec: DesignSpec, df: pd.DataFrame) -> ParetoResult:
    """Extract Pareto front from feasible rows when available."""
    if "feasible" in df.columns and bool(df["feasible"].any()):
        base = df[df["feasible"]].copy()
        used_feasible = True
        warning = None
    else:
        base = df.copy()
        used_feasible = False
        warning = "No feasible rows were found; Pareto front was computed over all samples."

    objective_names = spec.objective_names
    values = base[objective_names].to_numpy()
    directions = [objective.direction for objective in spec.objectives]
    mask = pareto_mask(values, directions)
    pareto = base.loc[mask].copy()
    return ParetoResult(pareto_front=pareto, used_feasible_rows=used_feasible, warning=warning)
