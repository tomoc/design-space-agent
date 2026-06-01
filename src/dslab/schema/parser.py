from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import ValidationError

from dslab.audit.constraints import constraint_target_columns
from dslab.schema.design_schema import DesignSpec


class DesignConfigError(ValueError):
    """Raised when the YAML or CSV does not match the design-space schema."""


@dataclass(frozen=True)
class DesignProblem:
    spec: DesignSpec
    yaml_path: Path
    base_dir: Path
    data_path: Path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    if not path.exists():
        raise DesignConfigError(f"Design YAML not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DesignConfigError(f"Failed to parse YAML {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DesignConfigError(f"Design YAML must contain a mapping at the top level: {path}")
    return raw


def load_design_problem(path: str | Path) -> DesignProblem:
    """Load and validate a design-space YAML file."""
    yaml_path = Path(path).expanduser().resolve()
    raw = load_yaml(yaml_path)
    try:
        spec = DesignSpec.model_validate(raw)
    except ValidationError as exc:
        raise DesignConfigError(f"Invalid design YAML {yaml_path}:\n{exc}") from exc

    base_dir = yaml_path.parent
    data_path = Path(spec.data.path)
    if not data_path.is_absolute():
        data_path = base_dir / data_path
    return DesignProblem(
        spec=spec,
        yaml_path=yaml_path,
        base_dir=base_dir,
        data_path=data_path.resolve(),
    )


def load_dataset(problem: DesignProblem) -> pd.DataFrame:
    """Load the CSV dataset and validate required columns."""
    if not problem.data_path.exists():
        raise DesignConfigError(f"CSV data file not found: {problem.data_path}")
    try:
        df = pd.read_csv(problem.data_path)
    except Exception as exc:  # noqa: BLE001
        raise DesignConfigError(f"Failed to read CSV data {problem.data_path}: {exc}") from exc
    validate_dataframe_columns(problem.spec, df)
    return df


def validate_dataframe_columns(spec: DesignSpec, df: pd.DataFrame) -> None:
    """Ensure the CSV contains all columns referenced by the design spec."""
    required = set(spec.variable_names)
    required.update(spec.objective_names)
    required.update(constraint_target_columns(spec.constraints))
    if spec.data.group_column:
        required.add(spec.data.group_column)

    missing = sorted(required.difference(df.columns))
    if missing:
        raise DesignConfigError(
            "CSV is missing required columns referenced by the YAML: " + ", ".join(missing)
        )
