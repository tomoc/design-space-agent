from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dslab.schema.design_schema import DesignSpec
from dslab.schema.parser import DesignConfigError, load_dataset, load_design_problem

ROOT = Path(__file__).resolve().parents[1]


def test_load_vehicle_schema_and_dataset() -> None:
    problem = load_design_problem(ROOT / "examples" / "vehicle_concept" / "design.yaml")
    df = load_dataset(problem)

    assert problem.spec.project.name == "vehicle_concept_demo"
    assert problem.data_path.name == "samples.csv"
    assert len(df) == 120
    assert "battery_kwh" in df.columns


def test_continuous_variable_requires_range() -> None:
    raw = {
        "project": {"name": "bad"},
        "data": {"path": "samples.csv"},
        "variables": {"x": {"type": "continuous"}},
        "objectives": [{"name": "y", "direction": "maximize"}],
    }

    with pytest.raises(ValueError, match="continuous variables require range"):
        DesignSpec.model_validate(raw)


def test_missing_csv_columns_are_reported(tmp_path: Path) -> None:
    design = {
        "project": {"name": "missing_column_demo"},
        "data": {"path": "samples.csv"},
        "variables": {"x": {"type": "continuous", "range": [0, 1]}},
        "objectives": [{"name": "missing_y", "direction": "maximize"}],
    }
    (tmp_path / "design.yaml").write_text(yaml.safe_dump(design), encoding="utf-8")
    (tmp_path / "samples.csv").write_text("x\n0.1\n", encoding="utf-8")

    problem = load_design_problem(tmp_path / "design.yaml")
    with pytest.raises(DesignConfigError, match="missing_y"):
        load_dataset(problem)
