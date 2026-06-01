from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from dslab.cli import app


def _write_small_problem(tmp_path: Path) -> Path:
    rows = ["scenario_id,x,y,mode,obj,cost,temp"]
    modes = ["a", "b"]
    for i in range(30):
        x = i / 29
        y = 1.0 - x / 2
        mode = modes[i % 2]
        obj = 10 + 3 * x - y + (0.2 if mode == "b" else 0.0)
        cost = 100 + 20 * x + 10 * y + (5 if mode == "b" else 0)
        temp = 70 + 12 * x - 5 * y + (1 if mode == "b" else 0)
        rows.append(f"g{i // 5},{x:.4f},{y:.4f},{mode},{obj:.4f},{cost:.4f},{temp:.4f}")
    (tmp_path / "samples.csv").write_text("\n".join(rows), encoding="utf-8")
    design = {
        "project": {"name": "cli_demo", "description": "Synthetic CLI smoke-test data"},
        "data": {"path": "samples.csv", "group_column": "scenario_id"},
        "variables": {
            "x": {"type": "continuous", "range": [0, 1]},
            "y": {"type": "continuous", "range": [0, 1]},
            "mode": {"type": "categorical", "values": ["a", "b"]},
        },
        "objectives": [
            {"name": "obj", "direction": "maximize"},
            {"name": "cost", "direction": "minimize"},
        ],
        "constraints": [{"name": "temp_limit", "expression": "temp <= 78.0"}],
        "recommendation": {
            "candidate_pool_size": 40,
            "random_seed": 4,
            "trust_penalty_weight": 0.35,
            "feasibility_weight": 0.40,
            "objective_weight": 0.25,
        },
    }
    design_path = tmp_path / "design.yaml"
    design_path.write_text(yaml.safe_dump(design), encoding="utf-8")
    return design_path


def test_cli_smoke_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    design_path = _write_small_problem(tmp_path)
    out = tmp_path / "outputs"

    for args in [
        ["audit", str(design_path), "--out", str(out)],
        ["explore", str(design_path), "--out", str(out)],
        ["recommend", str(design_path), "--n", "3", "--out", str(out)],
        ["report", str(design_path), "--out", str(out / "report.html")],
        ["run", str(design_path), "--out", str(out), "--n", "3"],
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output

    assert (out / "audit.md").exists()
    assert (out / "pareto_front.csv").exists()
    assert (out / "recommendations.csv").exists()
    assert (out / "report.html").exists()
    assert (out / "plots" / "feasible_map.png").exists()
