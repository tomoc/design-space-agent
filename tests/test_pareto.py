from __future__ import annotations

import pandas as pd

from dslab.optimize.pareto import extract_pareto_front, pareto_mask
from dslab.schema.design_schema import DesignSpec


def test_pareto_mask_respects_objective_directions() -> None:
    values = pd.DataFrame(
        {
            "performance": [10.0, 9.0, 8.0, 10.0],
            "cost": [5.0, 4.0, 7.0, 6.0],
        }
    ).to_numpy()

    mask = pareto_mask(values, ["maximize", "minimize"])

    assert mask.tolist() == [True, True, False, False]


def test_extract_pareto_front_uses_feasible_rows() -> None:
    spec = DesignSpec.model_validate(
        {
            "project": {"name": "pareto_demo"},
            "data": {"path": "samples.csv"},
            "variables": {"x": {"type": "continuous", "range": [0, 1]}},
            "objectives": [
                {"name": "performance", "direction": "maximize"},
                {"name": "cost", "direction": "minimize"},
            ],
        }
    )
    df = pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.3],
            "performance": [10.0, 9.0, 12.0],
            "cost": [5.0, 4.0, 1.0],
            "feasible": [True, True, False],
        }
    )

    result = extract_pareto_front(spec, df)

    assert result.used_feasible_rows is True
    assert result.pareto_front["x"].tolist() == [0.1, 0.2]
