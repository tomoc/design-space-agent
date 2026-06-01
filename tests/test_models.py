from __future__ import annotations

import numpy as np
import pandas as pd

from dslab.models.preprocessing import build_preprocessor
from dslab.models.surrogate import _split_indices
from dslab.schema.design_schema import DesignSpec


def test_one_hot_encoder_handles_unseen_categories() -> None:
    spec = DesignSpec.model_validate(
        {
            "project": {"name": "one_hot_demo"},
            "data": {"path": "samples.csv"},
            "variables": {
                "x": {"type": "continuous", "range": [0, 1]},
                "mode": {"type": "categorical", "values": ["a", "b"]},
            },
            "objectives": [{"name": "y", "direction": "maximize"}],
        }
    )
    preprocessor = build_preprocessor(spec)
    train = pd.DataFrame({"x": [0.0, 1.0], "mode": ["a", "b"]})

    preprocessor.fit(train)
    transformed = preprocessor.transform(pd.DataFrame({"x": [0.5], "mode": ["unseen"]}))

    assert transformed.shape == (1, 3)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed[0, 1:], [0.0, 0.0])


def test_group_column_uses_group_shuffle_split() -> None:
    spec = DesignSpec.model_validate(
        {
            "project": {"name": "group_split_demo"},
            "data": {"path": "samples.csv", "group_column": "scenario_id"},
            "variables": {"x": {"type": "continuous", "range": [0, 1]}},
            "objectives": [{"name": "y", "direction": "maximize"}],
            "recommendation": {"random_seed": 3},
        }
    )
    df = pd.DataFrame(
        {
            "scenario_id": ["g0"] * 4 + ["g1"] * 4 + ["g2"] * 4 + ["g3"] * 4,
            "x": np.linspace(0.0, 1.0, 16),
            "y": np.linspace(1.0, 2.0, 16),
        }
    )

    train_idx, test_idx = _split_indices(spec, df)
    train_groups = set(df.loc[train_idx, "scenario_id"])
    test_groups = set(df.loc[test_idx, "scenario_id"])

    assert train_groups
    assert test_groups
    assert train_groups.isdisjoint(test_groups)
