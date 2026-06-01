from __future__ import annotations

import numpy as np
import pandas as pd

from dslab.schema.design_schema import DesignSpec


def generate_candidates(spec: DesignSpec, n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Generate random mixed-variable candidates from the YAML design space."""
    size = n or spec.recommendation.candidate_pool_size
    random_seed = spec.recommendation.random_seed if seed is None else seed
    rng = np.random.default_rng(random_seed)
    data: dict[str, np.ndarray] = {}

    for name, variable in spec.variables.items():
        if variable.type == "continuous":
            assert variable.range is not None
            lower, upper = variable.range
            data[name] = rng.uniform(lower, upper, size=size)
        else:
            assert variable.values is not None
            data[name] = rng.choice(variable.values, size=size)
    return pd.DataFrame(data)
