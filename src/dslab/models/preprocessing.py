from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dslab.schema.design_schema import DesignSpec


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder across supported scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(spec: DesignSpec) -> ColumnTransformer:
    """Build preprocessing for mixed continuous/categorical design variables."""
    transformers = []
    continuous = spec.continuous_variable_names
    categorical = spec.categorical_variable_names
    if continuous:
        transformers.append(("continuous", StandardScaler(), continuous))
    if categorical:
        transformers.append(("categorical", make_one_hot_encoder(), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")
