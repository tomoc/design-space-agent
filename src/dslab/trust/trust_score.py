from __future__ import annotations

import pandas as pd

from dslab.schema.design_schema import DesignSpec
from dslab.trust.extrapolation import (
    build_trust_profile,
    categorical_risk,
    continuous_distance_risk,
    range_risk,
)


def compute_trust_scores(spec: DesignSpec, train_df: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    """Compute heuristic trust scores in [0, 1] for candidate designs."""
    profile = build_trust_profile(spec, train_df)
    distance = continuous_distance_risk(profile, candidates)
    out_of_range = range_risk(profile, candidates)
    unseen_category, unseen_combo = categorical_risk(profile, candidates)
    risk = (
        0.45 * distance
        + 0.30 * out_of_range
        + 0.20 * unseen_category
        + 0.05 * unseen_combo
    )
    return (1.0 - risk).clip(0.0, 1.0)
