from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from dslab.schema.design_schema import DesignSpec


@dataclass
class TrustProfile:
    continuous_columns: list[str]
    categorical_columns: list[str]
    continuous_min: dict[str, float]
    continuous_max: dict[str, float]
    categorical_values: dict[str, set[str]]
    categorical_combinations: set[tuple[str, ...]]
    nearest_neighbors: NearestNeighbors | None
    distance_scale: float


def build_trust_profile(spec: DesignSpec, train_df: pd.DataFrame) -> TrustProfile:
    """Build a simple profile of the observed training design space."""
    continuous = spec.continuous_variable_names
    categorical = spec.categorical_variable_names
    continuous_min = {column: float(train_df[column].min()) for column in continuous}
    continuous_max = {column: float(train_df[column].max()) for column in continuous}
    categorical_values = {
        column: set(train_df[column].dropna().astype(str).unique().tolist()) for column in categorical
    }
    combinations = set()
    if categorical:
        for row in train_df[categorical].astype(str).itertuples(index=False, name=None):
            combinations.add(tuple(row))

    nearest_neighbors: NearestNeighbors | None = None
    distance_scale = 1.0
    if continuous:
        normalized = _normalize_continuous(train_df, continuous, continuous_min, continuous_max)
        nearest_neighbors = NearestNeighbors(n_neighbors=min(2, len(normalized)))
        nearest_neighbors.fit(normalized)
        if len(normalized) > 1:
            distances, _ = nearest_neighbors.kneighbors(normalized)
            neighbor_distance = distances[:, -1]
            distance_scale = float(np.quantile(neighbor_distance, 0.95))
            if distance_scale <= 1e-9:
                distance_scale = 1.0

    return TrustProfile(
        continuous_columns=continuous,
        categorical_columns=categorical,
        continuous_min=continuous_min,
        continuous_max=continuous_max,
        categorical_values=categorical_values,
        categorical_combinations=combinations,
        nearest_neighbors=nearest_neighbors,
        distance_scale=distance_scale,
    )


def continuous_distance_risk(profile: TrustProfile, candidates: pd.DataFrame) -> pd.Series:
    """Return normalized nearest-neighbor distance risk for candidates."""
    if not profile.continuous_columns or profile.nearest_neighbors is None:
        return pd.Series(0.0, index=candidates.index)
    normalized = _normalize_continuous(
        candidates,
        profile.continuous_columns,
        profile.continuous_min,
        profile.continuous_max,
    )
    distances, _ = profile.nearest_neighbors.kneighbors(normalized)
    risk = distances[:, 0] / profile.distance_scale
    return pd.Series(np.clip(risk, 0.0, 1.0), index=candidates.index)


def range_risk(profile: TrustProfile, candidates: pd.DataFrame) -> pd.Series:
    """Return fraction of continuous variables outside observed training ranges."""
    if not profile.continuous_columns:
        return pd.Series(0.0, index=candidates.index)
    risks = []
    for column in profile.continuous_columns:
        low = profile.continuous_min[column]
        high = profile.continuous_max[column]
        risks.append(((candidates[column] < low) | (candidates[column] > high)).astype(float))
    return pd.concat(risks, axis=1).mean(axis=1)


def categorical_risk(profile: TrustProfile, candidates: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return unseen-category and unseen-combination risks."""
    if not profile.categorical_columns:
        zeros = pd.Series(0.0, index=candidates.index)
        return zeros, zeros

    unseen_parts = []
    for column in profile.categorical_columns:
        allowed = profile.categorical_values[column]
        unseen_parts.append(~candidates[column].astype(str).isin(allowed))
    unseen = pd.concat(unseen_parts, axis=1).mean(axis=1).astype(float)

    combo_unseen = []
    for row in candidates[profile.categorical_columns].astype(str).itertuples(index=False, name=None):
        combo_unseen.append(0.0 if tuple(row) in profile.categorical_combinations else 1.0)
    return unseen, pd.Series(combo_unseen, index=candidates.index)


def _normalize_continuous(
    df: pd.DataFrame,
    columns: list[str],
    lower: dict[str, float],
    upper: dict[str, float],
) -> np.ndarray:
    values = []
    for column in columns:
        span = upper[column] - lower[column]
        if abs(span) <= 1e-12:
            values.append(np.zeros(len(df)))
        else:
            values.append((df[column].astype(float).to_numpy() - lower[column]) / span)
    return np.column_stack(values)
