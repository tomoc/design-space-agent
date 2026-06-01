from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from dslab.audit.constraints import constraint_target_columns
from dslab.models.metrics import regression_metrics
from dslab.models.preprocessing import build_preprocessor
from dslab.schema.design_schema import DesignSpec


@dataclass
class SurrogateResult:
    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    target_columns: list[str]


def surrogate_target_columns(spec: DesignSpec) -> list[str]:
    """Return objective and constraint target columns to be modeled."""
    targets: list[str] = []
    for name in spec.objective_names + constraint_target_columns(spec.constraints):
        if name not in targets:
            targets.append(name)
    return targets


def train_surrogates(spec: DesignSpec, df: pd.DataFrame) -> SurrogateResult:
    """Train one RandomForestRegressor surrogate per objective/constraint target."""
    feature_columns = spec.variable_names
    target_columns = surrogate_target_columns(spec)
    clean = df.dropna(subset=feature_columns + target_columns).reset_index(drop=True)
    if len(clean) < 4:
        raise ValueError("At least four complete rows are required to train surrogate models.")

    train_idx, test_idx = _split_indices(spec, clean)
    x_train = clean.loc[train_idx, feature_columns]
    x_test = clean.loc[test_idx, feature_columns]

    models: dict[str, Pipeline] = {}
    metric_rows: list[dict[str, float | str]] = []
    random_state = spec.recommendation.random_seed

    for target in target_columns:
        y_train = clean.loc[train_idx, target]
        y_test = clean.loc[test_idx, target]
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(spec)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        metrics = regression_metrics(y_test.to_numpy(), pred)
        metric_rows.append({"target": target, **metrics})
        models[target] = pipeline

    return SurrogateResult(
        models=models,
        metrics=pd.DataFrame(metric_rows),
        target_columns=target_columns,
    )


def predict_surrogates(result: SurrogateResult, candidates: pd.DataFrame) -> pd.DataFrame:
    """Predict all surrogate targets for a candidate table."""
    predictions = candidates.copy()
    for target, model in result.models.items():
        predictions[f"pred_{target}"] = model.predict(candidates)
    return predictions


def _split_indices(spec: DesignSpec, df: pd.DataFrame) -> tuple[list[int], list[int]]:
    indices = list(range(len(df)))
    if spec.data.group_column and spec.data.group_column in df.columns:
        groups = df[spec.data.group_column]
        if groups.nunique(dropna=True) >= 2:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=0.25,
                random_state=spec.recommendation.random_seed,
            )
            train_idx, test_idx = next(splitter.split(df, groups=groups))
            return train_idx.tolist(), test_idx.tolist()

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=spec.recommendation.random_seed,
    )
    return list(train_idx), list(test_idx)
