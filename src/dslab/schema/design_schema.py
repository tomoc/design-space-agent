from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

VariableType = Literal["continuous", "categorical"]
ObjectiveDirection = Literal["maximize", "minimize"]


class ProjectSpec(BaseModel):
    name: str
    description: str = ""


class DataSpec(BaseModel):
    path: str
    group_column: str | None = None


class VariableSpec(BaseModel):
    type: VariableType
    range: tuple[float, float] | None = None
    values: list[str] | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def validate_variable_shape(self) -> VariableSpec:
        if self.type == "continuous":
            if self.range is None:
                msg = "continuous variables require range: [min, max]"
                raise ValueError(msg)
            lower, upper = self.range
            if lower >= upper:
                msg = "continuous variable range must satisfy min < max"
                raise ValueError(msg)
        if self.type == "categorical":
            if not self.values:
                msg = "categorical variables require a non-empty values list"
                raise ValueError(msg)
            if len(set(self.values)) != len(self.values):
                msg = "categorical variable values must be unique"
                raise ValueError(msg)
        return self


class ObjectiveSpec(BaseModel):
    name: str
    direction: ObjectiveDirection
    unit: str | None = None


class ConstraintSpec(BaseModel):
    name: str
    expression: str


class RecommendationSpec(BaseModel):
    candidate_pool_size: int = Field(default=5000, ge=1)
    random_seed: int = 42
    trust_penalty_weight: float = Field(default=0.35, ge=0.0)
    feasibility_weight: float = Field(default=0.40, ge=0.0)
    objective_weight: float = Field(default=0.25, ge=0.0)


class DesignSpec(BaseModel):
    project: ProjectSpec
    data: DataSpec
    variables: dict[str, VariableSpec]
    objectives: list[ObjectiveSpec]
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    recommendation: RecommendationSpec = Field(default_factory=RecommendationSpec)

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, value: dict[str, VariableSpec]) -> dict[str, VariableSpec]:
        if not value:
            msg = "at least one design variable is required"
            raise ValueError(msg)
        return value

    @field_validator("objectives")
    @classmethod
    def validate_objectives(cls, value: list[ObjectiveSpec]) -> list[ObjectiveSpec]:
        if not value:
            msg = "at least one objective is required"
            raise ValueError(msg)
        names = [objective.name for objective in value]
        if len(set(names)) != len(names):
            msg = "objective names must be unique"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_weight_sum(self) -> DesignSpec:
        rec = self.recommendation
        total = rec.trust_penalty_weight + rec.feasibility_weight + rec.objective_weight
        if total <= 0:
            msg = "at least one recommendation weight must be positive"
            raise ValueError(msg)
        return self

    @property
    def variable_names(self) -> list[str]:
        return list(self.variables)

    @property
    def continuous_variable_names(self) -> list[str]:
        return [name for name, spec in self.variables.items() if spec.type == "continuous"]

    @property
    def categorical_variable_names(self) -> list[str]:
        return [name for name, spec in self.variables.items() if spec.type == "categorical"]

    @property
    def objective_names(self) -> list[str]:
        return [objective.name for objective in self.objectives]
