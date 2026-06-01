# design-space-agent

<p align="center">
  <img src="assets/readme-hero.png" alt="Trust-aware engineering design-space exploration with Pareto front, feasibility, and recommended candidates" width="100%">
</p>

Local toolkit for engineering design-space exploration.

design-space-agent helps engineers explore small and expensive engineering datasets by combining design-space visualization, surrogate model auditing, trust-aware candidate recommendation, Pareto analysis, feasibility checks, and decision-report generation.

The MVP is centered on trust-aware candidate recommendation. It helps answer whether the surrogate model is extrapolating, where the feasible region appears to be, why a candidate is worth reviewing next, and how to explain the decision to engineering stakeholders.

## Concept

**Trust-aware candidate recommendation for early design-space exploration.**

Many early engineering studies start with a small table of experiments, CAE runs, or measurements. A point suggestion is only useful when engineering teams can also inspect extrapolation risk, likely active constraints, and the reasoning behind the next candidate.

The MVP focuses on local, inspectable workflows:

- Load a YAML design-space definition.
- Load CSV data.
- Validate variables, objectives, constraints, and data columns.
- Classify feasible and infeasible samples.
- Extract a direction-aware Pareto front.
- Train simple RandomForest surrogate models.
- Compute a **heuristic trust score** for extrapolation risk.
- Recommend next design candidates with trust-aware scoring.
- Generate static HTML and Markdown reports.

This project does not call external APIs and does not require API keys.

## Install

### uv

```bash
git clone <repo-url>
cd design-space-agent
uv venv
uv pip install -e ".[dev]"
```

### python -m venv + pip

```bash
cd design-space-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Python 3.11 or newer is required.

## Quickstart

```bash
dslab run examples/vehicle_concept/design.yaml --out outputs --n 5
```

Expected generated files:

```text
outputs/
  audit.md
  pareto_front.csv
  recommendations.csv
  report.html
  report.md
  surrogate_metrics.csv
  plots/
    feasible_map.png
    pareto_front.png
    objective_scatter.png
```

The included example datasets are **fully synthetic demo data**. They do not represent real companies, products, measurement programs, CAE studies, or field data.

## YAML Schema Example

```yaml
project:
  name: vehicle_concept_demo
  description: Synthetic early-stage vehicle concept design exploration

data:
  path: samples.csv
  group_column: scenario_id

variables:
  battery_kwh:
    type: continuous
    range: [50, 120]
    unit: kWh

  tire_type:
    type: categorical
    values: [eco, standard, sport]

objectives:
  - name: range_km
    direction: maximize
    unit: km

  - name: cost_jpy
    direction: minimize
    unit: JPY

constraints:
  - name: accel_constraint
    expression: accel_0_100_sec <= 8.0

recommendation:
  candidate_pool_size: 5000
  random_seed: 42
  trust_penalty_weight: 0.35
  feasibility_weight: 0.40
  objective_weight: 0.25
```

Constraint expressions intentionally support only safe numeric comparisons:

- `column <= number`
- `column < number`
- `column >= number`
- `column > number`
- `column == number`

The tool never uses Python `eval` for constraints.

## CLI Commands

```bash
dslab audit examples/vehicle_concept/design.yaml --out outputs
dslab explore examples/vehicle_concept/design.yaml --out outputs
dslab recommend examples/vehicle_concept/design.yaml --n 5 --out outputs
dslab report examples/vehicle_concept/design.yaml --out outputs/report.html
dslab run examples/vehicle_concept/design.yaml --out outputs --n 5
```

## Output Examples

- `audit.md`: row counts, variables, objectives, constraints, missing values, duplicates, range violations, unknown categories, and feasible/infeasible counts.
- `pareto_front.csv`: nondominated rows, computed over feasible samples when feasible samples exist.
- `recommendations.csv`: ranked candidates with objective, feasibility, trust, prediction, and explanation columns.
- `report.html`: static report for engineering review.
- `plots/*.png`: matplotlib visual summaries of feasibility, Pareto front, and objective scatter.

## What This Tool Does

- Provides a local-first MVP for design-space exploration.
- Helps inspect feasibility and Pareto trade-offs.
- Trains simple surrogate models for objectives and constraint targets.
- Scores candidates with a heuristic trust score.
- Produces explainable candidate recommendations and static reports.

## What This Tool Does Not Do

- It is not a full optimization backend.
- It does not provide calibrated uncertainty guarantees.
- It does not prove candidate safety or feasibility.
- It does not use external APIs, LLM APIs, or API keys.
- It does not implement MCP in the MVP.
- It does not include a web dashboard in the MVP.

## Trust Score

The MVP `trust_score` is a **heuristic trust score** in `[0, 1]`.

- `1` means lower extrapolation risk relative to the available training data.
- `0` means high extrapolation risk.

It combines:

- observed training range checks for continuous variables,
- nearest-neighbor distance in continuous design-variable space,
- unseen categorical values,
- unseen categorical combinations.

The score is intended for screening and explanation, not theoretical uncertainty calibration.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Roadmap

- Optional probabilistic search backend.
- Optional integration with established optimization libraries.
- MCP server.
- Agent Skills.
- Streamlit dashboard.
- Uncertainty calibration.
- Mixed-variable recommendation benchmark.
- Better constraint parsers and derived-feature support.
- More robust trust-region and design-of-experiments strategies.

## License

MIT License.
