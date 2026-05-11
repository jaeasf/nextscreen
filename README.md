# NEXTscreen

**Screen. Understand. Optimize.**

NEXTscreen is an experiment screening and Bayesian optimization toolkit for
chemistry and chemical engineering. It guides experimentalists through the
full workflow — from raw data to optimized experiment planning — via a
point-and-click browser interface and a Python API.

## Features

- **Data upload** — CSV and Excel with automatic replicate detection and
  averaging
- **Feature selection** — LASSO, Random Forest, SHAP, PCA, Pearson/Spearman
  correlations, and ARD-GP lengthscales; categorical features assessed with
  one-way ANOVA (η²) rather than spurious r-values
- **Consensus ranking** — aggregate importance scores across all methods with
  plain-English interpretation (no LLM required)
- **Single-objective BO** — powered by NEXTorch (EI / qEI); supports
  continuous, categorical, and integer (ordinal) parameters; returns
  uncertainty estimates
- **Multi-objective BO (Pareto front)** — q-Expected Hypervolume Improvement
  (qEHVI) via BoTorch; discovers the full yield–selectivity trade-off without
  requiring weight specification; per-objective uncertainty columns included
- **Discrete parameter handling** — categorical and integer parameters are
  enumerated exactly, guaranteeing experimentally feasible suggestions
- **Report export** — self-contained HTML (and PDF) report including dataset
  summary, all feature-selection results, BO bounds, and suggestions from
  both strategies

## Installation

```bash
pip install nextscreen
```

## Quick start

```bash
nextscreen          # launches the Streamlit app in your browser
```

Or use the Python API directly:

```python
from nextscreen.features.correlations import run_correlations
from nextscreen.nextorch_integration.handoff import (
    build_parameter_space, run_optimization, run_pareto_optimization,
)
```

See `examples/tutorial.ipynb` for a complete walkthrough.

## Development

```bash
git clone https://github.com/VlachosGroup/nextscreen
cd nextscreen
pip install -e ".[dev]"
pytest
```

## Tech stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| ML / feature selection | scikit-learn, shap, scipy |
| Single-objective BO | NEXTorch |
| Multi-objective BO | BoTorch (qEHVI), GPyTorch |
| Plotting | Plotly |
| Report generation | WeasyPrint |

## License

MIT
