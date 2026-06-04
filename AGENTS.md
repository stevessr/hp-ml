# Repository Guidelines

## Project Structure & Module Organization

`hp_ml/` contains the importable Python package for CSI broad ETF discovery, data loading, feature engineering, training, reporting, and charts. Main entry points are `hp_ml.train`, `hp_ml.lite_train`, `hp_ml.discover`, and `hp_ml.auto_tune_baseline`. `scripts/` holds exploratory and strategy/backtest runners. `tests/` contains pytest unit tests. `data/` stores raw, processed, alternative, and topic-specific datasets; avoid committing large regenerated data unless intentionally required. `models/` stores trained artifacts, `reports/` stores generated CSV/JSON/Markdown/SVG outputs, `docs/` holds notes, and `ref/` contains external reference material.

## Build, Test, and Development Commands

- `make setup` installs runtime dependencies into `.venv` from `requirements.txt`.
- `make discover` writes the latest ETF candidate pool to `reports/latest_candidates.csv`.
- `make quick` runs a short full training smoke run from 2023 data.
- `make train` runs the standard pandas/scikit-learn training pipeline.
- `make train-lite` or `python -m hp_ml.lite_train ...` runs the lower-dependency fallback pipeline.
- `python -m pytest` runs the unit test suite.
- `python -m py_compile hp_ml/*.py tests/*.py` is a quick syntax check before committing.

## Coding Style & Naming Conventions

Use Python 3.10+ and PEP 8 conventions: 4-space indentation, snake_case functions and modules, PascalCase classes, and UPPER_CASE constants. Keep CLI parsers explicit and prefer `pathlib.Path` for project paths. Add type hints for new public helpers and keep data-processing functions deterministic where possible. Generated outputs should use descriptive suffixes such as `_lite.csv`, `_summary.json`, or `_chart.svg`.

## Testing Guidelines

Tests use pytest and live in `tests/test_*.py`. Name test functions `test_<behavior>` and prefer compact fixtures or inline DataFrames/lists over network-dependent tests. Cover classification rules, strategy math, date boundaries, and pure transformation helpers. For data-source changes, mock HTTP responses or isolate cache behavior rather than requiring live market APIs.

## Commit & Pull Request Guidelines

Recent commit history is terse and inconsistent, so use clearer future commits: `fix: handle A500 classification`, `test: cover strategy threshold`, or `docs: update training outputs`. Pull requests should describe the research or pipeline impact, list commands run, note any changed data/model/report artifacts, and include screenshots only when chart/report visuals changed.

## Security & Configuration Tips

Do not commit credentials, Kaggle tokens, or local `.venv` files. Treat generated investment outputs as research artifacts, not financial advice. Large CSV/model updates should be intentional and called out in the PR.
