PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup discover train train-lite train-multi train-advanced auto-evolve auto-beat-baseline long-lite stock-deep-dive export-tdx quick quick-lite clean

setup:
	$(PY) -m pip install -U pip wheel setuptools
	$(PIP) install -r requirements.txt

discover:
	$(PY) -m hp_ml.discover --out reports/latest_candidates.csv

train:
	$(PY) -m hp_ml.train --start 20180101 --max-etfs-per-index 3 --horizon 5 --test-days 252

train-lite:
	$(PY) -m hp_ml.lite_train --start 20180101 --max-etfs-per-index 3 --horizon 5 --test-days 252

train-multi:
	$(PY) -m hp_ml.multi_model_train --start 20200101 --models ridge hgb enhanced_rf --horizon 5 --top-k 3

train-advanced:
	$(PY) -m hp_ml.advanced_train --start 20200101 --base-models ridge hgb rf --enable-auto-tuning --enable-ensemble --enable-feature-learning

auto-evolve:
	$(PY) scripts/auto_evolve.py --goal "提升夏普比率至1.5以上" --max-iterations 5

test-multi:
	$(PY) scripts/test_multi_model.py

auto-beat-baseline:
	$(PY) -m hp_ml.auto_tune_baseline --max-model-configs 1

long-lite:
	$(PY) -m hp_ml.lite_train --start 20050101 --max-etfs-per-index 5 --horizon 5 --test-days 504 --scale-horizons 5,20,60,120,250 --related-stocks-per-index 30

stock-deep-dive:
	$(PY) -m hp_ml.stock_deep_dive --top-etfs 8 --stocks-per-family 25 --max-candidates 80 --top-stocks 30

export-tdx:
	$(PY) -m hp_ml.export_tdx --model models/csi_broad_etf_model_lite.pkl --out reports/tdx_formulas --all-families

quick:
	$(PY) -m hp_ml.train --start 20230101 --max-etfs-per-index 2 --horizon 5 --test-days 120

quick-lite:
	$(PY) -m hp_ml.lite_train --start 20230101 --max-etfs-per-index 2 --horizon 5 --test-days 120

clean:
	rm -rf data/processed/*.csv models/*.joblib reports/*.json reports/*.md reports/latest_predictions.csv
