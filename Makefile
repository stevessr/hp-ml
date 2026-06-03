PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup discover train train-lite quick quick-lite clean

setup:
	$(PY) -m pip install -U pip wheel setuptools
	$(PIP) install -r requirements.txt

discover:
	$(PY) -m hp_ml.discover --out reports/latest_candidates.csv

train:
	$(PY) -m hp_ml.train --start 20180101 --max-etfs-per-index 3 --horizon 5 --test-days 252

train-lite:
	$(PY) -m hp_ml.lite_train --start 20180101 --max-etfs-per-index 3 --horizon 5 --test-days 252

quick:
	$(PY) -m hp_ml.train --start 20230101 --max-etfs-per-index 2 --horizon 5 --test-days 120

quick-lite:
	$(PY) -m hp_ml.lite_train --start 20230101 --max-etfs-per-index 2 --horizon 5 --test-days 120

clean:
	rm -rf data/processed/*.csv models/*.joblib reports/*.json reports/*.md reports/latest_predictions.csv
