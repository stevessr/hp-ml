#!/usr/bin/env python3
"""时间序列交叉验证超参数调优脚本。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import PROCESSED_DIR, REPORTS_DIR

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def build_time_splits(df: pd.DataFrame, n_splits: int = 5) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    min_train_size = max(int(len(df) * 0.5), 1)
    test_size = max(int(len(df) * 0.15), 1)
    max_start = len(df) - test_size

    if max_start <= min_train_size:
        raise ValueError("数据不足以构造时间序列交叉验证切分")

    if n_splits == 1:
        train_end = max_start
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:train_end + test_size]
        return [(train_df, test_df)] if len(train_df) and len(test_df) else []

    step = max((max_start - min_train_size) // (n_splits - 1), 1)
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for i in range(n_splits):
        train_end = min(min_train_size + i * step, max_start)
        test_end = min(train_end + test_size, len(df))
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        if len(train_df) > 0 and len(test_df) > 0:
            splits.append((train_df, test_df))

    return splits


def _prepare_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = (df[target_col].fillna(0).values >= 0).astype(int)
    return X, y


def _evaluate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, (y_pred >= 0).astype(int)))


def train_eval_lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str, params: dict[str, Any]) -> float:
    X_train, y_train = _prepare_xy(train_df, feature_cols, target_col)
    X_test, y_test = _prepare_xy(test_df, feature_cols, target_col)

    train_set = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[train_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    y_pred = model.predict(X_test)
    return _evaluate_directional_accuracy(y_test, y_pred)


def train_eval_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str, params: dict[str, Any]) -> float:
    X_train, y_train = _prepare_xy(train_df, feature_cols, target_col)
    X_test, y_test = _prepare_xy(test_df, feature_cols, target_col)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    y_pred = model.predict(dtest)
    return _evaluate_directional_accuracy(y_test, y_pred)


def train_eval_sklearn_gbm(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str, params: dict[str, Any]) -> float:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is not installed")

    X_train, y_train = _prepare_xy(train_df, feature_cols, target_col)
    X_test, y_test = _prepare_xy(test_df, feature_cols, target_col)

    model = GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 3),
        subsample=params.get("subsample", 1.0),
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    return _evaluate_directional_accuracy(y_test, y_pred)


def build_param_grid(model_type: str) -> dict[str, list[Any]]:
    if model_type == "lightgbm":
        return {
            "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_child_samples": [5, 10, 20, 50],
            "bagging_fraction": [0.6, 0.7, 0.8, 0.9],
            "feature_fraction": [0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0, 0.01, 0.1, 1],
        }
    if model_type == "xgboost":
        return {
            "max_depth": [15, 31, 63, 127],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_child_weight": [5, 10, 20, 50],
            "subsample": [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0, 0.01, 0.1, 1],
        }
    return {
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "n_estimators": [100, 200, 300, 500],
    }


def search_configurations(
    panel_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_type: str,
    n_splits: int = 5,
    search_mode: str = "grid",
    max_configs: int | None = None,
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    splits = build_time_splits(panel_df, n_splits=n_splits)
    if not splits:
        raise ValueError("无法生成有效的时间序列切分")

    param_grid = build_param_grid(model_type)
    param_names = list(param_grid.keys())
    all_combos = list(product(*param_grid.values()))

    if search_mode == "random" and max_configs is not None and max_configs < len(all_combos):
        rng = random.Random(random_seed)
        all_combos = rng.sample(all_combos, max_configs)
    elif max_configs is not None and max_configs < len(all_combos):
        all_combos = all_combos[:max_configs]

    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for idx, combo in enumerate(all_combos, 1):
        combo_params = dict(zip(param_names, combo))
        if model_type == "lightgbm":
            params = {
                **combo_params,
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "seed": random_seed,
                "bagging_freq": 1,
            }
            trainer = train_eval_lightgbm
        elif model_type == "xgboost":
            params = {
                **combo_params,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "seed": random_seed,
                "verbosity": 0,
            }
            trainer = train_eval_xgboost
        else:
            params = {**combo_params}
            trainer = train_eval_sklearn_gbm

        fold_scores = []
        for train_df, test_df in splits:
            score = trainer(train_df, test_df, feature_cols, target_col, params)
            fold_scores.append(score)

        mean_accuracy = float(np.mean(fold_scores))
        std_accuracy = float(np.std(fold_scores))
        record = {
            "params": combo_params,
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "fold_accuracies": [float(s) for s in fold_scores],
        }
        results.append(record)

        if best_result is None or mean_accuracy > best_result["mean_accuracy"]:
            best_result = record
            print(f"  ✓ 配置 {idx}/{len(all_combos)}: {mean_accuracy:.4f} (±{std_accuracy:.4f}) [NEW BEST]")
        elif idx % 10 == 0:
            print(f"    配置 {idx}/{len(all_combos)}: {mean_accuracy:.4f} (±{std_accuracy:.4f})")

    assert best_result is not None
    results_df = pd.DataFrame(results).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
    return best_result["params"], results_df


def main() -> int:
    parser = argparse.ArgumentParser(description="sklearn_gbm/LightGBM/XGBoost 时间序列超参数调优")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--model-type", choices=["sklearn_gbm", "lightgbm", "xgboost"], default="sklearn_gbm")
    parser.add_argument("--target", default="fwd_ret_5")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--search-mode", choices=["grid", "random"], default="grid")
    parser.add_argument("--max-configs", type=int, default=128)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "hyperparameter_tune_results.json")
    parser.add_argument("--csv-output", type=Path, default=REPORTS_DIR / "hyperparameter_tune_results.csv")
    args = parser.parse_args()

    if args.model_type == "lightgbm" and not HAS_LIGHTGBM:
        print("❌ LightGBM 未安装")
        return 1
    if args.model_type == "xgboost" and not HAS_XGBOOST:
        print("❌ XGBoost 未安装")
        return 1
    if args.model_type == "sklearn_gbm" and not HAS_SKLEARN:
        print("❌ scikit-learn 未安装")
        return 1

    panel_df = pd.read_csv(args.panel)
    panel_df = panel_df[panel_df[args.target].notna()].copy()

    exclude_cols = {"date", "code", "name", "family_id", "close", args.target, "is_trainable", "is_future"}
    feature_cols = [c for c in panel_df.columns if c not in exclude_cols and not c.startswith("fwd_")]

    print("=" * 80)
    print("超参数调优")
    print("=" * 80)
    print(f"数据路径：{args.panel}")
    print(f"模型类型：{args.model_type}")
    print(f"特征数量：{len(feature_cols)}")
    print(f"交叉验证折数：{args.n_splits}")

    best_params, results_df = search_configurations(
        panel_df=panel_df,
        feature_cols=feature_cols,
        target_col=args.target,
        model_type=args.model_type,
        n_splits=args.n_splits,
        search_mode=args.search_mode,
        max_configs=args.max_configs,
    )

    best_row = results_df.iloc[0].to_dict()
    best_accuracy = float(best_row["mean_accuracy"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(args.csv_output, index=False)
    payload = {
        "model_type": args.model_type,
        "best_params": best_params,
        "best_accuracy": best_accuracy,
        "target_accuracy": 0.80,
        "meets_target": best_accuracy >= 0.80,
        "top_results": results_df.head(20).to_dict(orient="records"),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("最佳超参数配置")
    print("=" * 80)
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print(f"平均测试准确率：{best_accuracy:.4f}")
    if best_accuracy >= 0.80:
        print("已达到 80% 目标")
    else:
        print(f"距离 80% 目标还差 {(0.80 - best_accuracy):.4f}")
    print(f"结果已保存到：{args.output}")
    print(f"详细结果已保存到：{args.csv_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
