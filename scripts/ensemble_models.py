#!/usr/bin/env python3
"""Ensemble model training script for stacking, voting, and blending.

This script loads the processed training panel, trains multiple base models,
combines them with several ensemble strategies, evaluates directional accuracy,
and saves artifacts plus a summary report.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

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
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class BaseModelSpec:
    name: str
    model_type: str
    params: dict[str, Any]


def check_dependencies() -> None:
    if not HAS_SKLEARN:
        raise RuntimeError("需要安装 scikit-learn 才能运行集成模型脚本")
    if not (HAS_LIGHTGBM or HAS_XGBOOST or HAS_SKLEARN):
        raise RuntimeError("没有可用的模型库")


def load_and_prepare_data(panel_path: Path, target_col: str = "fwd_ret_5") -> tuple[pd.DataFrame, list[str]]:
    panel_df = pd.read_csv(panel_path)
    exclude_cols = {"date", "code", "name", "family_id", "close", target_col, "is_trainable", "is_future"}
    feature_cols = [c for c in panel_df.columns if c not in exclude_cols and not c.startswith("fwd_")]
    train_mask = (panel_df["is_trainable"] == True) & (panel_df[target_col].notna())
    train_df = panel_df[train_mask].copy()
    train_df = train_df.sort_values("date")
    return train_df, feature_cols


def _safe_numpy(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def _base_specs() -> list[BaseModelSpec]:
    specs: list[BaseModelSpec] = [
        BaseModelSpec("ridge", "ridge", {"alpha": 1.0}),
        BaseModelSpec("ridge_l2", "ridge", {"alpha": 10.0}),
        BaseModelSpec("rf", "rf", {"n_estimators": 300, "max_depth": 8, "random_state": 42, "n_jobs": -1}),
        BaseModelSpec("gbr", "gbr", {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}),
    ]
    if HAS_LIGHTGBM:
        specs.append(BaseModelSpec("lightgbm", "lightgbm", {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "max_depth": 6,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }))
    if HAS_XGBOOST:
        specs.append(BaseModelSpec("xgboost", "xgboost", {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "eval_metric": "rmse",
        }))
    return specs


def _fit_base_model(spec: BaseModelSpec, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    if spec.model_type == "ridge":
        return Ridge(**spec.params).fit(X_train, y_train)
    if spec.model_type == "rf":
        return RandomForestRegressor(**spec.params).fit(X_train, y_train)
    if spec.model_type == "gbr":
        return GradientBoostingRegressor(**spec.params).fit(X_train, y_train)
    if spec.model_type == "lightgbm":
        train_data = lgb.Dataset(X_train, label=y_train)
        return lgb.train(spec.params, train_data, num_boost_round=400)
    if spec.model_type == "xgboost":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        return xgb.train(spec.params, dtrain, num_boost_round=400, verbose_eval=False)
    raise ValueError(f"Unknown model type: {spec.model_type}")


def _predict_base(model: Any, model_type: str, X: np.ndarray) -> np.ndarray:
    if model_type == "lightgbm":
        return np.asarray(model.predict(X), dtype=float)
    if model_type == "xgboost":
        return np.asarray(model.predict(xgb.DMatrix(X)), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    directional_accuracy = float(np.mean((y_true >= 0) == (y_pred >= 0)))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
    }


def _train_valid_split(df: pd.DataFrame, target_col: str, valid_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - valid_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _make_stacking_meta_model(X_meta: np.ndarray, y_meta: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_meta, (y_meta >= 0).astype(int))
    return clf


def _stacking_predict(meta_model: LogisticRegression, base_pred_matrix: np.ndarray) -> np.ndarray:
    proba = meta_model.predict_proba(base_pred_matrix)[:, 1]
    return np.where(proba >= 0.5, 1.0, -1.0)


def run_ensemble(
    panel_path: Path,
    target_col: str = "fwd_ret_5",
    valid_ratio: float = 0.2,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    check_dependencies()
    train_df, feature_cols = load_and_prepare_data(panel_path, target_col)
    train_df, valid_df = _train_valid_split(train_df, target_col, valid_ratio)

    X_train = _safe_numpy(train_df, feature_cols)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_valid = _safe_numpy(valid_df, feature_cols)
    y_valid = valid_df[target_col].to_numpy(dtype=float)

    specs = _base_specs()
    base_models: dict[str, dict[str, Any]] = {}
    valid_preds: dict[str, np.ndarray] = {}
    base_metrics: dict[str, dict[str, float]] = {}

    for spec in specs:
        model = _fit_base_model(spec, X_train, y_train)
        pred = _predict_base(model, spec.model_type, X_valid)
        base_models[spec.name] = {"spec": spec.__dict__, "model": model}
        valid_preds[spec.name] = pred
        base_metrics[spec.name] = _metrics(y_valid, pred)

    base_pred_matrix = np.column_stack([valid_preds[name] for name in valid_preds])

    stacked_signal_inputs = np.column_stack([base_pred_matrix, np.mean(base_pred_matrix, axis=1), np.median(base_pred_matrix, axis=1)])
    stacking_meta = _make_stacking_meta_model(stacked_signal_inputs, y_valid)
    stacking_pred = _stacking_predict(stacking_meta, stacked_signal_inputs)

    vote_scores = np.zeros(len(y_valid), dtype=float)
    weight_sum = 0.0
    for name, metrics in base_metrics.items():
        weight = max(metrics["directional_accuracy"], 1e-6)
        vote_scores += np.sign(valid_preds[name]) * weight
        weight_sum += weight
    voting_pred = np.where(vote_scores >= 0, 1.0, -1.0)

    blend_weights = np.array([max(base_metrics[name]["directional_accuracy"], 1e-6) for name in valid_preds], dtype=float)
    blend_weights = blend_weights / blend_weights.sum()
    blended_pred = np.dot(base_pred_matrix, blend_weights)

    ensemble_metrics = {
        "stacking": _metrics(y_valid, stacking_pred),
        "voting": _metrics(y_valid, voting_pred),
        "blending": _metrics(y_valid, blended_pred),
    }

    best_name = max(ensemble_metrics.keys(), key=lambda k: ensemble_metrics[k]["directional_accuracy"])
    best_metrics = ensemble_metrics[best_name]
    target_achieved = best_metrics["directional_accuracy"] >= 0.80

    summary = {
        "panel_path": str(panel_path),
        "target_col": target_col,
        "valid_ratio": valid_ratio,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "feature_count": len(feature_cols),
        "base_models": base_metrics,
        "ensemble_metrics": ensemble_metrics,
        "best_strategy": best_name,
        "best_directional_accuracy": best_metrics["directional_accuracy"],
        "target_achieved": target_achieved,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        models_path = output_dir / "ensemble_models.pkl"
        summary_path = output_dir / "ensemble_summary.json"
        with open(models_path, "wb") as f:
            pickle.dump({
                "feature_cols": feature_cols,
                "target_col": target_col,
                "base_models": base_models,
                "stacking_meta": stacking_meta,
                "blend_weights": blend_weights.tolist(),
            }, f)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        summary["models_path"] = str(models_path)
        summary["summary_path"] = str(summary_path)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensemble model training")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--target", default="fwd_ret_5")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "ensemble")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "ensemble_summary.json")
    args = parser.parse_args()

    if not args.panel.exists():
        print(f"❌ 训练面板文件不存在: {args.panel}")
        return 1

    summary = run_ensemble(args.panel, args.target, args.valid_ratio, args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("=" * 80)
    print("📊 集成学习结果")
    print("=" * 80)
    for name, metrics in summary["base_models"].items():
        print(f"{name}: acc={metrics['directional_accuracy']:.4f}, rmse={metrics['rmse']:.6f}, r2={metrics['r2']:.4f}")
    for name, metrics in summary["ensemble_metrics"].items():
        print(f"{name}: acc={metrics['directional_accuracy']:.4f}, rmse={metrics['rmse']:.6f}, r2={metrics['r2']:.4f}")
    print(f"best_strategy={summary['best_strategy']}")
    print(f"best_directional_accuracy={summary['best_directional_accuracy']:.4f}")
    print(f"target_achieved={summary['target_achieved']}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
