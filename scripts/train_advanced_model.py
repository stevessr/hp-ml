#!/usr/bin/env python3
"""高级模型训练脚本 - 使用 XGBoost/LightGBM 提升准确度

替换线性 Ridge 模型为更强大的梯度提升树模型
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR

# 尝试导入梯度提升树库
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
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def check_dependencies() -> str:
    """检查可用的模型库并返回推荐模型"""
    if HAS_LIGHTGBM:
        return "lightgbm"
    elif HAS_XGBOOST:
        return "xgboost"
    elif HAS_SKLEARN:
        return "sklearn_gbm"
    else:
        raise RuntimeError(
            "需要安装至少一个梯度提升树库:\n"
            "  pip install lightgbm  # 推荐\n"
            "  pip install xgboost\n"
        )


def load_and_prepare_data(panel_path: Path, target_col: str = "fwd_ret_5") -> tuple[pd.DataFrame, list[str]]:
    """加载并准备训练数据"""
    print("📊 加载训练面板数据...")
    panel_df = pd.read_csv(panel_path)
    print(f"  ✓ 数据行数: {len(panel_df)}")
    print(f"  ✓ 日期范围: {panel_df['date'].min()} ~ {panel_df['date'].max()}")

    # 提取特征列
    exclude_cols = {'date', 'code', 'name', 'family_id', 'close', target_col, 'is_trainable', 'is_future'}
    feature_cols = [c for c in panel_df.columns if c not in exclude_cols and not c.startswith('fwd_')]
    print(f"  ✓ 特征数量: {len(feature_cols)}")

    # 过滤可训练数据并按时间排序
    train_mask = (panel_df['is_trainable'] == True) & (panel_df[target_col].notna())
    train_df = panel_df[train_mask].copy()
    train_df = train_df.sort_values('date').reset_index(drop=True)
    print(f"  ✓ 可训练行数: {len(train_df)}")

    return train_df, feature_cols


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    """训练 LightGBM 模型"""
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 6,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    """训练 XGBoost 模型"""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'eval_metric': 'rmse',
        }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return model


def train_sklearn_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
) -> Any:
    """训练 Sklearn GradientBoosting 模型"""
    if params is None:
        params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'max_features': 0.8,
            'random_state': 42,
        }

    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
) -> dict[str, float]:
    """评估模型性能"""
    # 预测
    if model_type == "lightgbm":
        y_pred = model.predict(X_test)
    elif model_type == "xgboost":
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
    else:
        y_pred = model.predict(X_test)

    # 计算指标
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))

    # 方向准确率
    directional_accuracy = np.mean((y_test >= 0) == (y_pred >= 0))

    # R²
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "directional_accuracy": float(directional_accuracy),
        "r2": float(r2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="高级模型训练")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--model-type", choices=["auto", "lightgbm", "xgboost", "sklearn_gbm"], default="auto")
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "advanced_model.pkl")
    parser.add_argument("--target", default="fwd_ret_5")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 高级模型训练")
    print("="*80)

    # 检查依赖
    if args.model_type == "auto":
        model_type = check_dependencies()
        print(f"✓ 自动选择模型: {model_type}")
    else:
        model_type = args.model_type
        print(f"✓ 使用指定模型: {model_type}")

    # 加载数据
    train_df, feature_cols = load_and_prepare_data(args.panel, args.target)

    # 准备训练测试集（按时间序列划分）
    split_idx = int(len(train_df) * (1 - args.test_ratio))

    train_data = train_df.iloc[:split_idx]
    test_data = train_df.iloc[split_idx:]

    print(f"\n📊 数据划分:")
    print(f"  训练集: {len(train_data)} 行")
    print(f"  测试集: {len(test_data)} 行")

    # 准备特征和标签
    X_train = train_data[feature_cols].fillna(0).values
    y_train = train_data[args.target].values
    X_test = test_data[feature_cols].fillna(0).values
    y_test = test_data[args.target].values

    # 训练模型
    print(f"\n🔨 训练 {model_type} 模型...")
    if model_type == "lightgbm":
        model = train_lightgbm(X_train, y_train)
    elif model_type == "xgboost":
        model = train_xgboost(X_train, y_train)
    else:
        model = train_sklearn_gbm(X_train, y_train)

    print("✓ 模型训练完成")

    # 评估
    print("\n📈 评估模型性能...")
    train_metrics = evaluate_model(model, X_train, y_train, model_type)
    test_metrics = evaluate_model(model, X_test, y_test, model_type)

    print("\n" + "="*80)
    print("📊 模型性能")
    print("="*80)
    print(f"\n训练集:")
    print(f"  RMSE: {train_metrics['rmse']:.6f}")
    print(f"  MAE: {train_metrics['mae']:.6f}")
    print(f"  方向准确率: {train_metrics['directional_accuracy']:.4f} ({train_metrics['directional_accuracy']*100:.2f}%)")
    print(f"  R²: {train_metrics['r2']:.4f}")

    print(f"\n测试集:")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    print(f"  方向准确率: {test_metrics['directional_accuracy']:.4f} ({test_metrics['directional_accuracy']*100:.2f}%)")
    print(f"  R²: {test_metrics['r2']:.4f}")

    # 保存模型
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_package = {
        'model': model,
        'model_type': model_type,
        'feature_cols': feature_cols,
        'target_col': args.target,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
    }

    with open(args.output, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"\n💾 模型已保存: {args.output}")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
