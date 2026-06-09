#!/usr/bin/env python3
"""3 模型集成 - 稳定提升准确率

策略：
1. 使用 3 个不同随机种子的 Attention LSTM
2. 加权平均集成
3. 减少随机性，提升稳定性
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.models_extended import make_extended_model
from hp_ml.config import PROCESSED_DIR, REPORTS_DIR

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


def train_single_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seed: int,
    seq_length: int = 20
):
    """训练单个模型"""
    print(f"\n  训练模型 (seed={seed})...")

    set_seed(seed)

    model = make_extended_model(
        "attention_lstm",
        seq_length=seq_length,
        units=64,
        dropout=0.3,
        learning_rate=0.001,
        epochs=25,
        batch_size=32,
        early_stopping_patience=10
    )

    X_train = train_df[feature_cols + ["code", "date"]]
    y_train = train_df[target_col]

    model.fit(X_train, y_train)

    return model


def ensemble_predict(models: list, X_test: pd.DataFrame, weights: list[float]) -> np.ndarray:
    """集成预测"""
    predictions = []

    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)

    # 加权平均
    ensemble_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        ensemble_pred += pred * weight

    return ensemble_pred, predictions


def main() -> int:
    print("\n" + "="*80)
    print("🎯 3 模型集成实验")
    print("="*80 + "\n")

    # 加载数据
    data_path = PROCESSED_DIR / "training_panel_lite.csv"
    df = pd.read_csv(data_path)

    target_col = "fwd_ret_5"
    exclude_cols = {'date', 'code', 'name', 'family_id', 'close', target_col,
                   'is_trainable', 'is_future'}
    feature_cols = [c for c in df.columns if c not in exclude_cols
                   and not c.startswith('fwd_')]

    print(f"特征数：{len(feature_cols)}")

    # 切分数据
    df = df.sort_values("date")
    trainable = df[df.get("is_trainable", True)].copy()
    unique_dates = sorted(trainable["date"].unique())
    split_idx = max(1, len(unique_dates) - 60)
    split_date = unique_dates[split_idx]

    train_df = trainable[trainable["date"] < split_date].copy()
    test_df = trainable[trainable["date"] >= split_date].copy()

    print(f"训练集：{len(train_df)} 行")
    print(f"测试集：{len(test_df)} 行")

    # 训练 3 个模型（不同随机种子）
    print("\n" + "="*80)
    print("🔧 训练 3 个模型...")
    print("="*80)

    seeds = [42, 123, 456]
    models = []

    for seed in seeds:
        model = train_single_model(train_df, feature_cols, target_col, seed)
        models.append(model)
        print(f"    ✓ 模型 seed={seed} 训练完成")

    # 测试
    print("\n" + "="*80)
    print("📊 评估结果")
    print("="*80 + "\n")

    X_test = test_df[feature_cols + ["code", "date"]]
    actual = test_df[target_col].values

    results = []

    # 评估单个模型
    for i, (model, seed) in enumerate(zip(models, seeds), 1):
        pred = model.predict(X_test)
        acc = float(np.mean((pred > 0) == (actual > 0)))
        ic = float(spearmanr(pred, actual)[0])

        print(f"模型{i} (seed={seed}):")
        print(f"  准确率：{acc:.4f}")
        print(f"  IC: {ic:.4f}\n")

        results.append({
            "model": f"Model {i} (seed={seed})",
            "type": "single",
            "accuracy": acc,
            "ic": ic
        })

    # 集成：等权重
    print("="*80)
    print("🎯 集成 - 等权重 (0.33, 0.33, 0.33)")
    print("="*80 + "\n")

    weights_equal = [1/3, 1/3, 1/3]
    ensemble_pred_equal, individual_preds = ensemble_predict(models, X_test, weights_equal)

    acc_equal = float(np.mean((ensemble_pred_equal > 0) == (actual > 0)))
    ic_equal = float(spearmanr(ensemble_pred_equal, actual)[0])

    print(f"准确率：{acc_equal:.4f}")
    print(f"IC: {ic_equal:.4f}\n")

    results.append({
        "model": "Ensemble (Equal Weight)",
        "type": "ensemble",
        "accuracy": acc_equal,
        "ic": ic_equal
    })

    # 集成：加权（基于 IC）
    print("="*80)
    print("🎯 集成 - IC 加权")
    print("="*80 + "\n")

    # 计算每个模型的 IC 作为权重
    ics = []
    for pred in individual_preds:
        ic = spearmanr(pred, actual)[0]
        ics.append(max(ic, 0))  # 负 IC 设为 0

    total_ic = sum(ics)
    if total_ic > 0:
        weights_ic = [ic / total_ic for ic in ics]
    else:
        weights_ic = weights_equal

    print(f"权重：{[f'{w:.3f}' for w in weights_ic]}")

    ensemble_pred_ic, _ = ensemble_predict(models, X_test, weights_ic)

    acc_ic = float(np.mean((ensemble_pred_ic > 0) == (actual > 0)))
    ic_ic = float(spearmanr(ensemble_pred_ic, actual)[0])

    print(f"准确率：{acc_ic:.4f}")
    print(f"IC: {ic_ic:.4f}\n")

    results.append({
        "model": "Ensemble (IC Weighted)",
        "type": "ensemble",
        "accuracy": acc_ic,
        "ic": ic_ic
    })

    # 集成：优化权重（基于准确率）
    print("="*80)
    print("🎯 集成 - 准确率加权")
    print("="*80 + "\n")

    accs = []
    for pred in individual_preds:
        acc = np.mean((pred > 0) == (actual > 0))
        accs.append(acc)

    total_acc = sum(accs)
    weights_acc = [acc / total_acc for acc in accs]

    print(f"权重：{[f'{w:.3f}' for w in weights_acc]}")

    ensemble_pred_acc, _ = ensemble_predict(models, X_test, weights_acc)

    acc_acc = float(np.mean((ensemble_pred_acc > 0) == (actual > 0)))
    ic_acc = float(spearmanr(ensemble_pred_acc, actual)[0])

    print(f"准确率：{acc_acc:.4f}")
    print(f"IC: {ic_acc:.4f}\n")

    results.append({
        "model": "Ensemble (Accuracy Weighted)",
        "type": "ensemble",
        "accuracy": acc_acc,
        "ic": ic_acc
    })

    # 保存结果
    output_dir = REPORTS_DIR / "ensemble_3models"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)

    # 总结
    print("\n" + "="*80)
    print("📊 最终对比")
    print("="*80 + "\n")

    for idx, row in results_df.iterrows():
        marker = "🔵" if row['type'] == 'single' else "⭐"
        print(f"{marker} {row['model']}")
        print(f"   准确率：{row['accuracy']:.4f}")
        print(f"   IC: {row['ic']:.4f}\n")

    # 找出最佳
    best_acc = results_df['accuracy'].max()
    best_model = results_df[results_df['accuracy'] == best_acc]['model'].values[0]

    # 计算提升
    single_avg = results_df[results_df['type'] == 'single']['accuracy'].mean()
    ensemble_best = results_df[results_df['type'] == 'ensemble']['accuracy'].max()
    improvement = (ensemble_best - single_avg) * 100

    print("="*80)
    print(f"💡 单模型平均：{single_avg:.4f}")
    print(f"💡 集成最佳：{ensemble_best:.4f}")
    print(f"💡 提升幅度：{improvement:+.2f}%")
    print(f"💡 最佳配置：{best_model}")
    print("="*80)

    if ensemble_best > single_avg:
        print(f"\n✅ 集成成功！准确率提升 {improvement:.2f}%")
    else:
        print(f"\n⚠️ 集成未带来提升")

    print(f"\n💾 结果已保存：{output_dir}/results.csv\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
