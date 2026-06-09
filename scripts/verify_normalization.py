#!/usr/bin/env python3
"""在优化数据上验证性能

对比三个数据集的性能：
1. 原始数据（38 特征）
2. 过度归一化数据（50 特征）
3. 优化归一化数据（33 特征）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.models_extended import make_extended_model
from hp_ml.config import PROCESSED_DIR, REPORTS_DIR


def train_and_evaluate(
    data_path: Path,
    data_name: str,
    seq_length: int = 20,
    test_days: int = 60
) -> dict:
    """训练并评估模型"""

    print(f"\n{'='*80}")
    print(f"📊 测试数据集：{data_name}")
    print(f"{'='*80}")

    # 加载数据
    df = pd.read_csv(data_path)
    target_col = "fwd_ret_5"
    exclude_cols = {'date', 'code', 'name', 'family_id', 'close', target_col,
                   'is_trainable', 'is_future'}
    feature_cols = [c for c in df.columns if c not in exclude_cols
                   and not c.startswith('fwd_')]

    print(f"  特征数：{len(feature_cols)}")

    # 切分数据
    df = df.sort_values("date")
    trainable = df[df.get("is_trainable", True)].copy()
    unique_dates = sorted(trainable["date"].unique())
    split_idx = max(1, len(unique_dates) - test_days)
    split_date = unique_dates[split_idx]

    train_df = trainable[trainable["date"] < split_date].copy()
    test_df = trainable[trainable["date"] >= split_date].copy()

    print(f"  训练集：{len(train_df)} 行")
    print(f"  测试集：{len(test_df)} 行")

    # 训练模型
    print(f"\n  🔧 训练 Attention LSTM (seq={seq_length})...")

    model = make_extended_model(
        "attention_lstm",
        seq_length=seq_length,
        units=64,
        dropout=0.3,
        learning_rate=0.001,
        epochs=20,
        batch_size=32,
        early_stopping_patience=8
    )

    X_train = train_df[feature_cols + ["code", "date"]]
    y_train = train_df[target_col]

    model.fit(X_train, y_train)

    # 评估
    test_pred = model.predict(test_df[feature_cols + ["code", "date"]])
    test_actual = test_df[target_col].values

    metrics = {
        "accuracy": float(np.mean((test_pred > 0) == (test_actual > 0))),
        "ic": float(spearmanr(test_pred, test_actual)[0]),
        "rmse": float(np.sqrt(np.mean((test_pred - test_actual) ** 2))),
    }

    print(f"\n  📊 结果：")
    print(f"    准确率：{metrics['accuracy']:.4f}")
    print(f"    IC:     {metrics['ic']:.4f}")
    print(f"    RMSE:   {metrics['rmse']:.6f}")

    return {
        "data_name": data_name,
        "num_features": len(feature_cols),
        "seq_length": seq_length,
        **metrics
    }


def main() -> int:
    print("\n" + "="*80)
    print("🎯 归一化优化验证实验")
    print("="*80 + "\n")

    results = []

    # 1. 原始数据（基线）
    results.append(train_and_evaluate(
        PROCESSED_DIR / "training_panel_lite.csv",
        "原始数据（38 特征）",
        seq_length=20
    ))

    # 2. 优化归一化数据
    results.append(train_and_evaluate(
        PROCESSED_DIR / "training_panel_optimized.csv",
        "优化归一化（33 特征）",
        seq_length=20
    ))

    # 3. 优化数据 seq=25
    results.append(train_and_evaluate(
        PROCESSED_DIR / "training_panel_optimized.csv",
        "优化归一化（33 特征，seq=25）",
        seq_length=25
    ))

    # 保存结果
    output_dir = REPORTS_DIR / "normalization_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "comparison_results.csv", index=False)

    # 打印对比
    print("\n" + "="*80)
    print("📊 对比总结")
    print("="*80 + "\n")

    for idx, row in results_df.iterrows():
        print(f"{idx+1}. {row['data_name']}")
        print(f"   准确率：{row['accuracy']:.4f}")
        print(f"   IC:     {row['ic']:.4f}")
        print(f"   特征数：{row['num_features']}")
        print()

    # 计算提升
    baseline_acc = results_df.iloc[0]['accuracy']
    best_acc = results_df['accuracy'].max()
    improvement = (best_acc - baseline_acc) * 100

    print(f"💡 最佳准确率：{best_acc:.4f}")
    print(f"💡 基线准确率：{baseline_acc:.4f}")
    print(f"💡 提升幅度：{improvement:+.2f}%")

    if best_acc > baseline_acc:
        print(f"\n✅ 优化归一化成功！性能提升 {improvement:.2f}%")
    else:
        print(f"\n⚠️ 优化归一化未带来提升，差异 {improvement:.2f}%")

    print(f"\n💾 结果已保存：{output_dir}/comparison_results.csv\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
