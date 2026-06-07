#!/usr/bin/env python3
"""在扩展特征数据上测试模型

验证新特征是否能突破68.08%天花板
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import PROCESSED_DIR, REPORTS_DIR

# 导入多尺度LSTM（我们最好的模型）
from scripts.test_innovative_architectures import MultiScaleLSTM


def main() -> int:
    print("\n" + "="*80)
    print("🚀 在扩展特征数据上测试模型")
    print("="*80 + "\n")

    # 加载扩展特征数据
    data_path = PROCESSED_DIR / "training_panel_extended_features.csv"
    df = pd.read_csv(data_path)

    target_col = "fwd_ret_5"
    exclude_cols = {'date', 'code', 'name', 'family_id', 'close', target_col,
                   'is_trainable', 'is_future'}
    feature_cols = [c for c in df.columns if c not in exclude_cols
                   and not c.startswith('fwd_')]

    print(f"特征数: {len(feature_cols)} (原38 → 现{len(feature_cols)})")

    # 切分数据
    df = df.sort_values("date")
    trainable = df[df.get("is_trainable", True)].copy()
    unique_dates = sorted(trainable["date"].unique())
    split_idx = max(1, len(unique_dates) - 60)
    split_date = unique_dates[split_idx]

    train_df = trainable[trainable["date"] < split_date].copy()
    test_df = trainable[trainable["date"] >= split_date].copy()

    print(f"训练集: {len(train_df)} 行")
    print(f"测试集: {len(test_df)} 行\n")

    results = []

    # 测试多尺度LSTM（之前最好的模型）
    print("="*80)
    print("🔧 训练多尺度LSTM (67特征)...")
    print("="*80)

    model = MultiScaleLSTM(
        seq_length=20,
        units=64,
        dropout=0.3,
        learning_rate=0.001,
        epochs=30,
        batch_size=32
    )

    model.fit(train_df[feature_cols + ["code", "date"]], train_df[target_col])
    pred = model.predict(test_df[feature_cols + ["code", "date"]])
    actual = test_df[target_col].values

    acc = float(np.mean((pred > 0) == (actual > 0)))
    ic = float(spearmanr(pred, actual)[0])
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))

    print(f"  准确率: {acc:.4f}")
    print(f"  IC: {ic:.4f}")
    print(f"  RMSE: {rmse:.6f}\n")

    results.append({
        "model": "Multi-Scale LSTM (67 features)",
        "num_features": len(feature_cols),
        "accuracy": acc,
        "ic": ic,
        "rmse": rmse
    })

    # 保存结果
    output_dir = REPORTS_DIR / "extended_features_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)

    print("\n" + "="*80)
    print("📊 结果对比")
    print("="*80 + "\n")

    print("基线对比:")
    print(f"  多尺度LSTM (38特征): 68.08%, IC: 0.3976\n")

    print("扩展特征:")
    for idx, row in results_df.iterrows():
        print(f"{idx+1}. {row['model']}")
        print(f"   特征数: {row['num_features']}")
        print(f"   准确率: {row['accuracy']:.4f}")
        print(f"   IC: {row['ic']:.4f}\n")

    best_acc = results_df['accuracy'].max()
    print(f"💡 最佳准确率: {best_acc:.4f}")

    if best_acc > 0.6808:
        improvement = (best_acc - 0.6808) * 100
        print(f"💡 超越基线: +{improvement:.2f}%")
        print(f"\n✅ 特征扩展成功！突破天花板！")
    elif best_acc > 0.6750:
        diff = (best_acc - 0.6808) * 100
        print(f"💡 接近基线: {diff:+.2f}%")
        print(f"\n⚠️ 接近但未突破 (68.08%)")
    else:
        diff = (best_acc - 0.6808) * 100
        print(f"💡 低于基线: {diff:+.2f}%")
        print(f"\n⚠️ 新特征未带来提升")

    print(f"\n💾 结果已保存: {output_dir}/results.csv\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
