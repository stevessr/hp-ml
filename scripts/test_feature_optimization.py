#!/usr/bin/env python3
"""测试不同特征数量的性能

目标：找到最优特征数量，冲击 70%
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
from scripts.test_innovative_architectures import MultiScaleLSTM


def test_feature_set(data_path: Path, name: str) -> dict:
    """测试单个特征集"""
    print(f"\n{'='*80}")
    print(f"🔧 测试：{name}")
    print(f"{'='*80}")

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

    # 训练模型
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

    print(f"准确率：{acc:.4f}")
    print(f"IC: {ic:.4f}")
    print(f"RMSE: {rmse:.6f}")

    return {
        "name": name,
        "num_features": len(feature_cols),
        "accuracy": acc,
        "ic": ic,
        "rmse": rmse
    }


def main() -> int:
    print("\n" + "="*80)
    print("🎯 特征选择优化实验")
    print("="*80 + "\n")

    results = []

    # 测试不同特征数量
    test_sets = [
        (PROCESSED_DIR / "training_panel_top40.csv", "Top 40 特征"),
        (PROCESSED_DIR / "training_panel_top45.csv", "Top 45 特征"),
        (PROCESSED_DIR / "training_panel_top50.csv", "Top 50 特征"),
        (PROCESSED_DIR / "training_panel_top55.csv", "Top 55 特征"),
        (PROCESSED_DIR / "training_panel_extended_features.csv", "全部 67 特征"),
    ]

    for data_path, name in test_sets:
        if data_path.exists():
            result = test_feature_set(data_path, name)
            results.append(result)

    # 保存结果
    output_dir = REPORTS_DIR / "feature_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)

    # 打印对比
    print("\n" + "="*80)
    print("📊 结果对比")
    print("="*80 + "\n")

    print("基线：")
    print(f"  多尺度 LSTM (38 特征): 68.08%, IC: 0.3976\n")
    print(f"  多尺度 LSTM (67 特征): 68.97%, IC: 0.4133\n")

    print("特征优化：")
    for idx, row in results_df.iterrows():
        print(f"{idx+1}. {row['name']}")
        print(f"   特征数：{row['num_features']}")
        print(f"   准确率：{row['accuracy']:.4f}")
        print(f"   IC: {row['ic']:.4f}\n")

    # 找出最佳
    best_idx = results_df['accuracy'].idxmax()
    best = results_df.loc[best_idx]

    print("="*80)
    print(f"💡 最佳配置：{best['name']}")
    print(f"   特征数：{best['num_features']}")
    print(f"   准确率：{best['accuracy']:.4f}")
    print(f"   IC: {best['ic']:.4f}")

    if best['accuracy'] >= 0.70:
        print(f"\n🎉 达到 70% 目标！")
    elif best['accuracy'] > 0.6897:
        improvement = (best['accuracy'] - 0.6897) * 100
        print(f"\n✅ 超越基线：+{improvement:.2f}%")
    elif best['accuracy'] >= 0.6850:
        diff = (best['accuracy'] - 0.6897) * 100
        print(f"\n⚠️ 接近基线：{diff:+.2f}%")
    else:
        print(f"\n⚠️ 低于基线")

    print("="*80)
    print(f"\n💾 结果已保存：{output_dir}/results.csv\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
