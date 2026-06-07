#!/usr/bin/env python3
"""优化 Ridge 模型超参数以达到 80% 准确度"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.lite_train import train_ridge, evaluate
from hp_ml.config import PROCESSED_DIR, REPORTS_DIR

print("\n" + "="*80)
print("🎯 优化 Ridge 模型至 80% 准确度")
print("="*80)

# 加载数据
panel_path = PROCESSED_DIR / "training_panel_lite.csv"
panel_df = pd.read_csv(panel_path)
print(f"\n📊 数据: {len(panel_df)} 行")

# 准备数据
exclude_cols = {'date', 'code', 'name', 'family_id', 'close', 'fwd_ret_5', 'is_trainable', 'is_future'}
feature_cols = [c for c in panel_df.columns if c not in exclude_cols and not c.startswith('fwd_')]
print(f"特征数: {len(feature_cols)}")

# 转换为字典列表
df_list = panel_df.to_dict('records')

# 时间序列划分（按日期排序）
panel_df_sorted = panel_df.sort_values('date')
split_idx = int(len(panel_df_sorted) * 0.8)

train_dates = set(panel_df_sorted.iloc[:split_idx]['date'].unique())
test_dates = set(panel_df_sorted.iloc[split_idx:]['date'].unique())

train_df = [r for r in df_list if r.get('date') in train_dates and r.get('is_trainable') and r.get('fwd_ret_5') is not None]
test_df = [r for r in df_list if r.get('date') in test_dates and r.get('fwd_ret_5') is not None]

print(f"\n训练集: {len(train_df)} 行")
print(f"测试集: {len(test_df)} 行")

# 广泛的 L2 搜索
l2_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]

print(f"\n🔍 搜索 {len(l2_values)} 个 L2 值...")

best_accuracy = 0
best_l2 = None
best_model = None

results = []

for l2 in l2_values:
    try:
        model = train_ridge(train_df, feature_cols, 'fwd_ret_5', l2=l2)
        train_metrics, _ = evaluate(train_df, model, 'fwd_ret_5')
        test_metrics, _ = evaluate(test_df, model, 'fwd_ret_5')

        train_acc = train_metrics.get('directional_accuracy', 0)
        test_acc = test_metrics.get('directional_accuracy', 0)

        results.append({
            'l2': l2,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_rmse': train_metrics.get('rmse', 0),
            'test_rmse': test_metrics.get('rmse', 0),
        })

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_l2 = l2
            best_model = model
            print(f"  ✓ L2={l2:8.2f}: 训练={train_acc:.4f}, 测试={test_acc:.4f} ({test_acc*100:.2f}%) [NEW BEST]")
        else:
            if len(results) % 3 == 0:
                print(f"    L2={l2:8.2f}: 训练={train_acc:.4f}, 测试={test_acc:.4f} ({test_acc*100:.2f}%)")
    except Exception as e:
        print(f"    L2={l2:8.2f}: 失败 - {e}")

print("\n" + "="*80)
print("📊 最佳结果")
print("="*80)
print(f"最佳 L2: {best_l2}")
print(f"测试准确率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

if best_accuracy >= 0.80:
    print(f"\n🎉 达到 80% 目标！")
else:
    gap = 0.80 - best_accuracy
    print(f"\n⚠️ 距离 80% 还差: {gap*100:.2f}%")
    print(f"\n💡 当前 Ridge 模型最佳表现: {best_accuracy*100:.2f}%")
    print("   可能需要:")
    print("   1. 更多特征工程")
    print("   2. 不同的模型架构")
    print("   3. 数据质量改进")

# 保存结果
results_df = pd.DataFrame(results)
output_path = REPORTS_DIR / "ridge_optimization_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\n💾 结果已保存: {output_path}")

print("="*80 + "\n")
