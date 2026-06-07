#!/usr/bin/env python3
"""特征重要性分析和选择

目标：
1. 分析 67 个特征的重要性
2. 移除低价值特征
3. 优化特征组合
4. 冲击 70%
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import PROCESSED_DIR, REPORTS_DIR


def analyze_feature_importance(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """使用随机森林分析特征重要性"""
    print("📊 分析特征重要性...")

    exclude_cols = {'date', 'code', 'name', 'family_id', 'close', target_col,
                   'is_trainable', 'is_future'}
    feature_cols = [c for c in df.columns if c not in exclude_cols
                   and not c.startswith('fwd_')]

    # 准备数据
    trainable = df[df.get("is_trainable", True)].copy()
    X = trainable[feature_cols].fillna(0)
    y = trainable[target_col]

    # 随机森林
    print("  • 训练随机森林...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # 特征重要性
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # 相关性
    print("  • 计算相关性...")
    correlations = []
    for feat in feature_cols:
        corr = spearmanr(trainable[feat], trainable[target_col])[0]
        correlations.append(abs(corr))

    importance_df['correlation'] = correlations
    importance_df['combined_score'] = (
        importance_df['importance'] * 0.6 +
        importance_df['correlation'] * 0.4
    )
    importance_df = importance_df.sort_values('combined_score', ascending=False)

    return importance_df


def select_top_features(importance_df: pd.DataFrame, top_n: int = 50) -> list[str]:
    """选择 Top N 特征"""
    return importance_df.head(top_n)['feature'].tolist()


def main() -> int:
    print("\n" + "="*80)
    print("🔍 特征重要性分析和选择")
    print("="*80 + "\n")

    # 加载数据
    data_path = PROCESSED_DIR / "training_panel_extended_features.csv"
    df = pd.read_csv(data_path)

    target_col = "fwd_ret_5"

    # 分析特征重要性
    importance_df = analyze_feature_importance(df, target_col)

    # 保存结果
    output_dir = REPORTS_DIR / "feature_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    print("\n" + "="*80)
    print("📊 特征重要性排名")
    print("="*80 + "\n")

    print("Top 20 最重要特征:\n")
    for idx, row in importance_df.head(20).iterrows():
        print(f"{idx+1:2d}. {row['feature']:30s} "
              f"重要性：{row['importance']:.4f}  "
              f"相关性：{row['correlation']:.4f}  "
              f"综合：{row['combined_score']:.4f}")

    print("\n" + "-"*80 + "\n")
    print("Bottom 10 最不重要特征:\n")
    for idx, row in importance_df.tail(10).iterrows():
        print(f"{row['feature']:30s} "
              f"重要性：{row['importance']:.4f}  "
              f"相关性：{row['correlation']:.4f}  "
              f"综合：{row['combined_score']:.4f}")

    # 生成不同特征数量的数据集
    print("\n" + "="*80)
    print("📦 生成优化特征数据集")
    print("="*80 + "\n")

    for top_n in [40, 45, 50, 55]:
        selected_features = select_top_features(importance_df, top_n)

        # 保留必要列
        keep_cols = ['date', 'code', 'name', 'family_id', 'close', target_col, 'is_trainable']
        keep_cols.extend(selected_features)
        keep_cols = [c for c in keep_cols if c in df.columns]

        optimized_df = df[keep_cols].copy()

        output_path = PROCESSED_DIR / f"training_panel_top{top_n}.csv"
        optimized_df.to_csv(output_path, index=False)

        print(f"  ✓ Top {top_n} 特征数据集：{output_path}")

    print("\n" + "="*80)
    print("✅ 特征分析完成")
    print("="*80)
    print(f"💾 特征重要性：{output_dir}/feature_importance.csv")
    print(f"📦 优化数据集：{PROCESSED_DIR}/training_panel_top*.csv")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
