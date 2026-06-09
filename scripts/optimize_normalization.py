#!/usr/bin/env python3
"""优化的特征归一化

修复问题：
1. 过度归一化导致信号丢失
2. 按 stock 分组归一化（避免跨股票标准化）
3. 使用 RobustScaler（对异常值更稳健）
4. 特征重要性分析和选择
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import PROCESSED_DIR


def robust_normalize_by_stock(df: pd.DataFrame, feature_col: str) -> pd.Series:
    """
    按股票分组的鲁棒归一化

    使用中位数和 IQR，对异常值更稳健
    """
    def normalize_group(group):
        values = group[feature_col].values
        median = np.median(values)
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25 + 1e-8  # 避免除零
        return (values - median) / iqr

    return df.groupby('code')[feature_col].transform(
        lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25) + 1e-8)
    )


def rolling_normalize(df: pd.DataFrame, feature_col: str, window: int = 20) -> pd.Series:
    """
    滚动窗口归一化

    使用历史窗口的统计量，避免前视偏差
    """
    def normalize_group(group):
        values = group[feature_col]
        # 滚动均值和标准差
        rolling_mean = values.rolling(window, min_periods=1).mean()
        rolling_std = values.rolling(window, min_periods=1).std()
        return (values - rolling_mean) / (rolling_std + 1e-8)

    return df.groupby('code').apply(normalize_group).reset_index(level=0, drop=True)


def add_optimized_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加优化的特征"""
    print("📊 添加优化特征...")

    df = df.copy()
    new_features = []

    # 1. 涨跌幅特征 - 使用滚动归一化（避免全局标准化）
    ret_cols = [col for col in df.columns if col.startswith('ret_')]
    print(f"\n  处理{len(ret_cols)}个收益率特征...")

    for col in ret_cols:
        # 方案 A: 滚动 Z-score（20 日窗口）
        new_col = f'{col}_rolling_zscore'
        df[new_col] = rolling_normalize(df, col, window=20)
        new_features.append(new_col)
        print(f"    ✓ {new_col}")

    # 2. 波动率特征 - 相对于历史均值（而非全局归一化）
    vol_cols = [col for col in df.columns if col.startswith('vol_')]
    print(f"\n  处理{len(vol_cols)}个波动率特征...")

    for col in vol_cols:
        # 相对波动率（当前/20 日均值）
        new_col = f'{col}_relative'
        df[new_col] = df.groupby('code')[col].transform(
            lambda x: x / (x.rolling(20, min_periods=1).mean() + 1e-8)
        )
        new_features.append(new_col)
        print(f"    ✓ {new_col}")

    # 3. 换手率 - 相对于自身历史
    if 'turnover_rate' in df.columns:
        print(f"\n  处理换手率特征...")
        df['turnover_ratio'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x / (x.rolling(20, min_periods=1).mean() + 1e-8)
        )
        new_features.append('turnover_ratio')
        print(f"    ✓ turnover_ratio")

    # 4. 成交量变化 - 剪裁异常值（而非标准化）
    if 'volume_chg_5' in df.columns:
        print(f"\n  处理成交量变化...")
        # 剪裁到 [-3, 3] 范围
        df['volume_chg_5_clipped'] = df['volume_chg_5'].clip(-3, 3)
        new_features.append('volume_chg_5_clipped')
        print(f"    ✓ volume_chg_5_clipped")

    # 5. 流动性冲击 - 使用 RobustScaler
    if 'liquidity_shock_20' in df.columns:
        print(f"\n  处理流动性冲击...")
        df['liquidity_shock_robust'] = robust_normalize_by_stock(df, 'liquidity_shock_20')
        new_features.append('liquidity_shock_robust')
        print(f"    ✓ liquidity_shock_robust")

    print(f"\n✅ 新增{len(new_features)}个优化特征")
    return df, new_features


def analyze_feature_importance(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> pd.DataFrame:
    """
    分析特征重要性（相关性）

    返回特征重要性排序
    """
    print("\n📊 分析特征重要性...")

    correlations = []
    for feat in feature_cols:
        if feat in df.columns:
            # 计算 Spearman 相关系数（对非线性关系更稳健）
            corr = df[[feat, target_col]].corr(method='spearman').iloc[0, 1]
            correlations.append({
                'feature': feat,
                'correlation': abs(corr),  # 绝对值
                'raw_correlation': corr
            })

    importance_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

    print("\nTop 10 最重要特征：")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['correlation']:.4f} ({row['raw_correlation']:+.4f})")

    print("\nBottom 5 最不重要特征：")
    for idx, row in importance_df.tail(5).iterrows():
        print(f"  {row['feature']}: {row['correlation']:.4f} ({row['raw_correlation']:+.4f})")

    return importance_df


def select_features(importance_df: pd.DataFrame, threshold: float = 0.01) -> list[str]:
    """
    特征选择：只保留重要特征

    threshold: 最小相关系数阈值
    """
    selected = importance_df[importance_df['correlation'] >= threshold]['feature'].tolist()
    removed = importance_df[importance_df['correlation'] < threshold]['feature'].tolist()

    print(f"\n📊 特征选择（阈值={threshold}）:")
    print(f"  保留：{len(selected)}个特征")
    print(f"  移除：{len(removed)}个特征")

    if removed:
        print(f"\n  移除的特征：")
        for feat in removed:
            corr = importance_df[importance_df['feature'] == feat]['correlation'].values[0]
            print(f"    • {feat} (相关性：{corr:.4f})")

    return selected


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="优化特征归一化")
    parser.add_argument("--input", type=Path,
                       default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--output", type=Path,
                       default=PROCESSED_DIR / "training_panel_optimized.csv")
    parser.add_argument("--analyze", action="store_true",
                       help="分析特征重要性")
    parser.add_argument("--select", action="store_true",
                       help="执行特征选择（移除低价值特征）")
    parser.add_argument("--threshold", type=float, default=0.01,
                       help="特征选择阈值")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔧 优化特征归一化")
    print("="*80)
    print(f"📁 输入：{args.input}")
    print(f"📁 输出：{args.output}")
    print("="*80 + "\n")

    if not args.input.exists():
        print(f"❌ 输入文件不存在：{args.input}")
        return 1

    try:
        # 加载数据
        print("📊 加载数据...")
        df = pd.read_csv(args.input)
        original_cols = len(df.columns)
        print(f"  ✓ {len(df)} 行，{original_cols} 列")

        # 添加优化特征
        df, new_features = add_optimized_features(df)

        # 特征重要性分析
        if args.analyze or args.select:
            target_col = "fwd_ret_5"
            all_features = [col for col in df.columns
                          if col not in ['date', 'code', 'name', 'family_id',
                                       'close', target_col, 'is_trainable', 'is_future']
                          and not col.startswith('fwd_')]

            importance_df = analyze_feature_importance(df, target_col, all_features)

            # 保存重要性分析
            importance_path = args.output.parent / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            print(f"\n💾 特征重要性已保存：{importance_path}")

            # 特征选择
            if args.select:
                selected_features = select_features(importance_df, args.threshold)

                # 保留选中的特征
                keep_cols = ['date', 'code', 'name', 'family_id', 'close',
                           target_col, 'is_trainable'] + selected_features
                keep_cols = [c for c in keep_cols if c in df.columns]

                df = df[keep_cols]
                print(f"\n📊 特征选择后：{len(df.columns)} 列")

        # 保存
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)

        new_cols = len(df.columns) - original_cols

        print("\n" + "="*80)
        print("✅ 特征优化完成")
        print("="*80)
        print(f"📊 原始列数：{original_cols}")
        print(f"📊 新增列数：{len(new_features)}")
        print(f"📊 最终列数：{len(df.columns)}")
        print(f"💾 保存至：{args.output}")
        print("="*80 + "\n")

        # 打印新特征
        if new_features:
            print("🆕 新增特征：")
            for feat in new_features:
                if feat in df.columns:
                    print(f"  • {feat}")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
