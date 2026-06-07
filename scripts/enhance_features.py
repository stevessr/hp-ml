#!/usr/bin/env python3
"""增强特征工程：归一化和股东结构特征

添加以下特征：
1. 涨幅归一化（相对变化而非绝对价格）
2. 买入/卖出比例特征
3. 大股东结构特征（个人、国家背景）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import PROCESSED_DIR


def normalize_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """归一化和增强特征"""
    print("📊 归一化和增强特征...")

    df = df.copy()

    # 1. 标准化涨跌幅特征（已经是比例，进一步归一化）
    ret_cols = [col for col in df.columns if col.startswith('ret_')]
    for col in ret_cols:
        # Z-score 归一化（按code分组）
        df[f'{col}_zscore'] = df.groupby('code')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        print(f"  ✓ {col} → {col}_zscore (标准化)")

    # 2. 波动率特征归一化
    vol_cols = [col for col in df.columns if col.startswith('vol_')]
    for col in vol_cols:
        # 相对波动率（相对于均值）
        df[f'{col}_norm'] = df.groupby('code')[col].transform(
            lambda x: x / (x.mean() + 1e-8)
        )
        print(f"  ✓ {col} → {col}_norm (归一化)")

    # 3. 成交量变化归一化
    if 'volume_chg_5' in df.columns:
        df['volume_chg_5_clip'] = df['volume_chg_5'].clip(-3, 3)  # 剪裁异常值
        print(f"  ✓ volume_chg_5 → volume_chg_5_clip (剪裁)")

    # 4. 流动性冲击归一化
    if 'liquidity_shock_20' in df.columns:
        df['liquidity_shock_20_norm'] = df.groupby('code')['liquidity_shock_20'].transform(
            lambda x: (x - x.median()) / (x.std() + 1e-8)
        )
        print(f"  ✓ liquidity_shock_20 → liquidity_shock_20_norm")

    # 5. 换手率归一化（相对于历史均值）
    if 'turnover_rate' in df.columns:
        df['turnover_ratio'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x / (x.rolling(20, min_periods=1).mean() + 1e-8)
        )
        print(f"  ✓ turnover_rate → turnover_ratio")

    return df


def add_shareholder_features(df: pd.DataFrame, shareholder_data_path: Path) -> pd.DataFrame:
    """添加股东结构特征"""
    print("\n📊 添加股东结构特征...")

    if not shareholder_data_path.exists():
        print(f"  ⚠ 股东数据文件不存在: {shareholder_data_path}")
        return df

    try:
        shareholder_df = pd.read_csv(shareholder_data_path)
        print(f"  ✓ 加载股东数据: {len(shareholder_df)} 行")

        # 确保有必需的列
        required_cols = ['code', 'date']
        if not all(col in shareholder_df.columns for col in required_cols):
            print(f"  ⚠ 缺少必需列: {required_cols}")
            return df

        # 计算股东结构特征
        shareholder_features = []

        for code, group in shareholder_df.groupby('code'):
            group = group.sort_values('date')

            # 个人大股东比例
            if 'individual_holder_ratio' in group.columns:
                group['individual_ratio'] = group['individual_holder_ratio']

            # 国家背景大股东比例
            if 'state_holder_ratio' in group.columns:
                group['state_ratio'] = group['state_holder_ratio']

            # 股东集中度
            if 'top1_holder_ratio' in group.columns:
                group['concentration'] = group['top1_holder_ratio']

            shareholder_features.append(group[['code', 'date'] +
                                              [c for c in ['individual_ratio', 'state_ratio', 'concentration']
                                               if c in group.columns]])

        if shareholder_features:
            shareholder_df = pd.concat(shareholder_features, ignore_index=True)
            df = pd.merge(df, shareholder_df, on=['code', 'date'], how='left')
            print(f"  ✓ 添加股东特征: {len([c for c in df.columns if c in ['individual_ratio', 'state_ratio', 'concentration']])} 个")

    except Exception as e:
        print(f"  ❌ 处理股东数据失败: {e}")

    return df


def add_trading_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加交易流向特征（买入/卖出比例）"""
    print("\n📊 添加交易流向特征...")

    df = df.copy()

    # 如果有成交量和成交额数据
    if 'volume' in df.columns and 'amount' in df.columns:
        # 估算买入卖出力量（基于价格变动和成交量）
        df['price_change'] = df.groupby('code')['close'].pct_change()

        # 正向价格变动时的成交量视为买入力量
        df['buy_volume_proxy'] = df.apply(
            lambda x: x['volume'] if x['price_change'] > 0 else 0, axis=1
        )
        df['sell_volume_proxy'] = df.apply(
            lambda x: x['volume'] if x['price_change'] < 0 else 0, axis=1
        )

        # 计算买卖比例（5日窗口）
        df['buy_ratio_5d'] = df.groupby('code')['buy_volume_proxy'].transform(
            lambda x: x.rolling(5, min_periods=1).sum() / (x.rolling(5, min_periods=1).sum() +
                     df.loc[x.index, 'sell_volume_proxy'].rolling(5, min_periods=1).sum() + 1e-6)
        )

        # 昨日买入/卖出占总股本比例（需要总股本数据）
        # 这里使用成交量占流通股的比例作为代理
        if 'total_shares' in df.columns:
            df['buy_ratio_of_shares'] = df['buy_volume_proxy'] / df['total_shares']
            df['sell_ratio_of_shares'] = df['sell_volume_proxy'] / df['total_shares']
            print(f"  ✓ 买入/卖出占股本比例")

        print(f"  ✓ 买卖比例特征")

    return df


def enhance_panel_data(input_path: Path, output_path: Path, shareholder_path: Path | None = None) -> None:
    """增强面板数据"""
    print("\n" + "="*80)
    print("🔧 增强特征工程")
    print("="*80)
    print(f"📁 输入: {input_path}")
    print(f"📁 输出: {output_path}")
    print("="*80 + "\n")

    # 加载数据
    print("📊 加载原始面板数据...")
    df = pd.read_csv(input_path)
    print(f"  ✓ 加载 {len(df)} 行, {len(df.columns)} 列")

    original_cols = len(df.columns)

    # 应用转换
    df = normalize_price_features(df)
    df = add_trading_flow_features(df)

    if shareholder_path:
        df = add_shareholder_features(df, shareholder_path)

    new_cols = len(df.columns) - original_cols

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "="*80)
    print("✅ 特征增强完成")
    print("="*80)
    print(f"📊 原始特征数: {original_cols}")
    print(f"📊 新增特征数: {new_cols}")
    print(f"📊 总特征数: {len(df.columns)}")
    print(f"💾 保存至: {output_path}")
    print("="*80 + "\n")

    # 打印新特征列表
    new_features = [col for col in df.columns if col not in pd.read_csv(input_path).columns]
    if new_features:
        print("🆕 新增特征:")
        for feat in new_features:
            print(f"  • {feat}")
        print()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="增强特征工程")
    parser.add_argument("--input", type=Path,
                       default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--output", type=Path,
                       default=PROCESSED_DIR / "training_panel_enhanced.csv")
    parser.add_argument("--shareholder", type=Path,
                       help="股东数据CSV路径（可选）")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ 输入文件不存在: {args.input}")
        return 1

    try:
        enhance_panel_data(args.input, args.output, args.shareholder)
        return 0
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
