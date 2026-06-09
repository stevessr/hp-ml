#!/usr/bin/env python3
"""扩展特征工程 - 引入更多可收集的技术特征

新增特征类别：
1. 高级技术指标
2. 价格形态特征
3. 相对强弱特征
4. 资金流向指标
5. 市场微观结构
6. 时间周期特征
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import PROCESSED_DIR


def add_advanced_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加高级技术指标"""
    print("📊 添加高级技术指标...")

    df = df.sort_values(['code', 'date']).copy()

    # 1. 布林带指标
    print("  • 布林带相关指标...")
    df['ma_20'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    df['std_20'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=1).std()
    )
    df['bollinger_upper'] = df['ma_20'] + 2 * df['std_20']
    df['bollinger_lower'] = df['ma_20'] - 2 * df['std_20']
    df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['ma_20']
    df['price_to_bollinger'] = (df['close'] - df['ma_20']) / (df['std_20'] + 1e-8)

    # 2. RSI 相对强弱指标
    print("  • RSI 指标...")
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    df['rsi_14'] = df.groupby('code')['close'].transform(lambda x: calculate_rsi(x, 14))
    df['rsi_28'] = df.groupby('code')['close'].transform(lambda x: calculate_rsi(x, 28))

    # 3. MACD 指标
    print("  • MACD 指标...")
    df['ema_12'] = df.groupby('code')['close'].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    df['ema_26'] = df.groupby('code')['close'].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df.groupby('code')['macd'].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # 清理临时列
    df.drop(['ma_20', 'std_20', 'bollinger_upper', 'bollinger_lower',
             'ema_12', 'ema_26', 'macd', 'macd_signal'], axis=1, inplace=True)

    return df


def add_price_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加价格形态特征"""
    print("📊 添加价格形态特征...")

    df = df.sort_values(['code', 'date']).copy()

    # 1. 价格位置（相对高低点）
    print("  • 价格位置特征...")
    df['high_20'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=1).max()
    )
    df['low_20'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=1).min()
    )
    df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)

    # 2. 突破特征
    print("  • 突破特征...")
    df['breakout_high'] = (df['close'] >= df['high_20'].shift(1)).astype(int)
    df['breakout_low'] = (df['close'] <= df['low_20'].shift(1)).astype(int)

    # 3. 价格加速度
    print("  • 价格动量特征...")
    df['ret_1_shift'] = df.groupby('code')['ret_1'].shift(1)
    df['price_acceleration'] = df['ret_1'] - df['ret_1_shift']

    # 4. 连续涨跌天数
    print("  • 连续涨跌统计...")
    def count_consecutive(series):
        result = []
        count = 0
        for val in series:
            if pd.isna(val):
                result.append(0)
            elif val > 0:
                count = count + 1 if count > 0 else 1
                result.append(count)
            elif val < 0:
                count = count - 1 if count < 0 else -1
                result.append(count)
            else:
                count = 0
                result.append(0)
        return pd.Series(result, index=series.index)

    df['consecutive_days'] = df.groupby('code')['ret_1'].transform(count_consecutive)

    # 清理临时列
    df.drop(['high_20', 'low_20', 'ret_1_shift'], axis=1, inplace=True)

    return df


def add_relative_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加相对强弱特征（相对于市场/行业）"""
    print("📊 添加相对强弱特征...")

    df = df.sort_values(['code', 'date']).copy()

    # 计算每日市场平均收益
    print("  • 相对市场表现...")
    market_ret = df.groupby('date')['ret_5'].transform('mean')
    df['relative_ret_5'] = df['ret_5'] - market_ret

    market_ret_20 = df.groupby('date')['ret_20'].transform('mean')
    df['relative_ret_20'] = df['ret_20'] - market_ret_20

    # 相对波动率
    print("  • 相对波动率...")
    market_vol = df.groupby('date')['vol_20'].transform('mean')
    df['relative_vol_20'] = df['vol_20'] / (market_vol + 1e-8)

    # Beta 系数（简化版）
    print("  • Beta 系数...")
    def rolling_beta(group_ret, market_ret, window=20):
        cov = group_ret.rolling(window, min_periods=5).cov(market_ret)
        var = market_ret.rolling(window, min_periods=5).var()
        return cov / (var + 1e-8)

    df['beta_20'] = df.groupby('code').apply(
        lambda g: rolling_beta(g['ret_1'], market_ret.loc[g.index], 20)
    ).reset_index(level=0, drop=True)

    return df


def add_money_flow_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加资金流向指标"""
    print("📊 添加资金流向指标...")

    df = df.sort_values(['code', 'date']).copy()

    # 1. 成交金额变化
    print("  • 成交金额特征...")
    if 'amount' in df.columns:
        df['amount_ma_5'] = df.groupby('code')['amount'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['amount_ma_20'] = df.groupby('code')['amount'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        df['amount_ratio'] = df['amount'] / (df['amount_ma_20'] + 1e-8)
        df['amount_trend'] = (df['amount_ma_5'] - df['amount_ma_20']) / (df['amount_ma_20'] + 1e-8)

        df.drop(['amount_ma_5', 'amount_ma_20'], axis=1, inplace=True)

    # 2. 价量背离
    print("  • 价量关系...")
    df['price_volume_corr'] = df.groupby('code').apply(
        lambda g: g['ret_1'].rolling(10, min_periods=3).corr(g['volume_chg_5'])
    ).reset_index(level=0, drop=True)

    # 3. 换手率动量
    if 'turnover_rate' in df.columns:
        print("  • 换手率动量...")
        df['turnover_momentum'] = df.groupby('code')['turnover_rate'].transform(
            lambda x: x.diff(5)
        )

    return df


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加市场微观结构特征"""
    print("📊 添加微观结构特征...")

    df = df.sort_values(['code', 'date']).copy()

    # 1. 价格效率（波动率/收益率比）
    print("  • 价格效率...")
    df['price_efficiency'] = np.abs(df['ret_5']) / (df['vol_5'] + 1e-8)

    # 2. 收益波动率比（夏普比率的简化版）
    print("  • 收益波动率比...")
    df['return_volatility_ratio'] = df['ret_20'] / (df['vol_20'] + 1e-8)

    # 3. 价格跳跃（大幅波动）
    print("  • 价格跳跃检测...")
    df['large_move'] = (np.abs(df['ret_1']) > df.groupby('code')['ret_1'].transform(
        lambda x: x.rolling(20, min_periods=1).std() * 2
    )).astype(int)

    # 4. 波动率聚类（GARCH 效应）
    print("  • 波动率持续性...")
    df['vol_persistence'] = df.groupby('code')['vol_5'].transform(
        lambda x: x.rolling(5, min_periods=1).std()
    )

    return df


def add_time_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间周期特征"""
    print("📊 添加时间周期特征...")

    df = df.sort_values(['code', 'date']).copy()

    # 1. 星期几效应
    print("  • 星期效应...")
    df['date_dt'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date_dt'].dt.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)

    # 2. 月初月末效应
    print("  • 月度效应...")
    df['day_of_month'] = df['date_dt'].dt.day
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)

    # 3. 季度效应
    print("  • 季度效应...")
    df['quarter'] = df['date_dt'].dt.quarter
    df['is_quarter_end'] = (
        (df['date_dt'].dt.month.isin([3, 6, 9, 12])) &
        (df['day_of_month'] >= 25)
    ).astype(int)

    # 4. 距离上次大涨/大跌的天数
    print("  • 事件距离...")
    def days_since_event(series, threshold=0.03):
        result = []
        days = 0
        for val in series:
            if pd.isna(val):
                result.append(days)
            elif abs(val) > threshold:
                days = 0
                result.append(days)
            else:
                days += 1
                result.append(days)
        return pd.Series(result, index=series.index)

    df['days_since_large_move'] = df.groupby('code')['ret_1'].transform(
        lambda x: days_since_event(x, 0.03)
    )

    df.drop(['date_dt'], axis=1, inplace=True)

    return df


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="添加扩展特征")
    parser.add_argument("--input", type=Path,
                       default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--output", type=Path,
                       default=PROCESSED_DIR / "training_panel_extended_features.csv")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🔧 扩展特征工程")
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
        print(f"  ✓ {len(df)} 行，{original_cols} 列\n")

        # 添加各类特征
        df = add_advanced_technical_indicators(df)
        df = add_price_pattern_features(df)
        df = add_relative_strength_features(df)
        df = add_money_flow_indicators(df)
        df = add_microstructure_features(df)
        df = add_time_cycle_features(df)

        # 填充 NaN
        print("\n📊 处理缺失值...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # 保存
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)

        new_cols = len(df.columns) - original_cols

        print("\n" + "="*80)
        print("✅ 特征扩展完成")
        print("="*80)
        print(f"📊 原始列数：{original_cols}")
        print(f"📊 新增列数：{new_cols}")
        print(f"📊 最终列数：{len(df.columns)}")
        print(f"💾 保存至：{args.output}")
        print("="*80 + "\n")

        # 打印新增特征列表
        all_cols = set(df.columns)
        original_cols_set = set(pd.read_csv(args.input, nrows=0).columns)
        new_features = sorted(all_cols - original_cols_set)

        print(f"🆕 新增特征 ({len(new_features)}个):\n")
        for i, feat in enumerate(new_features, 1):
            print(f"  {i:2d}. {feat}")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
