#!/usr/bin/env python3
"""生成 ETF 策略回测可视化报告

包含：
1. 累计收益率曲线对比图
2. 日收益率分布直方图
3. 回撤曲线图
4. 月度收益热力图
5. 胜率统计饼图
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 使用非交互式后端
matplotlib.use('Agg')

# 设置中文字体 - 使用系统可用字体
plt.rcParams['font.sans-serif'] = ['Source Han Sans CN', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 尝试导入 seaborn（可选）
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  seaborn 未安装，将使用 matplotlib 原生绘图")

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
CHARTS_DIR = REPORTS / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_backtest_data(daily_csv: Path) -> pd.DataFrame:
    """加载回测数据"""
    df = pd.read_csv(daily_csv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 计算累计收益
    df['strategy_cumulative'] = (1 + df['strategy_return_raw'].fillna(0)).cumprod() - 1
    df['benchmark_cumulative'] = (1 + df['Benchmark'].fillna(0)).cumprod() - 1

    return df


def plot_cumulative_returns(df: pd.DataFrame, output_path: Path) -> None:
    """绘制累计收益率曲线"""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df['date'], df['strategy_cumulative'] * 100,
            label='策略', linewidth=2.5, color='#e74c3c')
    ax.plot(df['date'], df['benchmark_cumulative'] * 100,
            label='基准 (等权)', linewidth=2.5, color='#3498db', linestyle='--')

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('累计收益率 (%)', fontsize=11)
    ax.set_title('ETF 策略 vs 基准累计收益率对比', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 在图表下方添加关键指标说明
    final_strategy = df['strategy_cumulative'].iloc[-1] * 100
    final_benchmark = df['benchmark_cumulative'].iloc[-1] * 100
    excess = final_strategy - final_benchmark

    # 使用 fig.text 在图外添加说明
    fig.text(0.12, 0.02, f'策略收益：{final_strategy:.2f}%  |  基准收益：{final_benchmark:.2f}%  |  超额收益：{excess:.2f}%',
             fontsize=10, ha='left', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout(rect=[0, 0.04, 1, 1])  # 为底部文字留出空间
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 累计收益率曲线已保存：{output_path}")


def plot_daily_returns_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """绘制日收益率分布直方图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 策略收益分布
    strategy_returns = df['strategy_return_raw'].dropna() * 100
    ax1.hist(strategy_returns, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
    ax1.axvline(strategy_returns.mean(), color='darkred', linestyle='--', linewidth=2, label=f'均值：{strategy_returns.mean():.3f}%')
    ax1.axvline(0, color='gray', linestyle='-', linewidth=1)
    ax1.set_xlabel('日收益率 (%)', fontsize=10)
    ax1.set_ylabel('频数', fontsize=10)
    ax1.set_title('策略日收益率分布', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 基准收益分布
    benchmark_returns = df['Benchmark'].dropna() * 100
    ax2.hist(benchmark_returns, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax2.axvline(benchmark_returns.mean(), color='darkblue', linestyle='--', linewidth=2, label=f'均值：{benchmark_returns.mean():.3f}%')
    ax2.axvline(0, color='gray', linestyle='-', linewidth=1)
    ax2.set_xlabel('日收益率 (%)', fontsize=10)
    ax2.set_ylabel('频数', fontsize=10)
    ax2.set_title('基准日收益率分布', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 日收益率分布图已保存：{output_path}")


def plot_drawdown(df: pd.DataFrame, output_path: Path) -> None:
    """绘制回撤曲线"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # 计算回撤
    strategy_cum = (1 + df['strategy_return_raw'].fillna(0)).cumprod()
    strategy_drawdown = (strategy_cum / strategy_cum.cummax() - 1) * 100

    benchmark_cum = (1 + df['Benchmark'].fillna(0)).cumprod()
    benchmark_drawdown = (benchmark_cum / benchmark_cum.cummax() - 1) * 100

    ax.fill_between(df['date'], strategy_drawdown, 0, alpha=0.5, color='#e74c3c', label='策略回撤')
    ax.fill_between(df['date'], benchmark_drawdown, 0, alpha=0.5, color='#3498db', label='基准回撤')

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('回撤 (%)', fontsize=11)
    ax.set_title('策略回撤曲线对比', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower left', fontsize=10, frameon=True, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 在图表下方标注最大回撤
    max_dd_strategy = strategy_drawdown.min()
    max_dd_benchmark = benchmark_drawdown.min()
    fig.text(0.12, 0.02, f'策略最大回撤：{max_dd_strategy:.2f}%  |  基准最大回撤：{max_dd_benchmark:.2f}%',
             fontsize=10, ha='left', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 回撤曲线已保存：{output_path}")


def plot_monthly_returns_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """绘制月度收益热力图"""
    df_copy = df.copy()
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['month'] = df_copy['date'].dt.month

    # 计算每月收益
    monthly_strategy = df_copy.groupby(['year', 'month'])['strategy_return_raw'].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()

    # 透视表
    pivot = monthly_strategy.pivot(index='year', columns='month', values='strategy_return_raw') * 100

    # 确保所有月份都存在
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[sorted(pivot.columns)]

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 6))

    if HAS_SEABORN:
        import seaborn as sns
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': '月度收益率 (%)'}, ax=ax, linewidths=0.5)
    else:
        # 使用 matplotlib 的 imshow 作为替代
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

        # 添加数值标注
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=9)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label='月度收益率 (%)')

    ax.set_xlabel('月份', fontsize=11)
    ax.set_ylabel('年份', fontsize=11)
    ax.set_title('策略月度收益率热力图', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticklabels([f'{i}月' for i in range(1, 13)])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 月度收益热力图已保存：{output_path}")


def plot_win_rate_pie(df: pd.DataFrame, output_path: Path) -> None:
    """绘制胜率统计饼图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 策略胜率
    strategy_returns = df['strategy_return_raw'].dropna()
    strategy_win = (strategy_returns > 0).sum()
    strategy_loss = (strategy_returns < 0).sum()
    strategy_flat = (strategy_returns == 0).sum()

    colors1 = ['#2ecc71', '#e74c3c', '#95a5a6']
    wedges1, texts1, autotexts1 = ax1.pie([strategy_win, strategy_loss, strategy_flat],
            labels=[f'盈利 {strategy_win}天', f'亏损 {strategy_loss}天', f'持平 {strategy_flat}天'],
            autopct='%1.1f%%', colors=colors1, startangle=90, textprops={'fontsize': 9})
    ax1.set_title(f'策略胜率统计',
                  fontsize=12, fontweight='bold', pad=10)

    # 相对基准胜率
    excess = df['strategy_return_raw'] - df['Benchmark']
    excess_win = (excess > 0).sum()
    excess_loss = (excess < 0).sum()
    excess_flat = (excess == 0).sum()

    colors2 = ['#3498db', '#e67e22', '#95a5a6']
    wedges2, texts2, autotexts2 = ax2.pie([excess_win, excess_loss, excess_flat],
            labels=[f'跑赢 {excess_win}天', f'跑输 {excess_loss}天', f'持平 {excess_flat}天'],
            autopct='%1.1f%%', colors=colors2, startangle=90, textprops={'fontsize': 9})
    ax2.set_title(f'相对基准胜率',
                  fontsize=12, fontweight='bold', pad=10)

    # 在图表下方添加说明
    win_rate_strategy = strategy_win/(strategy_win+strategy_loss)*100
    win_rate_excess = excess_win/(excess_win+excess_loss)*100
    fig.text(0.5, 0.02, f'策略绝对胜率：{win_rate_strategy:.1f}%  |  相对基准胜率：{win_rate_excess:.1f}%',
             fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 胜率统计饼图已保存：{output_path}")


def plot_rolling_metrics(df: pd.DataFrame, output_path: Path, window: int = 60) -> None:
    """绘制滚动指标图（年化收益率和波动率）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 滚动年化收益率
    strategy_rolling_ret = df['strategy_return_raw'].rolling(window=window).mean() * 252 * 100
    benchmark_rolling_ret = df['Benchmark'].rolling(window=window).mean() * 252 * 100

    ax1.plot(df['date'], strategy_rolling_ret, label=f'策略 (滚动{window}天)', linewidth=2, color='#e74c3c')
    ax1.plot(df['date'], benchmark_rolling_ret, label=f'基准 (滚动{window}天)', linewidth=2, color='#3498db', linestyle='--')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax1.set_ylabel('年化收益率 (%)', fontsize=11, fontweight='bold')
    ax1.set_title(f'滚动年化收益率 ({window}天窗口)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 滚动年化波动率
    strategy_rolling_vol = df['strategy_return_raw'].rolling(window=window).std() * np.sqrt(252) * 100
    benchmark_rolling_vol = df['Benchmark'].rolling(window=window).std() * np.sqrt(252) * 100

    ax2.plot(df['date'], strategy_rolling_vol, label=f'策略 (滚动{window}天)', linewidth=2, color='#e74c3c')
    ax2.plot(df['date'], benchmark_rolling_vol, label=f'基准 (滚动{window}天)', linewidth=2, color='#3498db', linestyle='--')
    ax2.set_xlabel('日期', fontsize=11, fontweight='bold')
    ax2.set_ylabel('年化波动率 (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f'滚动年化波动率 ({window}天窗口)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 滚动指标图已保存：{output_path}")


def generate_summary_table(df: pd.DataFrame, output_path: Path) -> dict:
    """生成策略性能摘要表"""
    strategy_returns = df['strategy_return_raw'].dropna()
    benchmark_returns = df['Benchmark'].dropna()

    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    metrics = {
        '指标': ['累计收益率', '年化收益率', '年化波动率', '夏普比率', '最大回撤',
                '胜率', '平均日收益', '最佳单日收益', '最差单日收益', '总交易日'],
        '策略': [
            f"{(strategy_cum.iloc[-1] - 1) * 100:.2f}%",
            f"{(strategy_returns.mean() * 252) * 100:.2f}%",
            f"{(strategy_returns.std() * np.sqrt(252)) * 100:.2f}%",
            f"{(strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252)):.2f}",
            f"{((strategy_cum / strategy_cum.cummax() - 1).min()) * 100:.2f}%",
            f"{(strategy_returns > 0).sum() / len(strategy_returns) * 100:.1f}%",
            f"{strategy_returns.mean() * 100:.3f}%",
            f"{strategy_returns.max() * 100:.2f}%",
            f"{strategy_returns.min() * 100:.2f}%",
            f"{len(strategy_returns)}",
        ],
        '基准': [
            f"{(benchmark_cum.iloc[-1] - 1) * 100:.2f}%",
            f"{(benchmark_returns.mean() * 252) * 100:.2f}%",
            f"{(benchmark_returns.std() * np.sqrt(252)) * 100:.2f}%",
            f"{(benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252)):.2f}",
            f"{((benchmark_cum / benchmark_cum.cummax() - 1).min()) * 100:.2f}%",
            f"{(benchmark_returns > 0).sum() / len(benchmark_returns) * 100:.1f}%",
            f"{benchmark_returns.mean() * 100:.3f}%",
            f"{benchmark_returns.max() * 100:.2f}%",
            f"{benchmark_returns.min() * 100:.2f}%",
            f"{len(benchmark_returns)}",
        ],
    }

    summary_df = pd.DataFrame(metrics)
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 性能摘要表已保存：{output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='生成 ETF 策略回测可视化报告')
    parser.add_argument('--daily-csv', default=str(REPORTS / 'ml_auto_tune_until_baseline.csv'),
                        help='日度收益 CSV 文件路径')
    parser.add_argument('--output-dir', default=str(CHARTS_DIR),
                        help='图表输出目录')
    parser.add_argument('--rolling-window', type=int, default=60,
                        help='滚动窗口大小（天）')

    args = parser.parse_args()

    daily_csv = Path(args.daily_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("📊 ETF 策略回测可视化报告生成器")
    print("="*80)
    print(f"📁 输入文件：{daily_csv}")
    print(f"📁 输出目录：{output_dir}")
    print("="*80 + "\n")

    if not daily_csv.exists():
        print(f"❌ 错误：找不到输入文件 {daily_csv}")
        return 1

    try:
        # 加载数据
        print("🔄 加载回测数据...")
        df = load_backtest_data(daily_csv)
        print(f"✅ 已加载 {len(df)} 天的回测数据\n")

        # 生成各类图表
        print("🎨 生成可视化图表...\n")

        plot_cumulative_returns(df, output_dir / 'cumulative_returns.png')
        plot_daily_returns_distribution(df, output_dir / 'daily_returns_distribution.png')
        plot_drawdown(df, output_dir / 'drawdown.png')
        plot_monthly_returns_heatmap(df, output_dir / 'monthly_returns_heatmap.png')
        plot_win_rate_pie(df, output_dir / 'win_rate_pie.png')
        plot_rolling_metrics(df, output_dir / 'rolling_metrics.png', window=args.rolling_window)

        # 生成摘要表
        print("\n📋 生成性能摘要表...\n")
        generate_summary_table(df, output_dir / 'performance_summary.csv')

        print("\n" + "="*80)
        print("✅ 所有报告元素生成完成！")
        print("="*80)
        print(f"\n📊 生成的文件：")
        print(f"  1. 累计收益率曲线：{output_dir / 'cumulative_returns.png'}")
        print(f"  2. 日收益率分布：{output_dir / 'daily_returns_distribution.png'}")
        print(f"  3. 回撤曲线：{output_dir / 'drawdown.png'}")
        print(f"  4. 月度收益热力图：{output_dir / 'monthly_returns_heatmap.png'}")
        print(f"  5. 胜率统计饼图：{output_dir / 'win_rate_pie.png'}")
        print(f"  6. 滚动指标图：{output_dir / 'rolling_metrics.png'}")
        print(f"  7. 性能摘要表：{output_dir / 'performance_summary.csv'}\n")

        return 0

    except Exception as exc:
        print(f"\n❌ 错误：{exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
