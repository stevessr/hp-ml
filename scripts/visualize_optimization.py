#!/usr/bin/env python3
"""生成注意力窗口优化分析图表

展示注意力窗口大小与准确度的关系，以及其他优化因素的影响
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import REPORTS_DIR

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Source Han Sans CN", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def create_attention_window_analysis(results_df: pd.DataFrame, output_dir: Path) -> None:
    """创建注意力窗口分析图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 注意力窗口 vs 准确率（主图）
    print("📊 生成注意力窗口-准确率关系图...")

    fig, ax = plt.subplots(figsize=(14, 8))

    # 按序列长度分组
    seq_data = results_df.groupby('seq_length').agg({
        'test_accuracy': ['mean', 'max', 'min'],
        'test_ic': 'mean'
    }).reset_index()

    seq_lengths = seq_data['seq_length'].values
    acc_mean = seq_data['test_accuracy']['mean'].values * 100
    acc_max = seq_data['test_accuracy']['max'].values * 100
    acc_min = seq_data['test_accuracy']['min'].values * 100

    # 主折线图
    ax.plot(seq_lengths, acc_mean, 'o-', linewidth=3, markersize=12,
            color='#FF6B6B', label='平均准确率', zorder=3)

    # 填充区域（显示范围）
    ax.fill_between(seq_lengths, acc_min, acc_max,
                    alpha=0.3, color='#FF6B6B', label='准确率范围')

    # 标注最大值点
    max_idx = acc_mean.argmax()
    max_seq = seq_lengths[max_idx]
    max_acc = acc_mean[max_idx]
    ax.scatter([max_seq], [max_acc], s=500, color='red', marker='*',
              edgecolors='darkred', linewidth=2, zorder=5, label=f'最佳窗口: {max_seq}步')
    ax.annotate(f'{max_acc:.2f}%', (max_seq, max_acc),
               textcoords="offset points", xytext=(0, 15),
               ha='center', fontsize=12, fontweight='bold', color='red')

    # 数值标签
    for sl, acc in zip(seq_lengths, acc_mean):
        if sl != max_seq:
            ax.annotate(f'{acc:.2f}%', (sl, acc),
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=9)

    ax.set_xlabel('注意力窗口大小（序列长度）', fontsize=13, fontweight='bold')
    ax.set_ylabel('测试集准确率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('注意力窗口大小 vs 准确率\n更长的窗口 = 更好的性能？',
                fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(50, max(acc_max) + 5)

    # 添加80%目标线
    ax.axhline(y=80, color='green', linestyle=':', linewidth=2,
              alpha=0.7, label='目标: 80%')

    plt.tight_layout()
    plt.savefig(output_dir / "attention_window_vs_accuracy.svg", bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: attention_window_vs_accuracy.svg")

    # 2. 多因素影响分析（热力图）
    print("📊 生成多因素影响热力图...")

    # 创建透视表
    pivot_data = results_df.pivot_table(
        values='test_accuracy',
        index='units',
        columns='seq_length',
        aggfunc='max'
    )

    if not pivot_data.empty:
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(pivot_data.values * 100, cmap='RdYlGn', aspect='auto',
                      vmin=50, vmax=80)

        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticklabels(pivot_data.index)

        ax.set_xlabel('序列长度（注意力窗口）', fontsize=12, fontweight='bold')
        ax.set_ylabel('LSTM单元数', fontsize=12, fontweight='bold')
        ax.set_title('准确率热力图：序列长度 × 单元数\n颜色越绿越好',
                    fontsize=14, fontweight='bold', pad=20)

        # 添加数值标签
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                val = pivot_data.values[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val*100:.1f}%',
                                 ha="center", va="center",
                                 color="black" if val < 0.65 else "white",
                                 fontsize=9, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('准确率 (%)', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / "hyperparameter_heatmap.svg", bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: hyperparameter_heatmap.svg")

    # 3. 优化历程图
    print("📊 生成优化历程图...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # 按时间顺序排序（假设results_df已按实验顺序）
    results_sorted = results_df.sort_index()
    x = np.arange(len(results_sorted))
    accuracies = results_sorted['test_accuracy'].values * 100

    # 绘制优化曲线
    ax.plot(x, accuracies, 'o-', linewidth=2, markersize=6,
           color='#4ECDC4', alpha=0.7)

    # 标注最佳点
    best_idx = accuracies.argmax()
    best_acc = accuracies[best_idx]
    ax.scatter([best_idx], [best_acc], s=300, color='red', marker='*',
              edgecolors='darkred', linewidth=2, zorder=5)
    ax.annotate(f'最佳: {best_acc:.2f}%\n实验 #{best_idx+1}',
               (best_idx, best_acc),
               textcoords="offset points", xytext=(0, 20),
               ha='center', fontsize=10, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # 添加趋势线
    z = np.polyfit(x, accuracies, 2)
    p = np.poly1d(z)
    ax.plot(x, p(x), "--", linewidth=2, color='gray', alpha=0.5, label='趋势线')

    ax.set_xlabel('实验序号', fontsize=12, fontweight='bold')
    ax.set_ylabel('测试集准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('优化历程\n随着实验推进，性能如何变化？',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=80, color='green', linestyle=':', linewidth=2,
              alpha=0.7, label='目标: 80%')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "optimization_progress.svg", bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: optimization_progress.svg")

    # 4. Dropout vs 准确率
    print("📊 生成Dropout影响图...")

    dropout_data = results_df.groupby('dropout')['test_accuracy'].agg(['mean', 'max']).reset_index()

    if len(dropout_data) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        x_pos = np.arange(len(dropout_data))
        ax.bar(x_pos, dropout_data['mean'].values * 100,
              color='skyblue', alpha=0.7, label='平均准确率')
        ax.scatter(x_pos, dropout_data['max'].values * 100,
                  color='red', s=100, zorder=5, label='最高准确率')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{d:.1f}' for d in dropout_data['dropout']])
        ax.set_xlabel('Dropout率', fontsize=12, fontweight='bold')
        ax.set_ylabel('测试集准确率 (%)', fontsize=12, fontweight='bold')
        ax.set_title('Dropout率对准确率的影响',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / "dropout_impact.svg", bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: dropout_impact.svg")


def main() -> int:
    results_file = REPORTS_DIR / "optimization_80" / "optimization_results.csv"

    if not results_file.exists():
        print(f"❌ 结果文件不存在: {results_file}")
        print("   请先运行 optimize_to_80.py")
        return 1

    print("\n" + "="*80)
    print("📊 生成注意力窗口优化分析图表")
    print("="*80 + "\n")

    results_df = pd.read_csv(results_file)
    charts_dir = REPORTS_DIR / "optimization_80" / "charts"

    create_attention_window_analysis(results_df, charts_dir)

    print("\n" + "="*80)
    print("✅ 图表生成完成！")
    print("="*80)
    print(f"\n📁 输出目录: {charts_dir}")
    print("\n生成的图表:")
    print("  1. attention_window_vs_accuracy.svg  - 注意力窗口-准确率关系")
    print("  2. hyperparameter_heatmap.svg        - 超参数热力图")
    print("  3. optimization_progress.svg         - 优化历程")
    print("  4. dropout_impact.svg                - Dropout影响")
    print("\n" + "="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
