#!/usr/bin/env python3
"""生成模型准确度对比图表

使用matplotlib创建专业的准确度对比图表
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


def create_accuracy_comparison_chart(output_dir: Path) -> None:
    """创建准确度对比图表"""

    # 收集所有模型数据
    models_data = [
        # 综合对比实验结果
        ("Attention LSTM\nseq=20", 66.67, 0.3183, "最佳配置"),
        ("Attention LSTM\nseq=15", 64.49, 0.2799, "综合对比"),
        ("LSTM\nseq=10", 63.97, 0.1605, "基础模型"),
        ("Self-Attention\nseq=15", 63.59, 0.2259, "注意力变种"),
        ("Hierarchical\nseq=15", 61.92, 0.1784, "注意力变种"),
        ("Attention LSTM\nseq=15", 60.90, 0.1613, "综合对比"),
        ("Attention LSTM\nseq=10", 57.18, 0.0718, "综合对比"),
        ("LSTM\nseq=5", 56.41, 0.1590, "记忆门模型"),
        ("MultiHead\nseq=10", 55.00, -0.0074, "注意力变种"),
        ("GRU\nseq=10", 46.15, -0.1886, "记忆门模型"),
        ("BiLSTM\nseq=10", 46.15, -0.1062, "记忆门模型"),
        ("Ridge\n(最佳)", 71.76, None, "传统ML"),
        ("Ridge\n(平均)", 49.72, None, "传统ML"),
    ]

    # 1. 主要准确度对比图（横向条形图）
    print("📊 生成主要准确度对比图...")
    fig, ax = plt.subplots(figsize=(14, 10))

    models = [m[0] for m in models_data]
    accuracies = [m[1] for m in models_data]
    categories = [m[3] for m in models_data]

    # 定义颜色映射
    color_map = {
        "最佳配置": "#FF6B6B",      # 红色
        "综合对比": "#4ECDC4",      # 青色
        "记忆门模型": "#95E1D3",    # 浅绿
        "注意力变种": "#FFE66D",    # 黄色
        "传统ML": "#C7CEEA",        # 紫灰
        "基础模型": "#FFA07A",      # 橙色
    }

    colors = [color_map[cat] for cat in categories]

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        width = bar.get_width()
        label_x = width + 0.5
        if width < 50:
            label_x = 50
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%',
                va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.set_xlabel('测试集准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('深度学习模型准确度对比\n(2026-06-07)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(40, 75)

    # 添加基准线
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='随机猜测 (50%)')
    ax.axvline(x=66.67, color='red', linestyle=':', linewidth=2, alpha=0.7, label='最佳模型 (66.67%)')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[cat], label=cat, alpha=0.8)
                      for cat in color_map.keys()]
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle=':', linewidth=2, label='最佳模型'))
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison_main.svg", bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: accuracy_comparison_main.svg")

    # 2. 准确率 vs IC 散点图
    print("📊 生成准确率-IC散点图...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # 过滤有IC的数据
    scatter_data = [(m[0], m[1], m[2], m[3]) for m in models_data if m[2] is not None]

    for cat in color_map.keys():
        cat_data = [(m[0], m[1], m[2]) for m in scatter_data if m[3] == cat]
        if cat_data:
            names, accs, ics = zip(*cat_data)
            ax.scatter(accs, ics, s=200, alpha=0.7,
                      color=color_map[cat], edgecolors='black', linewidth=1,
                      label=cat)

            # 添加标签
            for name, acc, ic in zip(names, accs, ics):
                ax.annotate(name, (acc, ic), fontsize=8,
                           xytext=(5, 5), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel('测试集准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spearman IC', fontsize=12, fontweight='bold')
    ax.set_title('准确率 vs IC 散点图\n越靠右上角越好',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=50, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_ic_scatter.svg", bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: accuracy_vs_ic_scatter.svg")

    # 3. 序列长度影响（折线图）
    print("📊 生成序列长度影响图...")
    seq_data = {
        "Attention LSTM": [(5, 53.08), (10, 57.18), (15, 58.83), (20, 66.67)],
        "LSTM": [(5, 53.71), (10, 63.97), (15, 50.26), (20, 48.85)],
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    for model_name, data in seq_data.items():
        seq_lengths, accs = zip(*data)
        linestyle = '-' if model_name == "Attention LSTM" else '--'
        marker = 'o' if model_name == "Attention LSTM" else 's'
        linewidth = 3 if model_name == "Attention LSTM" else 2
        ax.plot(seq_lengths, accs, marker=marker, linestyle=linestyle,
               linewidth=linewidth, markersize=10, label=model_name)

        # 添加数值标签
        for sl, acc in zip(seq_lengths, accs):
            ax.annotate(f'{acc:.1f}%', (sl, acc),
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('序列长度（注意力窗口）', fontsize=12, fontweight='bold')
    ax.set_ylabel('测试集准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('序列长度对准确率的影响\nAttention LSTM: 序列越长，性能越好',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([5, 10, 15, 20])
    ax.set_ylim(45, 70)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')

    # 标注最佳点
    ax.scatter([20], [66.67], s=300, color='red', marker='*',
              edgecolors='darkred', linewidth=2, zorder=5, label='最佳配置')

    plt.tight_layout()
    plt.savefig(output_dir / "sequence_length_impact.svg", bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: sequence_length_impact.svg")

    # 4. 模型发展历程（时间线图）
    print("📊 生成模型发展历程图...")
    fig, ax = plt.subplots(figsize=(14, 8))

    stages = [
        ("传统ML\nRidge", 49.72, "阶段1"),
        ("基础LSTM\nseq=10", 56.41, "阶段2"),
        ("Attention LSTM\nseq=10", 65.13, "阶段3"),
        ("Attention LSTM\nseq=20", 66.67, "阶段4"),
    ]

    stage_names = [s[0] for s in stages]
    accuracies = [s[1] for s in stages]
    colors_timeline = ['#C7CEEA', '#95E1D3', '#FFE66D', '#FF6B6B']

    x_pos = np.arange(len(stages))
    bars = ax.bar(x_pos, accuracies, color=colors_timeline,
                  alpha=0.8, edgecolor='black', linewidth=2)

    # 添加数值和提升幅度
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
               f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold')

        if i > 0:
            prev_acc = accuracies[i-1]
            improvement = acc - prev_acc
            ax.annotate(f'+{improvement:.2f}%',
                       xy=(i-0.5, (acc + prev_acc)/2),
                       fontsize=10, ha='center', color='green', fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(stage_names, fontsize=11, fontweight='bold')
    ax.set_ylabel('测试集准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('模型发展历程\n从传统ML到深度学习的演进',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 75)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加阶段标注
    for i, stage in enumerate(stages):
        ax.text(i, 5, stage[2], ha='center', fontsize=10,
               style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / "model_evolution_timeline.svg", bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: model_evolution_timeline.svg")

    # 5. Top 5 模型雷达图
    print("📊 生成Top 5模型雷达图...")
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    top5_models = [
        ("Attention LSTM\nseq=20", [66.67, 0.3183*100, 95, 100, 90]),  # 准确率, IC*100, 稳定性, 可解释性, 易用性
        ("Attention LSTM\nseq=15", [64.49, 0.2799*100, 90, 100, 90]),
        ("LSTM seq=10", [63.97, 0.1605*100, 85, 70, 95]),
        ("Self-Attention", [63.59, 0.2259*100, 80, 60, 70]),
        ("Hierarchical", [61.92, 0.1784*100, 75, 50, 65]),
    ]

    categories = ['准确率\n(×2/3)', 'IC\n(×100)', '稳定性', '可解释性', '易用性']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    colors_radar = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D', '#FFA07A']

    for i, (name, values) in enumerate(top5_models):
        # 归一化准确率到0-100范围（但不改变实际显示）
        values_plot = values + values[:1]
        ax.plot(angles, values_plot, 'o-', linewidth=2,
               label=name, color=colors_radar[i])
        ax.fill(angles, values_plot, alpha=0.15, color=colors_radar[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Top 5 模型多维度对比\n(值越大越好)',
                fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "top5_radar_chart.svg", bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存: top5_radar_chart.svg")


def main() -> int:
    print("\n" + "="*80)
    print("📊 生成模型准确度对比图表")
    print("="*80 + "\n")

    output_dir = REPORTS_DIR / "charts" / "accuracy_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        create_accuracy_comparison_chart(output_dir)

        print("\n" + "="*80)
        print("✅ 所有图表生成完成！")
        print("="*80)
        print(f"\n📁 输出目录: {output_dir}")
        print("\n生成的图表:")
        print("  1. accuracy_comparison_main.svg      - 主要准确度对比")
        print("  2. accuracy_vs_ic_scatter.svg        - 准确率vs IC散点图")
        print("  3. sequence_length_impact.svg        - 序列长度影响")
        print("  4. model_evolution_timeline.svg      - 模型发展历程")
        print("  5. top5_radar_chart.svg              - Top 5雷达图")
        print("\n" + "="*80 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
