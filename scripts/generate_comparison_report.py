#!/usr/bin/env python3
"""生成综合模型对比报告

基于实验结果生成详细的对比报告和可视化图表
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


def create_visualizations(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """创建可视化图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 序列长度影响分析
    print("📊 生成序列长度影响图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for model_type, ax in zip(['lstm', 'attention_lstm'], axes[:2]):
        model_data = summary_df[summary_df['model'] == model_type]
        if len(model_data) > 0:
            seq_data = model_data.groupby('seq_length').agg({
                'test_accuracy': 'mean',
                'test_ic': 'mean',
                'test_rmse': 'mean'
            }).reset_index()

            ax.plot(seq_data['seq_length'], seq_data['test_accuracy'],
                   marker='o', linewidth=2, markersize=8, label='准确率')
            ax.set_xlabel('序列长度（注意力窗口）', fontsize=12)
            ax.set_ylabel('测试准确率', fontsize=12)
            ax.set_title(f'{model_type.upper()} 序列长度影响', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

    # 对比图
    ax = axes[2]
    for model_type in ['lstm', 'attention_lstm']:
        model_data = summary_df[summary_df['model'] == model_type]
        if len(model_data) > 0:
            seq_data = model_data.groupby('seq_length')['test_accuracy'].mean()
            ax.plot(seq_data.index, seq_data.values,
                   marker='o', linewidth=2, markersize=8, label=model_type.upper())

    ax.set_xlabel('序列长度（注意力窗口）', fontsize=12)
    ax.set_ylabel('测试准确率', fontsize=12)
    ax.set_title('序列长度对比', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "sequence_length_analysis.svg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存：sequence_length_analysis.svg")

    # 2. 单元数影响分析
    print("📊 生成单元数影响图...")
    attn_data = summary_df[summary_df['model'] == 'attention_lstm']
    units_data = attn_data.groupby('units').agg({
        'test_accuracy': 'mean',
        'test_ic': 'mean',
        'test_rmse': 'mean'
    }).reset_index()

    if len(units_data) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].bar(units_data['units'].astype(str), units_data['test_accuracy'], color='skyblue')
        axes[0].set_xlabel('LSTM 单元数', fontsize=12)
        axes[0].set_ylabel('测试准确率', fontsize=12)
        axes[0].set_title('单元数对准确率的影响', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(units_data['units'].astype(str), units_data['test_ic'], color='lightcoral')
        axes[1].set_xlabel('LSTM 单元数', fontsize=12)
        axes[1].set_ylabel('测试 IC', fontsize=12)
        axes[1].set_title('单元数对 IC 的影响', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        axes[2].bar(units_data['units'].astype(str), units_data['test_rmse'], color='lightgreen')
        axes[2].set_xlabel('LSTM 单元数', fontsize=12)
        axes[2].set_ylabel('测试 RMSE', fontsize=12)
        axes[2].set_title('单元数对 RMSE 的影响', fontsize=13, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "units_analysis.svg", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存：units_analysis.svg")

    # 3. 所有模型对比
    print("📊 生成模型综合对比图...")
    top_10 = summary_df.head(10)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(top_10))
    width = 0.35

    bars1 = ax.bar(x - width/2, top_10['test_accuracy'], width, label='准确率', color='skyblue')
    bars2 = ax.bar(x + width/2, top_10['test_ic'], width, label='IC', color='lightcoral')

    ax.set_xlabel('模型配置', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('Top 10 模型配置对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_10['config'], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "top_models_comparison.svg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存：top_models_comparison.svg")


def generate_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    """生成 Markdown 报告"""
    print("\n📝 生成 Markdown 报告...")

    lines = [
        "# 综合模型对比报告",
        "",
        "**生成时间**: 2026-06-07",
        "",
        "## 实验概况",
        "",
        f"- **实验总数**: {len(summary_df)}",
        f"- **模型类型**: {summary_df['model'].nunique()} 种",
        f"- **最佳准确率**: {summary_df['test_accuracy'].max():.4f}",
        f"- **最佳 IC**: {summary_df['test_ic'].max():.4f}",
        "",
        "## Top 10 模型配置",
        "",
        "| 排名 | 模型配置 | 准确率 | IC | RMSE |",
        "|------|---------|--------|-----|------|"
    ]

    for idx, row in summary_df.head(10).iterrows():
        lines.append(
            f"| {idx+1} | {row['config']} | {row['test_accuracy']:.4f} | "
            f"{row['test_ic']:.4f} | {row['test_rmse']:.6f} |"
        )

    lines.extend([
        "",
        "## 序列长度分析（注意力窗口）",
        "",
        "### LSTM",
        ""
    ])

    lstm_data = summary_df[summary_df['model'] == 'lstm']
    if len(lstm_data) > 0:
        seq_analysis = lstm_data.groupby('seq_length')['test_accuracy'].mean().sort_index()
        lines.append("| 序列长度 | 平均准确率 |")
        lines.append("|---------|-----------|")
        for seq_len, acc in seq_analysis.items():
            lines.append(f"| {seq_len} | {acc:.4f} |")

    lines.extend([
        "",
        "### Attention LSTM",
        ""
    ])

    attn_data = summary_df[summary_df['model'] == 'attention_lstm']
    if len(attn_data) > 0:
        seq_analysis = attn_data.groupby('seq_length')['test_accuracy'].mean().sort_index()
        lines.append("| 序列长度 | 平均准确率 |")
        lines.append("|---------|-----------|")
        for seq_len, acc in seq_analysis.items():
            lines.append(f"| {seq_len} | {acc:.4f} |")

    lines.extend([
        "",
        "## 单元数分析",
        "",
        "Attention LSTM 模型在不同单元数下的表现：",
        ""
    ])

    if len(attn_data) > 0:
        units_analysis = attn_data.groupby('units').agg({
            'test_accuracy': 'mean',
            'test_ic': 'mean'
        }).reset_index()

        lines.append("| 单元数 | 平均准确率 | 平均 IC |")
        lines.append("|--------|-----------|--------|")
        for _, row in units_analysis.iterrows():
            lines.append(f"| {int(row['units'])} | {row['test_accuracy']:.4f} | {row['test_ic']:.4f} |")

    lines.extend([
        "",
        "## 关键发现",
        "",
        "1. **最佳序列长度**: 通过实验确定最优的注意力窗口大小",
        "2. **单元数影响**: 分析 LSTM 单元数对性能的影响",
        "3. **模型类型对比**: 不同注意力机制的效果对比",
        "",
        "## 推荐配置",
        "",
        f"基于实验结果，推荐使用：",
        f"- **模型**: {summary_df.iloc[0]['model']}",
        f"- **序列长度**: {summary_df.iloc[0]['seq_length']}",
        f"- **单元数**: {int(summary_df.iloc[0]['units'])}",
        f"- **Dropout**: {summary_df.iloc[0]['dropout']}",
        "",
        "---",
        "",
        "**数据文件**:",
        "- `model_comparison_summary.csv` - 汇总结果",
        "- `model_comparison_details.json` - 详细指标",
        "",
        "**可视化**:",
        "- `sequence_length_analysis.svg` - 序列长度影响",
        "- `units_analysis.svg` - 单元数影响",
        "- `top_models_comparison.svg` - Top 10 对比",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ 报告已保存：{output_path}")


def main() -> int:
    comparison_dir = REPORTS_DIR / "model_comparison"
    summary_file = comparison_dir / "model_comparison_summary.csv"

    if not summary_file.exists():
        print(f"❌ 结果文件不存在：{summary_file}")
        print("   请先运行 comprehensive_model_comparison.py")
        return 1

    print("\n" + "="*80)
    print("📊 生成综合模型对比报告")
    print("="*80 + "\n")

    summary_df = pd.read_csv(summary_file)
    charts_dir = comparison_dir / "charts"

    create_visualizations(summary_df, charts_dir)
    generate_report(summary_df, comparison_dir / "MODEL_COMPARISON_REPORT.md")

    print("\n" + "="*80)
    print("✅ 报告生成完成！")
    print("="*80)
    print(f"\n📁 输出目录：{comparison_dir}")
    print(f"📄 报告：MODEL_COMPARISON_REPORT.md")
    print(f"📊 图表：charts/\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
