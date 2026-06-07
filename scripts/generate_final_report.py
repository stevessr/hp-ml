#!/usr/bin/env python3
"""生成包含所有模型的最终综合对比报告"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Source Han Sans CN', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

REPORTS_DIR = Path("reports")


def load_all_model_results():
    """加载所有模型结果"""
    all_results = []

    # 1. N-Gram 模型
    ngram_path = REPORTS_DIR / "ngram_models" / "ngram_models_summary.csv"
    if ngram_path.exists():
        df = pd.read_csv(ngram_path)
        for _, row in df.iterrows():
            all_results.append({
                'model': row['model'],
                'category': 'N-Gram CNN',
                'test_directional_accuracy': row['test_directional_accuracy'],
                'test_spearman_ic': row['test_spearman_ic'],
                'test_rmse': row['test_rmse']
            })

    # 2. Memory Transformer
    mt_path = REPORTS_DIR / "memory_transformer_variants" / "memory_transformer_variants_summary.csv"
    if mt_path.exists():
        df = pd.read_csv(mt_path)
        for _, row in df.iterrows():
            all_results.append({
                'model': row['model'],
                'category': 'Memory Transformer',
                'test_directional_accuracy': row['test_directional_accuracy'],
                'test_spearman_ic': row['test_spearman_ic'],
                'test_rmse': row['test_rmse']
            })

    # 3. 基线模型
    baseline_path = REPORTS_DIR / "all_models_comparison.csv"
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        for _, row in df.iterrows():
            if row['model'] not in [r['model'] for r in all_results]:
                all_results.append({
                    'model': row['model'],
                    'category': 'Baseline RNN',
                    'test_directional_accuracy': row['directional_accuracy'],
                    'test_spearman_ic': row['spearman_ic'],
                    'test_rmse': row['rmse']
                })

    return pd.DataFrame(all_results)


def create_final_comparison(df, output_dir):
    """创建最终综合对比图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values('test_directional_accuracy', ascending=True)

    fig, axes = plt.subplots(2, 2, figsize=(22, 16))

    # 颜色映射
    colors = {
        'N-Gram CNN': '#e74c3c',
        'Memory Transformer': '#2ecc71',
        'Baseline RNN': '#3498db'
    }
    bar_colors = [colors[cat] for cat in df['category']]

    # 1. 方向准确率
    ax1 = axes[0, 0]
    bars1 = ax1.barh(df['model'], df['test_directional_accuracy'], color=bar_colors)
    ax1.set_xlabel('准确率', fontsize=14, fontweight='bold')
    ax1.set_title('所有模型方向准确率对比', fontsize=16, fontweight='bold')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='随机基线 (50%)')
    ax1.legend(fontsize=11)
    ax1.grid(axis='x', alpha=0.3)

    for i, v in enumerate(df['test_directional_accuracy']):
        ax1.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10)

    # 2. Spearman IC
    ax2 = axes[0, 1]
    bars2 = ax2.barh(df['model'], df['test_spearman_ic'], color=bar_colors)
    ax2.set_xlabel('Spearman IC', fontsize=14, fontweight='bold')
    ax2.set_title('预测相关性对比', fontsize=16, fontweight='bold')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax2.grid(axis='x', alpha=0.3)

    for i, v in enumerate(df['test_spearman_ic']):
        offset = 0.01 if v >= 0 else -0.01
        ha = 'left' if v >= 0 else 'right'
        ax2.text(v + offset, i, f'{v:.4f}', va='center', ha=ha, fontsize=10)

    # 3. RMSE
    ax3 = axes[1, 0]
    bars3 = ax3.barh(df['model'], df['test_rmse'], color=bar_colors)
    ax3.set_xlabel('RMSE', fontsize=14, fontweight='bold')
    ax3.set_title('预测误差对比（越小越好）', fontsize=16, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    for i, v in enumerate(df['test_rmse']):
        ax3.text(v + 0.0001, i, f'{v:.6f}', va='center', fontsize=10)

    # 4. 分类散点图
    ax4 = axes[1, 1]
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        ax4.scatter(cat_df['test_spearman_ic'], cat_df['test_directional_accuracy'],
                   label=category, s=200, alpha=0.7, c=colors[category], edgecolors='black', linewidth=1.5)

        for _, row in cat_df.iterrows():
            ax4.annotate(row['model'],
                        (row['test_spearman_ic'], row['test_directional_accuracy']),
                        xytext=(7, 7), textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax4.set_xlabel('Spearman IC', fontsize=14, fontweight='bold')
    ax4.set_ylabel('方向准确率', fontsize=14, fontweight='bold')
    ax4.set_title('准确率 vs 预测相关性（按类别）', fontsize=16, fontweight='bold')
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=2)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=2)
    ax4.legend(fontsize=12, loc='lower right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "final_all_models_comparison.svg"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 最终综合对比图已保存: {output_path}")

    plt.close()


def print_final_summary(df):
    """打印最终总结"""
    print("\n" + "="*80)
    print("🏆 所有模型最终排名")
    print("="*80)

    df_sorted = df.sort_values('test_directional_accuracy', ascending=False)

    print("\n按测试集准确率排序：\n")
    for i, row in enumerate(df_sorted.itertuples(), 1):
        emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        cat_badge = "[NG]" if row.category == "N-Gram CNN" else "[MT]" if row.category == "Memory Transformer" else "[BL]"
        print(f"{emoji} {i:2d}. {cat_badge} {row.model.upper():25s} - "
              f"准确率: {row.test_directional_accuracy:.4f}, "
              f"IC: {row.test_spearman_ic:.4f}, "
              f"RMSE: {row.test_rmse:.6f}")

    # 分类统计
    print("\n" + "="*80)
    print("📊 分类统计")
    print("="*80)

    for category in ['N-Gram CNN', 'Memory Transformer', 'Baseline RNN']:
        cat_df = df_sorted[df_sorted['category'] == category]
        if len(cat_df) > 0:
            print(f"\n{category}:")
            print(f"  模型数量: {len(cat_df)}")
            print(f"  平均准确率: {cat_df['test_directional_accuracy'].mean():.4f}")
            print(f"  平均 IC: {cat_df['test_spearman_ic'].mean():.4f}")
            print(f"  平均 RMSE: {cat_df['test_rmse'].mean():.6f}")
            print(f"  最佳模型: {cat_df.iloc[0]['model'].upper()} ({cat_df.iloc[0]['test_directional_accuracy']:.4f})")

    # Top 5 统计
    print("\n" + "="*80)
    print("🎯 Top 5 分析")
    print("="*80)

    top5 = df_sorted.head(5)
    ng_in_top5 = len(top5[top5['category'] == 'N-Gram CNN'])
    mt_in_top5 = len(top5[top5['category'] == 'Memory Transformer'])
    bl_in_top5 = len(top5[top5['category'] == 'Baseline RNN'])

    print(f"\nTop 5 构成:")
    print(f"  N-Gram CNN: {ng_in_top5} 个")
    print(f"  Memory Transformer: {mt_in_top5} 个")
    print(f"  Baseline RNN: {bl_in_top5} 个")

    print("\n" + "="*80)


def main():
    print("🔍 加载所有模型结果...")
    df = load_all_model_results()

    if df.empty:
        print("❌ 没有找到模型结果")
        return 1

    print(f"✓ 加载了 {len(df)} 个模型的结果")
    print(f"  - N-Gram CNN: {len(df[df['category'] == 'N-Gram CNN'])}")
    print(f"  - Memory Transformer: {len(df[df['category'] == 'Memory Transformer'])}")
    print(f"  - Baseline RNN: {len(df[df['category'] == 'Baseline RNN'])}")

    # 创建最终对比图
    charts_dir = REPORTS_DIR / "charts" / "final"
    create_final_comparison(df, charts_dir)

    # 打印总结
    print_final_summary(df)

    # 保存最终对比表格
    output_path = REPORTS_DIR / "final_all_models_comparison.csv"
    df_sorted = df.sort_values('test_directional_accuracy', ascending=False)
    df_sorted.to_csv(output_path, index=False)
    print(f"\n💾 最终对比表格已保存: {output_path}")

    print("\n✅ 最终综合报告生成完成！")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
