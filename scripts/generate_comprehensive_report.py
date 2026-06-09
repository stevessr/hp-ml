#!/usr/bin/env python3
"""生成所有 Memory Transformer 模型的综合对比报告"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Source Han Sans CN', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 报告目录
REPORTS_DIR = Path("reports")

def load_all_results():
    """加载所有模型结果"""
    all_results = []

    # 加载 Memory Transformer 变种
    variants_path = REPORTS_DIR / "memory_transformer_variants" / "memory_transformer_variants_summary.csv"
    if variants_path.exists():
        df = pd.read_csv(variants_path)
        for _, row in df.iterrows():
            all_results.append({
                'model': row['model'],
                'category': 'Memory Transformer',
                'test_directional_accuracy': row['test_directional_accuracy'],
                'test_spearman_ic': row['test_spearman_ic'],
                'test_rmse': row['test_rmse']
            })

    # 加载之前的基础模型
    comparison_path = REPORTS_DIR / "all_models_comparison.csv"
    if comparison_path.exists():
        df = pd.read_csv(comparison_path)
        for _, row in df.iterrows():
            # 跳过重复的模型
            if row['model'] not in [r['model'] for r in all_results]:
                all_results.append({
                    'model': row['model'],
                    'category': 'Baseline',
                    'test_directional_accuracy': row['directional_accuracy'],
                    'test_spearman_ic': row['spearman_ic'],
                    'test_rmse': row['rmse']
                })

    return pd.DataFrame(all_results)


def create_comprehensive_comparison(df, output_dir):
    """创建综合对比图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按准确率排序
    df = df.sort_values('test_directional_accuracy', ascending=True)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # 颜色映射
    colors = {
        'Memory Transformer': '#2ecc71',
        'Baseline': '#3498db'
    }
    bar_colors = [colors[cat] for cat in df['category']]

    # 1. 方向准确率对比
    ax1 = axes[0, 0]
    bars1 = ax1.barh(df['model'], df['test_directional_accuracy'], color=bar_colors)
    ax1.set_xlabel('准确率', fontsize=12)
    ax1.set_title('方向准确率对比', fontsize=14, fontweight='bold')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='随机基线')
    ax1.legend()

    # 添加数值标签
    for i, v in enumerate(df['test_directional_accuracy']):
        ax1.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

    # 2. Spearman IC 对比
    ax2 = axes[0, 1]
    bars2 = ax2.barh(df['model'], df['test_spearman_ic'], color=bar_colors)
    ax2.set_xlabel('Spearman IC', fontsize=12)
    ax2.set_title('预测相关性对比', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    for i, v in enumerate(df['test_spearman_ic']):
        ax2.text(v + 0.01 if v >= 0 else v - 0.01, i, f'{v:.4f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=9)

    # 3. RMSE 对比（越小越好）
    ax3 = axes[1, 0]
    bars3 = ax3.barh(df['model'], df['test_rmse'], color=bar_colors)
    ax3.set_xlabel('RMSE', fontsize=12)
    ax3.set_title('预测误差对比（越小越好）', fontsize=14, fontweight='bold')

    for i, v in enumerate(df['test_rmse']):
        ax3.text(v + 0.0002, i, f'{v:.6f}', va='center', fontsize=9)

    # 4. 综合散点图（准确率 vs IC）
    ax4 = axes[1, 1]
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        ax4.scatter(cat_df['test_spearman_ic'], cat_df['test_directional_accuracy'],
                   label=category, s=150, alpha=0.7, c=colors[category])

        # 添加模型标签
        for _, row in cat_df.iterrows():
            ax4.annotate(row['model'],
                        (row['test_spearman_ic'], row['test_directional_accuracy']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax4.set_xlabel('Spearman IC', fontsize=12)
    ax4.set_ylabel('方向准确率', fontsize=12)
    ax4.set_title('准确率 vs 预测相关性', fontsize=14, fontweight='bold')
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    output_path = output_dir / "comprehensive_memory_transformer_comparison.svg"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 综合对比图表已保存：{output_path}")

    plt.close()


def print_summary(df):
    """打印总结"""
    print("\n" + "="*80)
    print("📊 所有模型综合对比总结")
    print("="*80)

    # 按准确率排序
    df_sorted = df.sort_values('test_directional_accuracy', ascending=False)

    print("\n按方向准确率排序：\n")
    for i, row in enumerate(df_sorted.itertuples(), 1):
        emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        category_badge = "[MT]" if row.category == "Memory Transformer" else "[BL]"
        print(f"{emoji} {i:2d}. {category_badge} {row.model.upper():25s} - "
              f"准确率：{row.test_directional_accuracy:.4f}, "
              f"IC: {row.test_spearman_ic:.4f}, "
              f"RMSE: {row.test_rmse:.6f}")

    # 统计
    print("\n" + "="*80)
    print("📈 性能统计")
    print("="*80)

    mt_models = df_sorted[df_sorted['category'] == 'Memory Transformer']
    baseline_models = df_sorted[df_sorted['category'] == 'Baseline']

    print(f"\nMemory Transformer 模型数量：{len(mt_models)}")
    print(f"基线模型数量：{len(baseline_models)}")

    print(f"\n最佳 Memory Transformer: {mt_models.iloc[0]['model'].upper()}")
    print(f"  准确率：{mt_models.iloc[0]['test_directional_accuracy']:.4f}")
    print(f"  Spearman IC: {mt_models.iloc[0]['test_spearman_ic']:.4f}")

    print(f"\n最佳基线模型：{baseline_models.iloc[0]['model'].upper()}")
    print(f"  准确率：{baseline_models.iloc[0]['test_directional_accuracy']:.4f}")
    print(f"  Spearman IC: {baseline_models.iloc[0]['test_spearman_ic']:.4f}")

    # Top 3 统计
    top3 = df_sorted.head(3)
    mt_in_top3 = len(top3[top3['category'] == 'Memory Transformer'])
    print(f"\nTop 3 中 Memory Transformer 占比：{mt_in_top3}/3")

    print("\n" + "="*80)


def main():
    print("🔍 加载所有模型结果...")
    df = load_all_results()

    if df.empty:
        print("❌ 没有找到模型结果")
        return 1

    print(f"✓ 加载了 {len(df)} 个模型的结果")

    # 创建对比图表
    charts_dir = REPORTS_DIR / "charts" / "comprehensive"
    create_comprehensive_comparison(df, charts_dir)

    # 打印总结
    print_summary(df)

    # 保存综合对比表格
    output_path = REPORTS_DIR / "comprehensive_model_comparison.csv"
    df_sorted = df.sort_values('test_directional_accuracy', ascending=False)
    df_sorted.to_csv(output_path, index=False)
    print(f"\n💾 综合对比表格已保存：{output_path}")

    print("\n✅ 对比报告生成完成！")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
