#!/usr/bin/env python3
"""对比 LSTM-Transformer 与其他记忆模型的性能"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Source Han Sans CN', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 报告目录
REPORTS_DIR = Path("reports")

def load_results():
    """加载所有模型的结果"""
    results = {}

    # 加载记忆模型结果
    memory_summary = REPORTS_DIR / "memory_models" / "memory_models_summary.csv"
    if memory_summary.exists():
        memory_df = pd.read_csv(memory_summary)
        for _, row in memory_df.iterrows():
            results[row['model']] = {
                'test_directional_accuracy': row['test_directional_accuracy'],
                'test_spearman_ic': row['test_spearman_ic'],
                'test_rmse': row['test_rmse'],
            }

    # 加载 LSTM-Transformer 结果
    lstm_transformer_summary = REPORTS_DIR / "lstm_transformer" / "lstm_transformer_summary.csv"
    if lstm_transformer_summary.exists():
        lt_df = pd.read_csv(lstm_transformer_summary)
        results['lstm_transformer'] = {
            'test_directional_accuracy': lt_df['test_directional_accuracy'].values[0],
            'test_spearman_ic': lt_df['test_spearman_ic'].values[0],
            'test_rmse': lt_df['test_rmse'].values[0],
        }

    return results


def create_comparison_chart(results, output_path):
    """创建对比图表"""
    if not results:
        print("没有可用的结果数据")
        return

    # 转换为 DataFrame
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'model': model_name,
            'directional_accuracy': metrics['test_directional_accuracy'],
            'spearman_ic': metrics['test_spearman_ic'],
            'rmse': metrics['test_rmse'],
        })

    df = pd.DataFrame(comparison_data)

    # 创建多子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 颜色映射
    colors = {
        'lstm': '#3498db',
        'gru': '#e74c3c',
        'bilstm': '#f39c12',
        'attention_lstm': '#9b59b6',
        'lstm_transformer': '#2ecc71'
    }

    model_colors = [colors.get(m, '#95a5a6') for m in df['model']]

    # 1. 方向准确率
    ax1 = axes[0]
    bars1 = ax1.barh(df['model'], df['directional_accuracy'], color=model_colors)
    ax1.set_xlabel('准确率', fontsize=12)
    ax1.set_title('方向准确率对比', fontsize=14, fontweight='bold')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='随机基线')
    ax1.legend()

    # 添加数值标签
    for i, v in enumerate(df['directional_accuracy']):
        ax1.text(v + 0.01, i, f'{v:.4f}', va='center')

    # 2. Spearman IC
    ax2 = axes[1]
    bars2 = ax2.barh(df['model'], df['spearman_ic'], color=model_colors)
    ax2.set_xlabel('Spearman IC', fontsize=12)
    ax2.set_title('预测相关性对比', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 添加数值标签
    for i, v in enumerate(df['spearman_ic']):
        ax2.text(v + 0.005, i, f'{v:.4f}', va='center')

    # 3. RMSE (越小越好)
    ax3 = axes[2]
    bars3 = ax3.barh(df['model'], df['rmse'], color=model_colors)
    ax3.set_xlabel('RMSE', fontsize=12)
    ax3.set_title('预测误差对比（越小越好）', fontsize=14, fontweight='bold')

    # 添加数值标签
    for i, v in enumerate(df['rmse']):
        ax3.text(v + 0.0005, i, f'{v:.6f}', va='center')

    plt.tight_layout()

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 对比图表已保存：{output_path}")

    return df


def print_summary(df):
    """打印总结"""
    print("\n" + "="*80)
    print("📊 模型性能对比总结")
    print("="*80)

    # 按准确率排序
    df_sorted = df.sort_values('directional_accuracy', ascending=False)

    print("\n按方向准确率排序：")
    for i, row in enumerate(df_sorted.itertuples(), 1):
        print(f"{i}. {row.model.upper():20s} - 准确率：{row.directional_accuracy:.4f}, "
              f"IC: {row.spearman_ic:.4f}, RMSE: {row.rmse:.6f}")

    # 找出最佳模型
    best_acc_model = df_sorted.iloc[0]
    print(f"\n🏆 最佳模型（准确率）: {best_acc_model['model'].upper()}")
    print(f"   准确率：{best_acc_model['directional_accuracy']:.4f}")
    print(f"   Spearman IC: {best_acc_model['spearman_ic']:.4f}")
    print(f"   RMSE: {best_acc_model['rmse']:.6f}")

    # 如果 lstm_transformer 在结果中，显示其排名
    if 'lstm_transformer' in df['model'].values:
        lt_row = df[df['model'] == 'lstm_transformer'].iloc[0]
        lt_rank = df_sorted[df_sorted['model'] == 'lstm_transformer'].index[0] + 1
        print(f"\n🤖 LSTM-Transformer 排名：#{lt_rank}")
        print(f"   准确率：{lt_row['directional_accuracy']:.4f}")
        print(f"   Spearman IC: {lt_row['spearman_ic']:.4f}")
        print(f"   RMSE: {lt_row['rmse']:.6f}")

    print("\n" + "="*80)


def main():
    print("🔍 加载模型结果...")
    results = load_results()

    if not results:
        print("❌ 没有找到任何模型结果")
        return 1

    print(f"✓ 加载了 {len(results)} 个模型的结果")

    # 创建对比图表
    charts_dir = REPORTS_DIR / "charts" / "comparison"
    output_path = charts_dir / "all_models_comparison.svg"

    df = create_comparison_chart(results, output_path)

    if df is not None:
        # 打印总结
        print_summary(df)

        # 保存对比表格
        summary_path = REPORTS_DIR / "all_models_comparison.csv"
        df.to_csv(summary_path, index=False)
        print(f"\n💾 对比表格已保存：{summary_path}")

    print("\n✅ 对比完成！")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
