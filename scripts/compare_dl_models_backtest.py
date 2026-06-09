#!/usr/bin/env python
"""深度学习模型效果对比和收益率回测

功能：
1. 使用通达信数据源训练多个深度学习模型
2. 对比各模型的预测准确率（MAE、RMSE、方向准确率）
3. 回测各模型的投资收益率
4. 生成完整的对比报告和可视化图表
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hp_ml.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from hp_ml.tdx_data_source import fetch_etf_batch_tdx
from hp_ml.features import build_feature_panel
from hp_ml.universe import discover_broad_etfs
from hp_ml.data_pipeline import time_series_split
from hp_ml.models_extended import make_extended_model
from hp_ml.model import evaluate_predictions
from hp_ml.backtest import backtest_strategy, BacktestMetrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="深度学习模型效果对比和收益率回测")

    # 数据参数
    parser.add_argument("--start", default="20230101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260609", help="结束日期 YYYYMMDD")
    parser.add_argument("--max-etfs", type=int, default=3, help="每个指数族 ETF 数量")
    parser.add_argument("--horizon", type=int, default=5, help="预测周期（交易日）")

    # 模型选择（默认训练常用模型）
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "bilstm", "attention_lstm", "lstm_transformer"],
        choices=[
            "lstm",
            "bilstm",
            "attention_lstm",
            "multihead_attention_lstm",
            "self_attention_lstm",
            "hierarchical_attention_lstm",
            "transformer_xl",
            "memory_transformer",
            "gru_transformer",
            "lstm_transformer",
        ],
        help="要对比的模型列表",
    )

    # 训练参数
    parser.add_argument("--seq-length", type=int, default=20, help="序列长度")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")

    # 回测参数
    parser.add_argument("--top-k", type=int, default=3, help="每次选择前 K 只 ETF")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="交易成本")

    # 输出参数
    parser.add_argument("--output-dir", default=str(REPORTS_DIR / "dl_comparison"), help="输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新拉取数据")

    return parser


def prepare_data(args) -> tuple[pd.DataFrame, list[str], str, pd.DataFrame]:
    """准备训练数据"""
    print("=" * 80)
    print("📦 准备训练数据（使用通达信数据源）")
    print("=" * 80)

    # 1. 发现 ETF 候选池
    print("\n🔍 发现 ETF 候选池...")
    universe = discover_broad_etfs(
        max_per_family=args.max_etfs,
        min_amount=0.0,
        include_enhanced=False,
        include_style=False,
        force=args.force,
    )
    print(f"✓ 发现 {len(universe)} 只候选 ETF")

    # 2. 使用通达信拉取数据
    codes = universe["code"].astype(str).str.zfill(6).tolist()

    print(f"\n📡 使用通达信数据源拉取 {len(codes)} 只 ETF...")
    codes_with_prefix = []
    for code in codes:
        if code.startswith(("510", "511", "512", "513", "515", "516", "517", "560", "561", "562", "563", "588", "589")):
            codes_with_prefix.append(f"sh{code}")
        else:
            codes_with_prefix.append(f"sz{code}")

    histories = fetch_etf_batch_tdx(
        codes_with_prefix,
        start_date=args.start,
        end_date=args.end,
        cache=not args.force,
    )

    # 转换回原始代码格式
    histories = {code[2:]: df for code, df in histories.items()}
    print(f"✓ 成功拉取 {len(histories)} 只 ETF 数据")

    # 3. 构建特征面板
    print(f"\n🔧 构建特征面板...")

    # 为每个历史数据添加 code 列（如果没有的话）
    for code, df in histories.items():
        if 'code' not in df.columns:
            df['code'] = code

    panel, feature_cols, target_col = build_feature_panel(
        histories,
        universe=universe,
        horizon=args.horizon,
    )
    print(f"✓ 面板数据：{len(panel)} 行")
    print(f"✓ 特征数量：{len(feature_cols)}")
    print(f"✓ 目标变量：{target_col}")

    return panel, feature_cols, target_col, universe


def train_and_evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    args,
) -> dict:
    """训练并评估单个模型"""
    print(f"\n{'='*80}")
    print(f"🚀 训练模型：{model_name.upper()}")
    print(f"{'='*80}")

    try:
        # 构建模型参数（根据模型类型）
        model_kwargs = {
            "seq_length": args.seq_length,
            "learning_rate": 0.001,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dropout": 0.3,
            "early_stopping_patience": 10,
        }

        # 根据模型类型添加特定参数
        if model_name in ["lstm", "bilstm", "gru_transformer"]:
            model_kwargs["units"] = 64

        if model_name in ["attention_lstm", "self_attention_lstm", "hierarchical_attention_lstm"]:
            model_kwargs["units"] = 64

        if model_name in ["multihead_attention_lstm", "lstm_transformer", "transformer_xl", "memory_transformer"]:
            model_kwargs["num_heads"] = 4
            model_kwargs["ff_dim"] = 128
            model_kwargs["num_transformer_blocks"] = 2

        # 构建模型
        model = make_extended_model(model_name, **model_kwargs)

        # 准备训练数据（深度学习模型需要完整的 DataFrame，包含 code 和 date 列）
        # 确保包含必要的列
        required_cols = feature_cols + ['code', 'date']
        X_train = train_df[[c for c in required_cols if c in train_df.columns]]
        y_train = train_df[target_col]
        X_test = test_df[[c for c in required_cols if c in test_df.columns]]
        y_test = test_df[target_col]

        print(f"训练集：{len(X_train)} 样本")
        print(f"测试集：{len(X_test)} 样本")
        print(f"开始训练...")

        model.fit(X_train, y_train)

        # 预测
        print(f"生成预测...")
        test_pred = model.predict(X_test)

        # 评估
        test_metrics = {
            'mae': float(np.mean(np.abs(y_test.values - test_pred))),
            'rmse': float(np.sqrt(np.mean((y_test.values - test_pred) ** 2))),
            'direction_accuracy': float(np.mean((y_test.values > 0) == (test_pred > 0))),
        }

        print(f"\n测试集指标：")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  方向准确率：{test_metrics['direction_accuracy']:.2%}")

        # 准备回测数据
        test_df_with_pred = test_df.copy()
        test_df_with_pred[f"pred_{model_name}"] = test_pred

        return {
            "model_name": model_name,
            "model": model,
            "metrics": test_metrics,
            "predictions": test_df_with_pred,
            "success": True,
        }

    except Exception as e:
        print(f"\n✗ 训练失败：{e}")
        import traceback
        traceback.print_exc()

        return {
            "model_name": model_name,
            "success": False,
            "error": str(e),
        }


def run_backtest(
    predictions_df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    args,
) -> BacktestMetrics:
    """运行回测"""
    metrics, _ = backtest_strategy(
        predictions_df,
        pred_col=pred_col,
        target_col=target_col,
        top_k=args.top_k,
        min_pred_threshold=0.0,
        transaction_cost=args.transaction_cost,
    )
    return metrics


def create_comparison_visualizations(results: list[dict], output_dir: Path):
    """创建对比可视化"""
    print(f"\n📊 生成对比图表...")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 预测准确率对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('深度学习模型效果对比', fontsize=16, fontweight='bold')

    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print("✗ 没有成功训练的模型，跳过可视化")
        plt.close()
        return

    # MAE 对比
    models = [r["model_name"] for r in successful_results]
    maes = [r["metrics"]["mae"] for r in successful_results]

    axes[0, 0].barh(models, maes, color='steelblue')
    axes[0, 0].set_xlabel('MAE')
    axes[0, 0].set_title('平均绝对误差 (越小越好)')
    axes[0, 0].invert_yaxis()

    # 方向准确率对比
    direction_accs = [r["metrics"]["direction_accuracy"] * 100 for r in successful_results]

    axes[0, 1].barh(models, direction_accs, color='seagreen')
    axes[0, 1].set_xlabel('方向准确率 (%)')
    axes[0, 1].set_title('方向准确率 (越高越好)')
    axes[0, 1].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='随机猜测')
    axes[0, 1].legend()
    axes[0, 1].invert_yaxis()

    # 回测收益率对比
    if "backtest_metrics" in successful_results[0]:
        annual_returns = [r["backtest_metrics"].annualized_return * 100 for r in successful_results]

        axes[1, 0].barh(models, annual_returns, color='coral')
        axes[1, 0].set_xlabel('年化收益率 (%)')
        axes[1, 0].set_title('回测年化收益率')
        axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].invert_yaxis()

        # 夏普比率对比
        sharpe_ratios = [r["backtest_metrics"].sharpe_ratio for r in successful_results]

        axes[1, 1].barh(models, sharpe_ratios, color='mediumpurple')
        axes[1, 1].set_xlabel('夏普比率')
        axes[1, 1].set_title('夏普比率 (越高越好)')
        axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].axvline(x=1, color='red', linestyle='--', alpha=0.5, label='良好水平')
        axes[1, 1].legend()
        axes[1, 1].invert_yaxis()

    plt.tight_layout()
    viz_path = output_dir / "model_comparison.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✓ 保存对比图：{viz_path}")
    plt.close()


def generate_comparison_report(results: list[dict], output_dir: Path, args):
    """生成对比报告"""
    print(f"\n📝 生成对比报告...")

    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print("✗ 没有成功训练的模型，无法生成报告")
        return

    # 创建对比表格
    comparison_data = []

    for r in successful_results:
        row = {
            "模型": r["model_name"],
            "MAE": f"{r['metrics']['mae']:.4f}",
            "RMSE": f"{r['metrics']['rmse']:.4f}",
            "方向准确率": f"{r['metrics']['direction_accuracy']:.2%}",
        }

        if "backtest_metrics" in r:
            bm = r["backtest_metrics"]
            row.update({
                "年化收益率": f"{bm.annualized_return:.2%}",
                "夏普比率": f"{bm.sharpe_ratio:.2f}",
                "最大回撤": f"{bm.max_drawdown:.2%}",
                "胜率": f"{bm.win_rate:.2%}",
            })

        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)

    # 保存 CSV
    csv_path = output_dir / "model_comparison.csv"
    df_comparison.to_csv(csv_path, index=False)
    print(f"✓ 保存对比表：{csv_path}")

    # 生成 Markdown 报告
    md_content = f"""# 深度学习模型效果对比报告

**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验设置

- **数据源**：通达信 (TDX)
- **时间范围**：{args.start} - {args.end}
- **预测周期**：{args.horizon} 日
- **训练模型**：{', '.join([r['model_name'] for r in successful_results])}
- **序列长度**：{args.seq_length}
- **训练轮数**：{args.epochs}

## 模型对比

### 预测准确率

| 模型 | MAE | RMSE | 方向准确率 |
|------|-----|------|-----------|
"""

    for row in comparison_data:
        md_content += f"| {row['模型']} | {row['MAE']} | {row['RMSE']} | {row['方向准确率']} |\n"

    if "年化收益率" in comparison_data[0]:
        md_content += """
### 回测收益

| 模型 | 年化收益率 | 夏普比率 | 最大回撤 | 胜率 |
|------|-----------|---------|---------|------|
"""
        for row in comparison_data:
            md_content += f"| {row['模型']} | {row['年化收益率']} | {row['夏普比率']} | {row['最大回撤']} | {row['胜率']} |\n"

    # 找出最佳模型
    best_by_accuracy = max(successful_results, key=lambda x: x['metrics']['direction_accuracy'])
    md_content += f"""
## 结论

### 最佳模型（方向准确率）

**{best_by_accuracy['model_name'].upper()}**
- 方向准确率：{best_by_accuracy['metrics']['direction_accuracy']:.2%}
- MAE：{best_by_accuracy['metrics']['mae']:.4f}
"""

    if "backtest_metrics" in successful_results[0]:
        best_by_return = max(successful_results, key=lambda x: x['backtest_metrics'].annualized_return)
        md_content += f"""
### 最佳模型（年化收益率）

**{best_by_return['model_name'].upper()}**
- 年化收益率：{best_by_return['backtest_metrics'].annualized_return:.2%}
- 夏普比率：{best_by_return['backtest_metrics'].sharpe_ratio:.2f}
- 最大回撤：{best_by_return['backtest_metrics'].max_drawdown:.2%}
"""

    md_content += """
## 说明

- **MAE**：平均绝对误差，越小越好
- **RMSE**：均方根误差，越小越好
- **方向准确率**：预测涨跌方向的准确率，>50% 表示优于随机
- **年化收益率**：回测期间的年化投资收益
- **夏普比率**：风险调整后收益，>1 为良好，>2 为优秀
- **最大回撤**：资金曲线的最大回撤幅度
- **胜率**：盈利交易占比

---

*本报告仅用于量化研究和模型验证，不构成投资建议。*
"""

    md_path = output_dir / "comparison_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"✓ 保存报告：{md_path}")

    # 显示报告
    print(f"\n{md_content}")


def main(argv: list[str] | None = None):
    """主函数"""
    args = build_parser().parse_args(argv)

    print("\n" + "=" * 80)
    print("深度学习模型效果对比和收益率回测")
    print("=" * 80)

    # 1. 准备数据
    panel, feature_cols, target_col, universe = prepare_data(args)

    # 2. 数据划分
    print(f"\n{'='*80}")
    print(f"📊 数据划分")
    print(f"{'='*80}")

    split = time_series_split(
        panel,
        target_col=target_col,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    print(f"\n训练集：{len(split.train)} 行")
    print(f"  日期：{split.train_dates[0].date()} ~ {split.train_dates[1].date()}")
    print(f"验证集：{len(split.val)} 行")
    print(f"  日期：{split.val_dates[0].date()} ~ {split.val_dates[1].date()}")
    print(f"测试集：{len(split.test)} 行")
    print(f"  日期：{split.test_dates[0].date()} ~ {split.test_dates[1].date()}")

    # 3. 训练所有模型
    results = []

    for model_name in args.models:
        result = train_and_evaluate_model(
            model_name,
            split.train,
            split.val,
            split.test,
            feature_cols,
            target_col,
            args,
        )

        # 运行回测
        if result["success"]:
            print(f"\n💰 运行回测...")
            backtest_metrics = run_backtest(
                result["predictions"],
                pred_col=f"pred_{model_name}",
                target_col=target_col,
                args=args,
            )

            result["backtest_metrics"] = backtest_metrics

            print(f"\n回测结果：")
            print(f"  年化收益率：{backtest_metrics.annualized_return:.2%}")
            print(f"  夏普比率：{backtest_metrics.sharpe_ratio:.2f}")
            print(f"  最大回撤：{backtest_metrics.max_drawdown:.2%}")
            print(f"  胜率：{backtest_metrics.win_rate:.2%}")

        results.append(result)

    # 4. 生成对比报告
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_comparison_visualizations(results, output_dir)
    generate_comparison_report(results, output_dir, args)

    # 5. 保存完整结果
    results_path = output_dir / "full_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            [{
                "model_name": r["model_name"],
                "success": r["success"],
                "metrics": r.get("metrics", {}),
                "backtest_metrics": {
                    "annualized_return": r["backtest_metrics"].annualized_return,
                    "sharpe_ratio": r["backtest_metrics"].sharpe_ratio,
                    "max_drawdown": r["backtest_metrics"].max_drawdown,
                    "win_rate": r["backtest_metrics"].win_rate,
                } if "backtest_metrics" in r else {},
            } for r in results],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n{'='*80}")
    print(f"✅ 对比完成！")
    print(f"{'='*80}")
    print(f"\n输出目录：{output_dir}")
    print(f"  - 对比表格：model_comparison.csv")
    print(f"  - 对比图表：model_comparison.png")
    print(f"  - 对比报告：comparison_report.md")
    print(f"  - 完整结果：full_results.json")

    success_count = sum(1 for r in results if r["success"])
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
