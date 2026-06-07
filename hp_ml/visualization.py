"""可视化模块：生成对比图表和报告"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_model_comparison_metrics(
    backtest_df: pd.DataFrame,
    output_path: Path | str,
) -> None:
    """绘制模型对比指标图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("多模型回测指标对比", fontsize=16, fontweight="bold")

    metrics = [
        ("annualized_return", "年化收益率", "{:.2%}"),
        ("sharpe_ratio", "夏普比率", "{:.3f}"),
        ("max_drawdown", "最大回撤", "{:.2%}"),
        ("win_rate", "胜率", "{:.2%}"),
        ("sortino_ratio", "索提诺比率", "{:.3f}"),
        ("calmar_ratio", "卡玛比率", "{:.3f}"),
    ]

    for idx, (metric, title, fmt) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        values = backtest_df[metric].values
        models = backtest_df["model"].values

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        bars = ax.barh(models, values, color=colors)

        ax.set_xlabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        for bar, val in zip(bars, values):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f" {fmt.format(val)}",
                    va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"指标对比图已保存: {output_path}")


def plot_equity_curves(
    model_predictions: dict[str, pd.DataFrame],
    target_col: str,
    output_path: Path | str,
    top_k: int = 3,
    transaction_cost: float = 0.001,
) -> None:
    """绘制多模型权益曲线对比"""
    from .backtest import backtest_strategy

    fig, ax = plt.subplots(figsize=(14, 7))

    for model_name, pred_df in model_predictions.items():
        pred_col = "prediction" if "prediction" in pred_df.columns else "pred_fwd_ret_5"

        try:
            _, returns_df = backtest_strategy(
                pred_df, pred_col, target_col,
                top_k=top_k,
                transaction_cost=transaction_cost
            )

            ax.plot(returns_df["date"], returns_df["equity"], label=model_name, linewidth=2)

        except Exception as e:
            print(f"模型 {model_name} 权益曲线绘制失败: {e}")
            continue

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="初始资金")
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("权益倍数", fontsize=12)
    ax.set_title("多模型权益曲线对比", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"权益曲线图已保存: {output_path}")


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    output_path: Path | str,
    top_n: int = 20,
) -> None:
    """绘制特征重要性图"""
    try:
        if hasattr(model, "get_feature_importance"):
            importance_df = model.get_feature_importance()
        elif hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
        elif hasattr(model, "named_steps") and hasattr(model.named_steps["model"], "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.named_steps["model"].feature_importances_
            }).sort_values("importance", ascending=False)
        else:
            print("模型不支持特征重要性分析")
            return

        top_features = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(top_features)))

        bars = ax.barh(top_features["feature"], top_features["importance"], color=colors)
        ax.set_xlabel("重要性", fontsize=12)
        ax.set_title(f"特征重要性 TOP {top_n}", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"特征重要性图已保存: {output_path}")

    except Exception as e:
        print(f"特征重要性图绘制失败: {e}")


def plot_prediction_distribution(
    predictions_df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    output_path: Path | str,
) -> None:
    """绘制预测值 vs 实际值分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 散点图
    ax1 = axes[0]
    ax1.scatter(predictions_df[pred_col], predictions_df[target_col], alpha=0.3, s=10)

    min_val = min(predictions_df[pred_col].min(), predictions_df[target_col].min())
    max_val = max(predictions_df[pred_col].max(), predictions_df[target_col].max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="理想预测")

    ax1.set_xlabel("预测值", fontsize=12)
    ax1.set_ylabel("实际值", fontsize=12)
    ax1.set_title("预测值 vs 实际值", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 分布直方图
    ax2 = axes[1]
    ax2.hist(predictions_df[pred_col], bins=50, alpha=0.6, label="预测值分布", color="blue")
    ax2.hist(predictions_df[target_col], bins=50, alpha=0.6, label="实际值分布", color="orange")
    ax2.set_xlabel("收益率", fontsize=12)
    ax2.set_ylabel("频数", fontsize=12)
    ax2.set_title("预测值与实际值分布对比", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"预测分布图已保存: {output_path}")


def generate_markdown_report(
    backtest_df: pd.DataFrame,
    models_metrics: dict[str, dict],
    output_path: Path | str,
) -> None:
    """生成 Markdown 格式的多模型对比报告"""
    lines = [
        "# 多模型训练与回测报告",
        "",
        f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. 模型训练评估指标",
        "",
        "| 模型 | MAE | RMSE | R² | 方向准确率 | Spearman IC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for model_name, metrics in models_metrics.items():
        mae = metrics.get("mae", 0)
        rmse = metrics.get("rmse", 0)
        r2 = metrics.get("r2", 0)
        dir_acc = metrics.get("directional_accuracy", 0)
        ic = metrics.get("spearman_ic_by_date", 0) or 0

        lines.append(
            f"| {model_name} | {mae:.6f} | {rmse:.6f} | {r2:.4f} | {dir_acc:.2%} | {ic:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. 回测性能指标",
        "",
        "| 模型 | 总收益率 | 年化收益率 | 夏普比率 | 索提诺比率 | 最大回撤 | 卡玛比率 | 胜率 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for _, row in backtest_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['total_return']:.2%} | {row['annualized_return']:.2%} | "
            f"{row['sharpe_ratio']:.3f} | {row['sortino_ratio']:.3f} | {row['max_drawdown']:.2%} | "
            f"{row['calmar_ratio']:.3f} | {row['win_rate']:.2%} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. 模型排名",
        "",
        f"**最佳模型（按夏普比率）**: {backtest_df.iloc[0]['model']}",
        "",
        "### 各指标排名",
        "",
    ])

    ranking_metrics = [
        ("annualized_return", "年化收益率", False),
        ("sharpe_ratio", "夏普比率", False),
        ("max_drawdown", "最大回撤", True),
        ("win_rate", "胜率", False),
    ]

    for metric, title, ascending in ranking_metrics:
        sorted_df = backtest_df.sort_values(metric, ascending=ascending)
        top3 = sorted_df.head(3)["model"].tolist()
        lines.append(f"**{title}**: {', '.join(top3)}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. 图表说明",
        "",
        "- `model_comparison_metrics.svg`: 各模型回测指标对比柱状图",
        "- `equity_curves.svg`: 多模型权益曲线对比",
        "- `feature_importance_*.svg`: 各模型特征重要性分析",
        "- `prediction_distribution_*.svg`: 预测值与实际值分布对比",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Markdown 报告已保存: {output_path}")


def create_all_visualizations(
    backtest_df: pd.DataFrame,
    models_results: dict[str, dict],
    all_predictions: dict[str, pd.DataFrame],
    target_col: str,
    charts_dir: Path,
    top_k: int = 3,
    transaction_cost: float = 0.001,
) -> None:
    """生成所有可视化图表"""
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1. 模型指标对比
    plot_model_comparison_metrics(backtest_df, charts_dir / "model_comparison_metrics.svg")

    # 2. 权益曲线
    plot_equity_curves(all_predictions, target_col, charts_dir / "equity_curves.svg", top_k, transaction_cost)

    # 3. 特征重要性（支持的模型）
    for model_name, result in models_results.items():
        model = result["model"]
        feature_cols = result.get("feature_cols", [])

        if feature_cols:
            plot_feature_importance(
                model,
                feature_cols,
                charts_dir / f"feature_importance_{model_name}.svg"
            )

    # 4. 预测分布
    for model_name, predictions in all_predictions.items():
        pred_col = "prediction"
        plot_prediction_distribution(
            predictions,
            pred_col,
            target_col,
            charts_dir / f"prediction_distribution_{model_name}.svg"
        )
