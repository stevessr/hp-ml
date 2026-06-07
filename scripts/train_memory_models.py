#!/usr/bin/env python3
"""训练记忆门模型（LSTM、GRU、BiLSTM、Attention LSTM）

使用时间序列数据训练多个基于记忆门机制的深度学习模型
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from hp_ml.models_extended import make_extended_model
from hp_ml.config import PROCESSED_DIR, REPORTS_DIR, MODELS_DIR
from hp_ml.lite_train import evaluate
from hp_ml.visualization import (
    plot_model_comparison_metrics,
    plot_equity_curves,
    plot_prediction_distribution,
)

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))




def load_training_data(panel_path: Path) -> tuple[pd.DataFrame, list[str], str]:
    """加载训练数据"""
    print("📊 加载训练面板数据...")
    panel_df = pd.read_csv(panel_path)

    # 确定特征列和目标列
    target_col = "fwd_ret_5"
    exclude_cols = {'date', 'code', 'name', 'family_id', 'close', target_col, 'is_trainable', 'is_future'}
    feature_cols = [c for c in panel_df.columns if c not in exclude_cols and not c.startswith('fwd_')]

    print(f"  ✓ 数据行数：{len(panel_df)}")
    print(f"  ✓ 日期范围：{panel_df['date'].min()} ~ {panel_df['date'].max()}")
    print(f"  ✓ 特征数量：{len(feature_cols)}")

    return panel_df, feature_cols, target_col


def split_train_test(
    panel_df: pd.DataFrame,
    test_days: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """按时间切分训练集和测试集"""
    panel_df = panel_df.sort_values("date")

    # 找到可训练的数据
    trainable = panel_df[panel_df.get("is_trainable", True)].copy()

    if len(trainable) == 0:
        trainable = panel_df.copy()

    # 按日期切分
    unique_dates = sorted(trainable["date"].unique())
    split_idx = max(1, len(unique_dates) - test_days)
    split_date = unique_dates[split_idx]

    train_df = trainable[trainable["date"] < split_date].copy()
    test_df = trainable[trainable["date"] >= split_date].copy()

    print(f"\n📅 数据切分：")
    print(f"  训练集：{train_df['date'].min()} ~ {train_df['date'].max()} ({len(train_df)} 行)")
    print(f"  测试集：{test_df['date'].min()} ~ {test_df['date'].max()} ({len(test_df)} 行)")
    print(f"  切分日期：{split_date}")

    return train_df, test_df, split_date


def train_memory_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_types: list[str] | None = None,
    seq_length: int = 20,
    epochs: int = 30,
) -> dict[str, dict[str, Any]]:
    """训练所有记忆门模型"""

    if model_types is None:
        model_types = ["lstm", "gru", "bilstm", "attention_lstm"]

    print(f"\n🚀 开始训练 {len(model_types)} 个记忆门模型...")
    print(f"   序列长度：{seq_length}")
    print(f"   训练轮数：{epochs}\n")

    results = {}

    for model_type in model_types:
        print(f"{'='*80}")
        print(f"🔧 训练模型：{model_type.upper()}")
        print(f"{'='*80}")

        try:
            # 创建模型
            model = make_extended_model(
                model_type,
                seq_length=seq_length,
                units=64,
                dropout=0.3,
                learning_rate=0.001,
                epochs=epochs,
                batch_size=32,
                early_stopping_patience=5,
            )

            # 准备数据
            X_train = train_df[feature_cols + ["code", "date"]]
            y_train = train_df[target_col]

            # 训练模型
            print(f"  训练中...")
            model.fit(X_train, y_train)
            print(f"  ✓ 训练完成")

            # 在训练集上评估
            train_data_list = train_df.to_dict('records')
            train_metrics, _ = evaluate(
                train_data_list,
                {"predict": lambda X: model.predict(pd.DataFrame(X))},
                target_col
            )

            # 在测试集上评估
            test_data_list = test_df.to_dict('records')
            test_metrics, test_preds = evaluate(
                test_data_list,
                {"predict": lambda X: model.predict(pd.DataFrame(X))},
                target_col
            )

            # 保存结果
            results[model_type] = {
                "model": model,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "test_predictions": test_preds,
            }

            print(f"\n  📊 训练集指标：")
            print(f"     方向准确率：{train_metrics.get('directional_accuracy', 0):.4f}")
            print(f"     Spearman IC: {train_metrics.get('spearman_ic_by_date', 0):.4f}")
            print(f"     RMSE: {train_metrics.get('rmse', 0):.6f}")

            print(f"\n  📊 测试集指标：")
            print(f"     方向准确率：{test_metrics.get('directional_accuracy', 0):.4f}")
            print(f"     Spearman IC: {test_metrics.get('spearman_ic_by_date', 0):.4f}")
            print(f"     RMSE: {test_metrics.get('rmse', 0):.6f}")
            print()

        except Exception as e:
            print(f"  ❌ 训练失败：{e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def save_results(
    results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """保存训练结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 汇总指标
    summary = []
    for model_name, result in results.items():
        train_metrics = result["train_metrics"]
        test_metrics = result["test_metrics"]

        summary.append({
            "model": model_name,
            "train_directional_accuracy": train_metrics.get("directional_accuracy", 0),
            "train_spearman_ic": train_metrics.get("spearman_ic_by_date", 0),
            "train_rmse": train_metrics.get("rmse", 0),
            "test_directional_accuracy": test_metrics.get("directional_accuracy", 0),
            "test_spearman_ic": test_metrics.get("spearman_ic_by_date", 0),
            "test_rmse": test_metrics.get("rmse", 0),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "memory_models_summary.csv", index=False)
    print(f"💾 汇总结果已保存：{output_dir / 'memory_models_summary.csv'}")

    # 保存详细指标
    with open(output_dir / "memory_models_details.json", "w", encoding="utf-8") as f:
        details = {
            model_name: {
                "train_metrics": result["train_metrics"],
                "test_metrics": result["test_metrics"],
            }
            for model_name, result in results.items()
        }
        json.dump(details, f, indent=2, ensure_ascii=False, default=str)
    print(f"💾 详细指标已保存：{output_dir / 'memory_models_details.json'}")


def create_visualizations(
    results: dict[str, dict[str, Any]],
    target_col: str,
    charts_dir: Path,
) -> None:
    """创建可视化图表"""
    charts_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 生成可视化图表...")

    # 准备数据
    summary_data = []
    for model_name, result in results.items():
        test_metrics = result["test_metrics"]
        summary_data.append({
            "model": model_name,
            "directional_accuracy": test_metrics.get("directional_accuracy", 0),
            "spearman_ic_by_date": test_metrics.get("spearman_ic_by_date", 0),
            "rmse": test_metrics.get("rmse", 0),
            "mae": test_metrics.get("mae", 0),
        })

    summary_df = pd.DataFrame(summary_data)

    # 1. 模型指标对比
    try:
        plot_model_comparison_metrics(
            summary_df,
            charts_dir / "memory_models_comparison.svg"
        )
        print(f"  ✓ 模型对比图：memory_models_comparison.svg")
    except Exception as e:
        print(f"  ⚠ 模型对比图生成失败：{e}")

    # 2. 预测分布图
    for model_name, result in results.items():
        try:
            test_preds = result["test_predictions"]
            preds_df = pd.DataFrame(test_preds)

            if "prediction" in preds_df.columns and target_col in preds_df.columns:
                plot_prediction_distribution(
                    preds_df,
                    "prediction",
                    target_col,
                    charts_dir / f"prediction_distribution_{model_name}.svg"
                )
                print(f"  ✓ 预测分布图：prediction_distribution_{model_name}.svg")
        except Exception as e:
            print(f"  ⚠ {model_name} 预测分布图生成失败：{e}")

    print(f"\n📁 图表已保存到：{charts_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="训练记忆门模型")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "memory_models")
    parser.add_argument("--charts", type=Path, default=REPORTS_DIR / "charts" / "memory_models")
    parser.add_argument("--test-days", type=int, default=60, help="测试集天数")
    parser.add_argument("--seq-length", type=int, default=20, help="序列长度")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "gru", "bilstm", "attention_lstm"],
        help="要训练的模型类型"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🧠 记忆门模型训练")
    print("="*80)
    print(f"📁 训练面板：{args.panel}")
    print(f"🔢 测试集天数：{args.test_days}")
    print(f"📊 序列长度：{args.seq_length}")
    print(f"🔄 训练轮数：{args.epochs}")
    print(f"🤖 模型列表：{', '.join(args.models)}")
    print("="*80 + "\n")

    # 检查数据文件
    if not args.panel.exists():
        print(f"❌ 训练面板文件不存在：{args.panel}")
        print("   请先运行 python hp_ml/lite_train.py 生成训练数据")
        return 1

    try:
        # 加载数据
        panel_df, feature_cols, target_col = load_training_data(args.panel)

        # 切分数据
        train_df, test_df, split_date = split_train_test(panel_df, args.test_days)

        # 训练模型
        results = train_memory_models(
            train_df, test_df, feature_cols, target_col,
            model_types=args.models,
            seq_length=args.seq_length,
            epochs=args.epochs,
        )

        if not results:
            print("\n❌ 没有成功训练任何模型")
            return 1

        # 保存结果
        save_results(results, args.output)

        # 创建可视化
        create_visualizations(results, target_col, args.charts)

        # 打印最终总结
        print("\n" + "="*80)
        print("📊 训练完成总结")
        print("="*80)
        print(f"\n成功训练 {len(results)} 个模型:\n")

        for model_name, result in results.items():
            test_metrics = result["test_metrics"]
            print(f"  {model_name.upper()}:")
            print(f"    测试集准确率：{test_metrics.get('directional_accuracy', 0):.4f}")
            print(f"    Spearman IC: {test_metrics.get('spearman_ic_by_date', 0):.4f}")
            print(f"    RMSE: {test_metrics.get('rmse', 0):.6f}")
            print()

        # 找出最佳模型
        best_model = max(
            results.items(),
            key=lambda x: x[1]["test_metrics"].get("directional_accuracy", 0)
        )

        print(f"🏆 最佳模型（按准确率）: {best_model[0].upper()}")
        print(f"   准确率：{best_model[1]['test_metrics'].get('directional_accuracy', 0):.4f}")

        print("\n" + "="*80)
        print("✅ 所有任务完成！")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
