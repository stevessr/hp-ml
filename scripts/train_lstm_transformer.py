#!/usr/bin/env python3
"""训练 LSTM-Transformer 混合模型

结合 LSTM 的序列编码能力和 Transformer 的自注意力机制
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
from hp_ml.visualization import (
    plot_model_comparison_metrics,
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


def train_lstm_transformer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_length: int = 20,
    lstm_units: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_transformer_blocks: int = 2,
    dropout: float = 0.3,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict[str, Any]:
    """训练 LSTM-Transformer 混合模型"""

    print(f"\n🚀 开始训练 LSTM-Transformer 混合模型...")
    print(f"   序列长度：{seq_length}")
    print(f"   LSTM 单元数：{lstm_units}")
    print(f"   注意力头数：{num_heads}")
    print(f"   前馈维度：{ff_dim}")
    print(f"   Transformer 块数：{num_transformer_blocks}")
    print(f"   Dropout 率：{dropout}")
    print(f"   训练轮数：{epochs}")
    print(f"   批次大小：{batch_size}\n")

    print(f"{'='*80}")
    print(f"🔧 训练 LSTM-Transformer 模型")
    print(f"{'='*80}")

    # 创建模型
    model = make_extended_model(
        "lstm_transformer",
        seq_length=seq_length,
        lstm_units=lstm_units,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        dropout=dropout,
        learning_rate=0.001,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=10,
    )

    # 准备数据
    X_train = train_df[feature_cols + ["code", "date"]]
    y_train = train_df[target_col]

    # 训练模型
    print(f"  训练中...")
    model.fit(X_train, y_train)
    print(f"  ✓ 训练完成")

    # 在训练集上评估
    print(f"  评估训练集...")
    train_predictions = model.predict(train_df[feature_cols + ["code", "date"]])
    train_actuals = train_df[target_col].values

    # 计算训练集指标
    train_mae = np.mean(np.abs(train_predictions - train_actuals))
    train_rmse = np.sqrt(np.mean((train_predictions - train_actuals) ** 2))
    train_directional_accuracy = np.mean(
        (train_predictions > 0) == (train_actuals > 0)
    )

    # 计算 Spearman IC
    from scipy.stats import spearmanr
    train_ic, _ = spearmanr(train_predictions, train_actuals)

    train_metrics = {
        "directional_accuracy": float(train_directional_accuracy),
        "spearman_ic_by_date": float(train_ic),
        "rmse": float(train_rmse),
        "mae": float(train_mae),
    }

    # 在测试集上评估
    print(f"  评估测试集...")
    test_predictions = model.predict(test_df[feature_cols + ["code", "date"]])
    test_actuals = test_df[target_col].values

    # 计算测试集指标
    test_mae = np.mean(np.abs(test_predictions - test_actuals))
    test_rmse = np.sqrt(np.mean((test_predictions - test_actuals) ** 2))
    test_directional_accuracy = np.mean(
        (test_predictions > 0) == (test_actuals > 0)
    )

    test_ic, _ = spearmanr(test_predictions, test_actuals)

    test_metrics = {
        "directional_accuracy": float(test_directional_accuracy),
        "spearman_ic_by_date": float(test_ic),
        "rmse": float(test_rmse),
        "mae": float(test_mae),
    }

    # 构建预测结果
    test_preds = []
    test_df_reset = test_df.reset_index(drop=True)
    for i in range(len(test_df_reset)):
        test_preds.append({
            "date": test_df_reset.loc[i, "date"],
            "code": test_df_reset.loc[i, "code"],
            "prediction": float(test_predictions[i]),
            target_col: float(test_actuals[i]),
        })

    print(f"\n  📊 训练集指标：")
    print(f"     方向准确率：{train_metrics['directional_accuracy']:.4f}")
    print(f"     Spearman IC: {train_metrics['spearman_ic_by_date']:.4f}")
    print(f"     RMSE: {train_metrics['rmse']:.6f}")
    print(f"     MAE: {train_metrics['mae']:.6f}")

    print(f"\n  📊 测试集指标：")
    print(f"     方向准确率：{test_metrics['directional_accuracy']:.4f}")
    print(f"     Spearman IC: {test_metrics['spearman_ic_by_date']:.4f}")
    print(f"     RMSE: {test_metrics['rmse']:.6f}")
    print(f"     MAE: {test_metrics['mae']:.6f}")
    print()

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_predictions": test_preds,
    }


def save_results(
    result: dict[str, Any],
    output_dir: Path,
) -> None:
    """保存训练结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 汇总指标
    train_metrics = result["train_metrics"]
    test_metrics = result["test_metrics"]

    summary = {
        "model": "lstm_transformer",
        "train_directional_accuracy": train_metrics["directional_accuracy"],
        "train_spearman_ic": train_metrics["spearman_ic_by_date"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "test_directional_accuracy": test_metrics["directional_accuracy"],
        "test_spearman_ic": test_metrics["spearman_ic_by_date"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "lstm_transformer_summary.csv", index=False)
    print(f"💾 汇总结果已保存：{output_dir / 'lstm_transformer_summary.csv'}")

    # 保存详细指标
    with open(output_dir / "lstm_transformer_details.json", "w", encoding="utf-8") as f:
        details = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"💾 详细指标已保存：{output_dir / 'lstm_transformer_details.json'}")

    # 保存预测结果
    predictions_df = pd.DataFrame(result["test_predictions"])
    predictions_df.to_csv(output_dir / "lstm_transformer_predictions.csv", index=False)
    print(f"💾 预测结果已保存：{output_dir / 'lstm_transformer_predictions.csv'}")


def create_visualizations(
    result: dict[str, Any],
    target_col: str,
    charts_dir: Path,
) -> None:
    """创建可视化图表"""
    charts_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 生成可视化图表...")

    # 预测分布图
    try:
        test_preds = result["test_predictions"]
        preds_df = pd.DataFrame(test_preds)

        if "prediction" in preds_df.columns and target_col in preds_df.columns:
            plot_prediction_distribution(
                preds_df,
                "prediction",
                target_col,
                charts_dir / "lstm_transformer_prediction_distribution.svg"
            )
            print(f"  ✓ 预测分布图：lstm_transformer_prediction_distribution.svg")
    except Exception as e:
        print(f"  ⚠ 预测分布图生成失败：{e}")

    print(f"\n📁 图表已保存到：{charts_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 LSTM-Transformer 混合模型")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "lstm_transformer")
    parser.add_argument("--charts", type=Path, default=REPORTS_DIR / "charts" / "lstm_transformer")
    parser.add_argument("--test-days", type=int, default=60, help="测试集天数")
    parser.add_argument("--seq-length", type=int, default=20, help="序列长度")
    parser.add_argument("--lstm-units", type=int, default=64, help="LSTM 单元数")
    parser.add_argument("--num-heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--ff-dim", type=int, default=128, help="前馈网络维度")
    parser.add_argument("--num-transformer-blocks", type=int, default=2, help="Transformer 块数")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 率")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🧠 LSTM-Transformer 混合模型训练")
    print("="*80)
    print(f"📁 训练面板：{args.panel}")
    print(f"🔢 测试集天数：{args.test_days}")
    print(f"📊 序列长度：{args.seq_length}")
    print(f"🔄 训练轮数：{args.epochs}")
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
        result = train_lstm_transformer(
            train_df, test_df, feature_cols, target_col,
            seq_length=args.seq_length,
            lstm_units=args.lstm_units,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            num_transformer_blocks=args.num_transformer_blocks,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # 保存结果
        save_results(result, args.output)

        # 创建可视化
        create_visualizations(result, target_col, args.charts)

        # 打印最终总结
        print("\n" + "="*80)
        print("📊 训练完成总结")
        print("="*80)
        test_metrics = result["test_metrics"]
        print(f"\n  LSTM-TRANSFORMER:")
        print(f"    测试集准确率：{test_metrics['directional_accuracy']:.4f}")
        print(f"    Spearman IC: {test_metrics['spearman_ic_by_date']:.4f}")
        print(f"    RMSE: {test_metrics['rmse']:.6f}")
        print(f"    MAE: {test_metrics['mae']:.6f}")

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
