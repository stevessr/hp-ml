#!/usr/bin/env python3
"""批量训练所有 Memory Transformer 变种模型

训练并对比：
1. LSTM-Transformer（已有）
2. GRU-Transformer
3. Transformer-XL
4. Memory-Augmented Transformer
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
from hp_ml.config import PROCESSED_DIR, REPORTS_DIR
from scipy.stats import spearmanr

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_training_data(panel_path: Path) -> tuple[pd.DataFrame, list[str], str]:
    """加载训练数据"""
    print("📊 加载训练面板数据...")
    panel_df = pd.read_csv(panel_path)

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
    trainable = panel_df[panel_df.get("is_trainable", True)].copy()

    if len(trainable) == 0:
        trainable = panel_df.copy()

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


def train_single_model(
    model_type: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_config: dict[str, Any]
) -> dict[str, Any]:
    """训练单个模型"""

    print(f"\n{'='*80}")
    print(f"🔧 训练模型：{model_type.upper()}")
    print(f"{'='*80}")

    try:
        # 创建模型
        model = make_extended_model(model_type, **model_config)

        # 准备数据
        X_train = train_df[feature_cols + ["code", "date"]]
        y_train = train_df[target_col]

        # 训练
        print(f"  训练中...")
        model.fit(X_train, y_train)
        print(f"  ✓ 训练完成")

        # 评估训练集
        print(f"  评估训练集...")
        train_predictions = model.predict(train_df[feature_cols + ["code", "date"]])
        train_actuals = train_df[target_col].values

        train_mae = np.mean(np.abs(train_predictions - train_actuals))
        train_rmse = np.sqrt(np.mean((train_predictions - train_actuals) ** 2))
        train_directional_accuracy = np.mean((train_predictions > 0) == (train_actuals > 0))
        train_ic, _ = spearmanr(train_predictions, train_actuals)

        train_metrics = {
            "directional_accuracy": float(train_directional_accuracy),
            "spearman_ic_by_date": float(train_ic),
            "rmse": float(train_rmse),
            "mae": float(train_mae),
        }

        # 评估测试集
        print(f"  评估测试集...")
        test_predictions = model.predict(test_df[feature_cols + ["code", "date"]])
        test_actuals = test_df[target_col].values

        test_mae = np.mean(np.abs(test_predictions - test_actuals))
        test_rmse = np.sqrt(np.mean((test_predictions - test_actuals) ** 2))
        test_directional_accuracy = np.mean((test_predictions > 0) == (test_actuals > 0))
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

        print(f"\n  📊 测试集指标：")
        print(f"     方向准确率：{test_metrics['directional_accuracy']:.4f}")
        print(f"     Spearman IC: {test_metrics['spearman_ic_by_date']:.4f}")
        print(f"     RMSE: {test_metrics['rmse']:.6f}")

        return {
            "model": model,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_predictions": test_preds,
            "success": True,
            "error": None
        }

    except Exception as e:
        print(f"  ❌ 训练失败：{e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "train_metrics": None,
            "test_metrics": None,
        }


def save_results(
    results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """保存所有结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 汇总指标
    summary = []
    for model_name, result in results.items():
        if not result.get("success", False):
            continue

        train_metrics = result["train_metrics"]
        test_metrics = result["test_metrics"]

        summary.append({
            "model": model_name,
            "train_directional_accuracy": train_metrics["directional_accuracy"],
            "train_spearman_ic": train_metrics["spearman_ic_by_date"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "test_directional_accuracy": test_metrics["directional_accuracy"],
            "test_spearman_ic": test_metrics["spearman_ic_by_date"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "memory_transformer_variants_summary.csv", index=False)
    print(f"\n💾 汇总结果已保存：{output_dir / 'memory_transformer_variants_summary.csv'}")

    # 保存详细指标
    with open(output_dir / "memory_transformer_variants_details.json", "w", encoding="utf-8") as f:
        details = {}
        for model_name, result in results.items():
            if result.get("success", False):
                details[model_name] = {
                    "train_metrics": result["train_metrics"],
                    "test_metrics": result["test_metrics"],
                }
            else:
                details[model_name] = {
                    "success": False,
                    "error": result.get("error")
                }
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"💾 详细指标已保存：{output_dir / 'memory_transformer_variants_details.json'}")

    # 保存每个模型的预测
    for model_name, result in results.items():
        if result.get("success", False) and "test_predictions" in result:
            pred_df = pd.DataFrame(result["test_predictions"])
            pred_df.to_csv(output_dir / f"{model_name}_predictions.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="批量训练 Memory Transformer 变种")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "memory_transformer_variants")
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🧠 Memory Transformer 变种批量训练")
    print("="*80)
    print(f"📁 训练面板：{args.panel}")
    print(f"🔢 测试集天数：{args.test_days}")
    print(f"📊 序列长度：{args.seq_length}")
    print(f"🔄 训练轮数：{args.epochs}")
    print("="*80 + "\n")

    if not args.panel.exists():
        print(f"❌ 训练面板文件不存在：{args.panel}")
        return 1

    try:
        # 加载数据
        panel_df, feature_cols, target_col = load_training_data(args.panel)
        train_df, test_df, split_date = split_train_test(panel_df, args.test_days)

        # 定义模型配置
        model_configs = {
            "lstm_transformer": {
                "seq_length": args.seq_length,
                "lstm_units": 64,
                "num_heads": 4,
                "ff_dim": 128,
                "num_transformer_blocks": 2,
                "dropout": 0.3,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
            "gru_transformer": {
                "seq_length": args.seq_length,
                "gru_units": 64,
                "num_heads": 4,
                "ff_dim": 128,
                "num_transformer_blocks": 2,
                "dropout": 0.3,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
            "transformer_xl": {
                "seq_length": args.seq_length,
                "d_model": 64,
                "num_heads": 4,
                "ff_dim": 128,
                "num_layers": 2,
                "memory_length": 10,
                "dropout": 0.3,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
            "memory_transformer": {
                "seq_length": args.seq_length,
                "d_model": 64,
                "num_heads": 4,
                "ff_dim": 128,
                "num_layers": 2,
                "memory_size": 32,
                "dropout": 0.3,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
        }

        # 训练所有模型
        results = {}
        for model_type, config in model_configs.items():
            result = train_single_model(
                model_type, train_df, test_df,
                feature_cols, target_col, config
            )
            results[model_type] = result

        # 统计成功的模型
        successful = [name for name, r in results.items() if r.get("success", False)]
        failed = [name for name, r in results.items() if not r.get("success", False)]

        if not successful:
            print("\n❌ 没有成功训练任何模型")
            return 1

        # 保存结果
        save_results(results, args.output)

        # 打印总结
        print("\n" + "="*80)
        print("📊 训练完成总结")
        print("="*80)
        print(f"\n✅ 成功训练 {len(successful)} 个模型")
        if failed:
            print(f"❌ 失败 {len(failed)} 个模型：{', '.join(failed)}")

        print("\n模型性能排序（按测试集准确率）:\n")

        summary_data = []
        for model_name in successful:
            result = results[model_name]
            test_metrics = result["test_metrics"]
            summary_data.append({
                "model": model_name,
                "accuracy": test_metrics["directional_accuracy"],
                "ic": test_metrics["spearman_ic_by_date"],
                "rmse": test_metrics["rmse"]
            })

        summary_data.sort(key=lambda x: x["accuracy"], reverse=True)

        for i, item in enumerate(summary_data, 1):
            emoji = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{emoji} {i}. {item['model'].upper():25s} - 准确率：{item['accuracy']:.4f}, "
                  f"IC: {item['ic']:.4f}, RMSE: {item['rmse']:.6f}")

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
