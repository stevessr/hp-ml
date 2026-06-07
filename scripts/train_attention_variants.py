#!/usr/bin/env python3
"""训练注意力机制变种模型

专门训练各种注意力架构的变种模型，比较它们的性能
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from hp_ml.models_extended import make_extended_model
from hp_ml.config import PROCESSED_DIR, REPORTS_DIR

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def load_and_split_data(
    panel_path: Path,
    test_days: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    """加载并切分数据"""
    print("📊 加载训练面板数据...")
    panel_df = pd.read_csv(panel_path)

    target_col = "fwd_ret_5"
    exclude_cols = {'date', 'code', 'name', 'family_id', 'close', target_col, 'is_trainable', 'is_future'}
    feature_cols = [c for c in panel_df.columns if c not in exclude_cols and not c.startswith('fwd_')]

    print(f"  ✓ 数据行数：{len(panel_df)}")
    print(f"  ✓ 特征数量：{len(feature_cols)}")

    # 切分数据
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
    print(f"  训练集：{len(train_df)} 行")
    print(f"  测试集：{len(test_df)} 行")

    return train_df, test_df, feature_cols, target_col


def train_single_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_length: int = 10,
    epochs: int = 20,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """训练单个模型"""
    print(f"\n{'='*80}")
    print(f"🔧 训练模型：{model_name.upper()}")
    print(f"{'='*80}")

    try:
        # 创建模型
        model = make_extended_model(
            model_name,
            seq_length=seq_length,
            units=64,
            dropout=0.3,
            learning_rate=0.001,
            epochs=epochs,
            batch_size=32,
            early_stopping_patience=5,
            **model_kwargs
        )

        # 准备数据
        X_train = train_df[feature_cols + ["code", "date"]]
        y_train = train_df[target_col]

        # 训练
        print("  训练中...")
        model.fit(X_train, y_train)
        print("  ✓ 训练完成")

        # 评估
        print("  评估中...")
        train_pred = model.predict(train_df[feature_cols + ["code", "date"]])
        test_pred = model.predict(test_df[feature_cols + ["code", "date"]])

        train_actual = train_df[target_col].values
        test_actual = test_df[target_col].values

        # 计算指标
        train_metrics = {
            "directional_accuracy": np.mean((train_pred > 0) == (train_actual > 0)),
            "spearman_ic": spearmanr(train_pred, train_actual)[0],
            "rmse": np.sqrt(np.mean((train_pred - train_actual) ** 2)),
            "mae": np.mean(np.abs(train_pred - train_actual)),
        }

        test_metrics = {
            "directional_accuracy": np.mean((test_pred > 0) == (test_actual > 0)),
            "spearman_ic": spearmanr(test_pred, test_actual)[0],
            "rmse": np.sqrt(np.mean((test_pred - test_actual) ** 2)),
            "mae": np.mean(np.abs(test_pred - test_actual)),
        }

        print(f"\n  📊 测试集指标：")
        print(f"     准确率：{test_metrics['directional_accuracy']:.4f}")
        print(f"     IC: {test_metrics['spearman_ic']:.4f}")
        print(f"     RMSE: {test_metrics['rmse']:.6f}")

        return {
            "model": model,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "success": True,
        }

    except Exception as e:
        print(f"  ❌ 训练失败：{e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="训练注意力机制变种模型")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "attention_models")
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--seq-length", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🎯 注意力机制变种模型训练")
    print("="*80)
    print(f"📁 训练面板：{args.panel}")
    print(f"📊 序列长度：{args.seq_length}")
    print(f"🔄 训练轮数：{args.epochs}")
    print("="*80 + "\n")

    if not args.panel.exists():
        print(f"❌ 训练面板不存在：{args.panel}")
        return 1

    try:
        # 加载数据
        train_df, test_df, feature_cols, target_col = load_and_split_data(
            args.panel, args.test_days
        )

        # 定义要训练的模型
        models_to_train = [
            ("attention_lstm", {}),
            ("multihead_attention", {"num_heads": 4}),
            ("self_attention", {"num_heads": 4, "ff_dim": 128}),
            ("hierarchical_attention", {}),
        ]

        print(f"\n🚀 将训练 {len(models_to_train)} 个注意力变种模型\n")

        results = {}
        for model_name, model_kwargs in models_to_train:
            result = train_single_model(
                model_name,
                train_df,
                test_df,
                feature_cols,
                target_col,
                seq_length=args.seq_length,
                epochs=args.epochs,
                **model_kwargs
            )

            if result["success"]:
                results[model_name] = result

        if not results:
            print("\n❌ 没有成功训练任何模型")
            return 1

        # 保存结果
        args.output.mkdir(parents=True, exist_ok=True)

        summary = []
        for model_name, result in results.items():
            train_m = result["train_metrics"]
            test_m = result["test_metrics"]

            summary.append({
                "model": model_name,
                "train_accuracy": train_m["directional_accuracy"],
                "train_ic": train_m["spearman_ic"],
                "train_rmse": train_m["rmse"],
                "test_accuracy": test_m["directional_accuracy"],
                "test_ic": test_m["spearman_ic"],
                "test_rmse": test_m["rmse"],
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(args.output / "attention_models_summary.csv", index=False)
        print(f"\n💾 结果已保存：{args.output / 'attention_models_summary.csv'}")

        # 打印总结
        print("\n" + "="*80)
        print("📊 训练完成总结")
        print("="*80)
        print(f"\n成功训练 {len(results)} 个模型:\n")

        for model_name, result in results.items():
            test_m = result["test_metrics"]
            print(f"  {model_name.upper()}:")
            print(f"    准确率：{test_m['directional_accuracy']:.4f}")
            print(f"    IC: {test_m['spearman_ic']:.4f}")
            print(f"    RMSE: {test_m['rmse']:.6f}")
            print()

        # 找出最佳模型
        best_model = max(
            results.items(),
            key=lambda x: x[1]["test_metrics"]["directional_accuracy"]
        )

        print(f"🏆 最佳模型：{best_model[0].upper()}")
        print(f"   准确率：{best_model[1]['test_metrics']['directional_accuracy']:.4f}")

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
