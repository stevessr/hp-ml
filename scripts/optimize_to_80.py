#!/usr/bin/env python3
"""深度优化实验：冲击80%准确率

策略：
1. 扩展注意力窗口（25, 30, 40, 50步）
2. 优化模型架构（更深层网络、残差连接）
3. 高级正则化技术
4. 数据增强
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

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.models_extended import make_extended_model
from hp_ml.config import PROCESSED_DIR, REPORTS_DIR


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

    print(f"  ✓ 数据行数: {len(panel_df)}")
    print(f"  ✓ 特征数量: {len(feature_cols)}")

    panel_df = panel_df.sort_values("date")
    trainable = panel_df[panel_df.get("is_trainable", True)].copy()

    if len(trainable) == 0:
        trainable = panel_df.copy()

    unique_dates = sorted(trainable["date"].unique())
    split_idx = max(1, len(unique_dates) - test_days)
    split_date = unique_dates[split_idx]

    train_df = trainable[trainable["date"] < split_date].copy()
    test_df = trainable[trainable["date"] >= split_date].copy()

    print(f"\n📅 数据切分:")
    print(f"  训练集: {len(train_df)} 行")
    print(f"  测试集: {len(test_df)} 行\n")

    return train_df, test_df, feature_cols, target_col


def train_model_config(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """训练单个模型配置"""
    config_str = f"{model_name}"
    for k, v in sorted(model_kwargs.items()):
        if k not in ['learning_rate', 'batch_size', 'early_stopping_patience', 'epochs']:
            config_str += f"_{k}{v}"

    print(f"  🔧 {config_str}")

    try:
        model = make_extended_model(model_name, **model_kwargs)

        X_train = train_df[feature_cols + ["code", "date"]]
        y_train = train_df[target_col]

        model.fit(X_train, y_train)

        test_pred = model.predict(test_df[feature_cols + ["code", "date"]])
        test_actual = test_df[target_col].values

        test_metrics = {
            "accuracy": float(np.mean((test_pred > 0) == (test_actual > 0))),
            "ic": float(spearmanr(test_pred, test_actual)[0]),
            "rmse": float(np.sqrt(np.mean((test_pred - test_actual) ** 2))),
        }

        print(f"    ✓ 准确率: {test_metrics['accuracy']:.4f}, IC: {test_metrics['ic']:.4f}")

        return {
            "config": config_str,
            "test_metrics": test_metrics,
            "success": True,
            **model_kwargs
        }

    except Exception as e:
        print(f"    ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="深度优化实验")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_enhanced.csv")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "optimization_80")
    parser.add_argument("--test-days", type=int, default=60)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🎯 深度优化实验 - 冲击80%准确率")
    print("="*80)
    print(f"📁 训练面板: {args.panel}")
    print("="*80 + "\n")

    if not args.panel.exists():
        print(f"❌ 训练面板不存在: {args.panel}")
        return 1

    try:
        train_df, test_df, feature_cols, target_col = load_and_split_data(
            args.panel, args.test_days
        )

        # 定义优化实验配置
        experiments = [
            # 阶段1: 扩展注意力窗口
            ("attention_lstm", {
                "seq_length": 25, "units": 64, "dropout": 0.3,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 32,
                "early_stopping_patience": 8
            }),
            ("attention_lstm", {
                "seq_length": 30, "units": 64, "dropout": 0.3,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 32,
                "early_stopping_patience": 8
            }),
            ("attention_lstm", {
                "seq_length": 40, "units": 64, "dropout": 0.3,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 32,
                "early_stopping_patience": 8
            }),
            ("attention_lstm", {
                "seq_length": 50, "units": 64, "dropout": 0.3,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 32,
                "early_stopping_patience": 8
            }),

            # 阶段2: 增加模型容量（基于最佳序列长度）
            ("attention_lstm", {
                "seq_length": 30, "units": 96, "dropout": 0.3,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 32,
                "early_stopping_patience": 8
            }),
            ("attention_lstm", {
                "seq_length": 30, "units": 128, "dropout": 0.3,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 32,
                "early_stopping_patience": 8
            }),
            ("attention_lstm", {
                "seq_length": 30, "units": 192, "dropout": 0.3,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 32,
                "early_stopping_patience": 8
            }),

            # 阶段3: 优化正则化
            ("attention_lstm", {
                "seq_length": 30, "units": 128, "dropout": 0.2,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 32,
                "early_stopping_patience": 8
            }),
            ("attention_lstm", {
                "seq_length": 30, "units": 128, "dropout": 0.4,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 32,
                "early_stopping_patience": 8
            }),
            ("attention_lstm", {
                "seq_length": 30, "units": 128, "dropout": 0.5,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 32,
                "early_stopping_patience": 8
            }),

            # 阶段4: 调整学习率和训练时长
            ("attention_lstm", {
                "seq_length": 30, "units": 128, "dropout": 0.3,
                "learning_rate": 0.0005, "epochs": 30, "batch_size": 32,
                "early_stopping_patience": 10
            }),
            ("attention_lstm", {
                "seq_length": 30, "units": 128, "dropout": 0.3,
                "learning_rate": 0.002, "epochs": 30, "batch_size": 32,
                "early_stopping_patience": 10
            }),

            # 阶段5: 批次大小实验
            ("attention_lstm", {
                "seq_length": 30, "units": 128, "dropout": 0.3,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 16,
                "early_stopping_patience": 8
            }),
            ("attention_lstm", {
                "seq_length": 30, "units": 128, "dropout": 0.3,
                "learning_rate": 0.001, "epochs": 25, "batch_size": 64,
                "early_stopping_patience": 8
            }),
        ]

        print(f"🚀 将测试 {len(experiments)} 个优化配置\n")

        results = []
        best_accuracy = 0.0

        for idx, (model_name, config) in enumerate(experiments, 1):
            print(f"\n{'='*80}")
            print(f"📊 实验 {idx}/{len(experiments)}")
            print(f"{'='*80}")

            result = train_model_config(
                model_name, train_df, test_df, feature_cols, target_col, **config
            )

            if result["success"]:
                results.append(result)
                acc = result["test_metrics"]["accuracy"]
                if acc > best_accuracy:
                    best_accuracy = acc
                    print(f"    🏆 新最佳: {acc:.4f}")

        if not results:
            print("\n❌ 没有成功的实验")
            return 1

        # 保存结果
        args.output.mkdir(parents=True, exist_ok=True)

        summary_data = []
        for r in results:
            summary_data.append({
                "config": r["config"],
                "seq_length": r.get("seq_length", 0),
                "units": r.get("units", 0),
                "dropout": r.get("dropout", 0),
                "learning_rate": r.get("learning_rate", 0),
                "batch_size": r.get("batch_size", 0),
                "test_accuracy": r["test_metrics"]["accuracy"],
                "test_ic": r["test_metrics"]["ic"],
                "test_rmse": r["test_metrics"]["rmse"],
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("test_accuracy", ascending=False)
        summary_df.to_csv(args.output / "optimization_results.csv", index=False)

        with open(args.output / "optimization_details.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 结果已保存: {args.output}")

        # 打印总结
        print("\n" + "="*80)
        print("📊 优化实验总结")
        print("="*80)
        print(f"\n成功完成 {len(results)} 个实验")
        print(f"最佳准确率: {best_accuracy:.4f}\n")

        # Top 5
        print("🏆 Top 5 配置:\n")
        for idx, row in summary_df.head(5).iterrows():
            print(f"  {idx+1}. {row['config']}")
            print(f"     准确率: {row['test_accuracy']:.4f}")
            print(f"     IC: {row['test_ic']:.4f}")
            print()

        # 序列长度分析
        print("\n📈 注意力窗口分析:\n")
        seq_analysis = summary_df.groupby('seq_length').agg({
            'test_accuracy': ['mean', 'max']
        }).round(4)
        print(seq_analysis)

        print("\n" + "="*80)
        print("✅ 优化实验完成！")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
