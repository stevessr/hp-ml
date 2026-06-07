#!/usr/bin/env python3
"""综合模型对比与优化

对比所有模型类型，测试不同的序列长度（注意力窗口），生成详细的对比报告
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


def train_model_with_config(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_length: int,
    units: int,
    dropout: float,
    epochs: int,
    **extra_kwargs: Any,
) -> dict[str, Any]:
    """训练单个模型配置"""
    config_str = f"{model_name}_seq{seq_length}_u{units}_d{dropout:.1f}"
    print(f"  🔧 配置: {config_str}")

    try:
        model = make_extended_model(
            model_name,
            seq_length=seq_length,
            units=units,
            dropout=dropout,
            learning_rate=0.001,
            epochs=epochs,
            batch_size=32,
            early_stopping_patience=5,
            **extra_kwargs
        )

        X_train = train_df[feature_cols + ["code", "date"]]
        y_train = train_df[target_col]

        model.fit(X_train, y_train)

        train_pred = model.predict(train_df[feature_cols + ["code", "date"]])
        test_pred = model.predict(test_df[feature_cols + ["code", "date"]])

        train_actual = train_df[target_col].values
        test_actual = test_df[target_col].values

        train_metrics = {
            "accuracy": float(np.mean((train_pred > 0) == (train_actual > 0))),
            "ic": float(spearmanr(train_pred, train_actual)[0]),
            "rmse": float(np.sqrt(np.mean((train_pred - train_actual) ** 2))),
        }

        test_metrics = {
            "accuracy": float(np.mean((test_pred > 0) == (test_actual > 0))),
            "ic": float(spearmanr(test_pred, test_actual)[0]),
            "rmse": float(np.sqrt(np.mean((test_pred - test_actual) ** 2))),
        }

        return {
            "model_name": model_name,
            "config": config_str,
            "seq_length": seq_length,
            "units": units,
            "dropout": dropout,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "success": True,
        }

    except Exception as e:
        print(f"    ❌ 失败: {e}")
        return {"success": False, "error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="综合模型对比与优化")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "model_comparison")
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🎯 综合模型对比与优化")
    print("="*80)
    print(f"📁 训练面板: {args.panel}")
    print(f"🔄 训练轮数: {args.epochs}")
    print("="*80 + "\n")

    if not args.panel.exists():
        print(f"❌ 训练面板不存在: {args.panel}")
        return 1

    try:
        train_df, test_df, feature_cols, target_col = load_and_split_data(
            args.panel, args.test_days
        )

        # 定义实验配置
        experiments = [
            # 序列长度实验 (注意力窗口大小)
            ("lstm", {"seq_length": 5, "units": 64, "dropout": 0.3}),
            ("lstm", {"seq_length": 10, "units": 64, "dropout": 0.3}),
            ("lstm", {"seq_length": 15, "units": 64, "dropout": 0.3}),
            ("lstm", {"seq_length": 20, "units": 64, "dropout": 0.3}),

            ("attention_lstm", {"seq_length": 5, "units": 64, "dropout": 0.3}),
            ("attention_lstm", {"seq_length": 10, "units": 64, "dropout": 0.3}),
            ("attention_lstm", {"seq_length": 15, "units": 64, "dropout": 0.3}),
            ("attention_lstm", {"seq_length": 20, "units": 64, "dropout": 0.3}),

            # 单元数实验
            ("attention_lstm", {"seq_length": 15, "units": 32, "dropout": 0.3}),
            ("attention_lstm", {"seq_length": 15, "units": 64, "dropout": 0.3}),
            ("attention_lstm", {"seq_length": 15, "units": 128, "dropout": 0.3}),

            # Dropout 实验
            ("attention_lstm", {"seq_length": 15, "units": 64, "dropout": 0.2}),
            ("attention_lstm", {"seq_length": 15, "units": 64, "dropout": 0.3}),
            ("attention_lstm", {"seq_length": 15, "units": 64, "dropout": 0.4}),

            # 其他模型类型
            ("gru", {"seq_length": 15, "units": 64, "dropout": 0.3}),
            ("self_attention", {"seq_length": 15, "units": 64, "dropout": 0.3, "num_heads": 4, "ff_dim": 128}),
            ("hierarchical_attention", {"seq_length": 15, "units": 64, "dropout": 0.3}),
        ]

        print(f"🚀 将测试 {len(experiments)} 个模型配置\n")

        results = []
        for idx, (model_name, config) in enumerate(experiments, 1):
            print(f"\n{'='*80}")
            print(f"📊 实验 {idx}/{len(experiments)}: {model_name.upper()}")
            print(f"{'='*80}")

            result = train_model_with_config(
                model_name,
                train_df,
                test_df,
                feature_cols,
                target_col,
                epochs=args.epochs,
                **config
            )

            if result["success"]:
                results.append(result)
                test_m = result["test_metrics"]
                print(f"    ✓ 测试准确率: {test_m['accuracy']:.4f}")
                print(f"    ✓ 测试IC: {test_m['ic']:.4f}")
                print(f"    ✓ 测试RMSE: {test_m['rmse']:.6f}")

        if not results:
            print("\n❌ 没有成功的实验")
            return 1

        # 保存结果
        args.output.mkdir(parents=True, exist_ok=True)

        # 转换为DataFrame
        summary_data = []
        for r in results:
            summary_data.append({
                "model": r["model_name"],
                "config": r["config"],
                "seq_length": r["seq_length"],
                "units": r["units"],
                "dropout": r["dropout"],
                "train_accuracy": r["train_metrics"]["accuracy"],
                "train_ic": r["train_metrics"]["ic"],
                "train_rmse": r["train_metrics"]["rmse"],
                "test_accuracy": r["test_metrics"]["accuracy"],
                "test_ic": r["test_metrics"]["ic"],
                "test_rmse": r["test_metrics"]["rmse"],
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("test_accuracy", ascending=False)
        summary_df.to_csv(args.output / "model_comparison_summary.csv", index=False)

        # 保存详细结果
        with open(args.output / "model_comparison_details.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 结果已保存: {args.output}")

        # 打印总结
        print("\n" + "="*80)
        print("📊 实验总结")
        print("="*80)
        print(f"\n成功完成 {len(results)} 个实验\n")

        # Top 5
        print("🏆 Top 5 模型配置（按测试准确率）:\n")
        for idx, row in summary_df.head(5).iterrows():
            print(f"  {idx+1}. {row['config']}")
            print(f"     准确率: {row['test_accuracy']:.4f}")
            print(f"     IC: {row['test_ic']:.4f}")
            print(f"     RMSE: {row['test_rmse']:.6f}")
            print()

        # 分析
        print("\n📈 关键发现:\n")

        # 序列长度分析
        attn_lstm_results = summary_df[summary_df['model'] == 'attention_lstm']
        if len(attn_lstm_results) > 0:
            print("  序列长度影响 (Attention LSTM):")
            seq_analysis = attn_lstm_results.groupby('seq_length')['test_accuracy'].mean().sort_index()
            for seq_len, acc in seq_analysis.items():
                print(f"    seq_length={seq_len}: {acc:.4f}")
            print()

        print("\n" + "="*80)
        print("✅ 所有实验完成！")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
