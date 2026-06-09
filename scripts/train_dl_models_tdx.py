#!/usr/bin/env python
"""使用通达信数据源训练深度学习模型

支持的模型：
- LSTM：长短期记忆网络
- Bidirectional LSTM：双向 LSTM
- Attention LSTM：带注意力机制的 LSTM
- Multi-Head Attention LSTM：多头注意力 LSTM
- Self-Attention LSTM：自注意力 LSTM
- Hierarchical Attention LSTM：层次注意力 LSTM
- Transformer XL：扩展 Transformer
- Memory-Augmented Transformer：记忆增强 Transformer
- GRU-Transformer：GRU-Transformer 混合模型
- LSTM-Transformer：LSTM-Transformer 混合模型
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hp_ml.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from hp_ml.tdx_data_source import fetch_etf_batch_tdx
from hp_ml.features import build_feature_panel
from hp_ml.universe import discover_broad_etfs
from hp_ml.data_pipeline import time_series_split
from hp_ml.models_extended import make_extended_model
from hp_ml.model import evaluate_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用通达信数据源训练深度学习模型")

    # 数据参数
    parser.add_argument("--start", default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260609", help="结束日期 YYYYMMDD")
    parser.add_argument("--max-etfs", type=int, default=3, help="每个指数族 ETF 数量")
    parser.add_argument("--horizon", type=int, default=5, help="预测周期（交易日）")

    # 模型选择
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
        help="要训练的模型列表",
    )

    # 训练参数
    parser.add_argument("--seq-length", type=int, default=20, help="序列长度")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 率")

    # 模型架构参数
    parser.add_argument("--units", type=int, default=64, help="LSTM/GRU 单元数")
    parser.add_argument("--num-heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--ff-dim", type=int, default=128, help="前馈层维度")
    parser.add_argument("--num-blocks", type=int, default=2, help="Transformer 块数")

    # 输出参数
    parser.add_argument("--output-dir", default=str(MODELS_DIR / "dl_models"), help="模型输出目录")
    parser.add_argument("--force", action="store_true", help="强制重新拉取数据")

    return parser


def prepare_data(args) -> tuple[pd.DataFrame, list[str], str]:
    """准备训练数据"""
    print("=" * 80)
    print("📦 准备训练数据")
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

    # 保存面板数据
    panel_path = PROCESSED_DIR / "training_panel_tdx.csv"
    panel.to_csv(panel_path, index=False)
    print(f"✓ 保存面板：{panel_path}")

    return panel, feature_cols, target_col


def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    args,
) -> dict:
    """训练单个模型"""
    print(f"\n{'='*80}")
    print(f"🚀 训练模型：{model_name.upper()}")
    print(f"{'='*80}")

    # 构建模型
    model_kwargs = {
        "seq_length": args.seq_length,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "early_stopping_patience": 10,
    }

    # 根据模型类型添加特定参数
    if "lstm" in model_name or "gru" in model_name:
        model_kwargs["units"] = args.units

    if "attention" in model_name or "transformer" in model_name:
        model_kwargs["num_heads"] = args.num_heads
        model_kwargs["ff_dim"] = args.ff_dim

    if "transformer" in model_name:
        model_kwargs["num_transformer_blocks"] = args.num_blocks

    print(f"\n模型参数：")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")

    try:
        # 创建模型
        model = make_extended_model(model_name, **model_kwargs)

        # 准备训练数据
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        print(f"\n训练集：{len(X_train)} 样本")
        print(f"验证集：{len(X_val)} 样本")
        print(f"测试集：{len(X_test)} 样本")

        # 训练模型
        print(f"\n开始训练...")
        model.fit(X_train, y_train)

        # 预测
        print(f"\n生成预测...")
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        # 评估
        print(f"\n评估模型...")

        # 训练集评估
        train_metrics = evaluate_predictions(y_train.values, train_pred)
        print(f"\n训练集指标：")
        print(f"  MAE: {train_metrics['mae']:.4f}")
        print(f"  RMSE: {train_metrics['rmse']:.4f}")
        print(f"  方向准确率：{train_metrics['direction_accuracy']:.2%}")

        # 验证集评估
        val_metrics = evaluate_predictions(y_val.values, val_pred)
        print(f"\n验证集指标：")
        print(f"  MAE: {val_metrics['mae']:.4f}")
        print(f"  RMSE: {val_metrics['rmse']:.4f}")
        print(f"  方向准确率：{val_metrics['direction_accuracy']:.2%}")

        # 测试集评估
        test_metrics = evaluate_predictions(y_test.values, test_pred)
        print(f"\n测试集指标：")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  方向准确率：{test_metrics['direction_accuracy']:.2%}")

        # 保存模型
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{model_name}_model.pkl"

        import joblib
        joblib.dump({
            "model": model,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "model_name": model_name,
            "params": model_kwargs,
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
            "created_at": datetime.now().isoformat(),
        }, model_path)

        print(f"\n✓ 模型已保存：{model_path}")

        return {
            "model_name": model_name,
            "model_path": str(model_path),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
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


def main(argv: list[str] | None = None):
    """主函数"""
    args = build_parser().parse_args(argv)

    print("\n" + "="*80)
    print("使用通达信数据源训练深度学习模型")
    print("="*80)
    print(f"\n数据源：通达信 (TDX)")
    print(f"日期范围：{args.start} - {args.end}")
    print(f"预测周期：{args.horizon} 日")
    print(f"模型列表：{', '.join(args.models)}")
    print()

    # 1. 准备数据
    panel, feature_cols, target_col = prepare_data(args)

    # 2. 数据划分
    print(f"\n{'='*80}")
    print(f"📊 数据划分")
    print(f"{'='*80}")

    split = time_series_split(
        panel,
        target_col=target_col,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    print(f"\n训练集：{len(split.train)} 行")
    print(f"  日期：{split.train_dates[0]} ~ {split.train_dates[1]}")
    print(f"\n验证集：{len(split.val)} 行")
    print(f"  日期：{split.val_dates[0]} ~ {split.val_dates[1]}")
    print(f"\n测试集：{len(split.test)} 行")
    print(f"  日期：{split.test_dates[0]} ~ {split.test_dates[1]}")

    # 3. 训练所有模型
    results = []

    for model_name in args.models:
        result = train_model(
            model_name,
            split.train,
            split.val,
            split.test,
            feature_cols,
            target_col,
            args,
        )
        results.append(result)

    # 4. 汇总结果
    print(f"\n{'='*80}")
    print(f"📈 训练结果汇总")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r["success"])
    print(f"\n成功训练：{success_count}/{len(results)} 个模型\n")

    # 创建对比表格
    comparison_data = []
    for r in results:
        if r["success"]:
            comparison_data.append({
                "模型": r["model_name"],
                "训练 MAE": f"{r['train_metrics']['mae']:.4f}",
                "训练方向": f"{r['train_metrics']['direction_accuracy']:.2%}",
                "验证 MAE": f"{r['val_metrics']['mae']:.4f}",
                "验证方向": f"{r['val_metrics']['direction_accuracy']:.2%}",
                "测试 MAE": f"{r['test_metrics']['mae']:.4f}",
                "测试方向": f"{r['test_metrics']['direction_accuracy']:.2%}",
            })

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))

        # 保存对比结果
        output_dir = Path(args.output_dir)
        comparison_path = output_dir / "model_comparison.csv"
        df_comparison.to_csv(comparison_path, index=False)
        print(f"\n✓ 对比结果已保存：{comparison_path}")

    # 保存完整结果
    results_path = output_dir / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"✓ 完整结果已保存：{results_path}")

    print(f"\n{'='*80}")
    print(f"✅ 训练完成！")
    print(f"{'='*80}")
    print(f"\n模型目录：{args.output_dir}")
    print(f"面板数据：{PROCESSED_DIR / 'training_panel_tdx.csv'}")

    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
