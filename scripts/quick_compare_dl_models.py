#!/usr/bin/env python
"""快速对比深度学习模型效果和收益率

使用最少的参数快速对比常用的深度学习模型。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def quick_comparison():
    """快速对比"""
    print("=" * 80)
    print("快速对比：深度学习模型效果和收益率")
    print("=" * 80)
    print()

    from scripts.compare_dl_models_backtest import main

    # 使用精简参数快速对比
    args = [
        "--start", "20240101",
        "--end", "20260609",
        "--max-etfs", "2",  # 每个指数族2只ETF
        "--horizon", "5",
        "--models", "lstm", "attention_lstm", "lstm_transformer",  # 对比3个经典模型
        "--epochs", "30",  # 减少训练时间
        "--seq-length", "15",
        "--top-k", "2",  # 每次买入前2只
    ]

    print("对比设置：")
    print("  - 数据源：通达信")
    print("  - 时间范围：2024-01-01 至 2026-06-09")
    print("  - ETF数量：每个指数族2只")
    print("  - 对比模型：LSTM、Attention LSTM、LSTM-Transformer")
    print("  - 训练轮数：30")
    print("  - 回测策略：每次买入预测收益最高的2只ETF")
    print()
    print("预计耗时：15-25分钟")
    print()

    result = main(args)

    if result == 0:
        print("\n" + "=" * 80)
        print("✅ 对比完成！")
        print("=" * 80)
        print("\n查看结果：")
        print("  1. 对比表格：reports/dl_comparison/model_comparison.csv")
        print("  2. 对比图表：reports/dl_comparison/model_comparison.png")
        print("  3. 对比报告：reports/dl_comparison/comparison_report.md")
        print("\n扩展对比（训练更多模型）：")
        print("  python scripts/compare_dl_models_backtest.py \\")
        print("    --models lstm bilstm attention_lstm multihead_attention_lstm \\")
        print("             self_attention_lstm lstm_transformer")

    return result


if __name__ == "__main__":
    try:
        sys.exit(quick_comparison())
    except KeyboardInterrupt:
        print("\n\n已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
