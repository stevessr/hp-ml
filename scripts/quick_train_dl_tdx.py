#!/usr/bin/env python
"""快速开始 - 使用通达信数据源训练深度学习模型

这是一个简化版本，用于快速测试和验证。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def quick_train_demo():
    """快速训练演示"""
    print("=" * 80)
    print("快速开始：使用通达信数据源训练 LSTM 模型")
    print("=" * 80)
    print()

    from scripts.train_dl_models_tdx import main

    # 使用最小参数快速训练
    args = [
        "--start", "20240101",
        "--end", "20260609",
        "--max-etfs", "2",  # 每个指数族只取 2 只 ETF
        "--horizon", "5",
        "--models", "lstm",  # 只训练 LSTM
        "--epochs", "20",  # 减少训练轮数
        "--seq-length", "10",  # 减少序列长度
        "--batch-size", "32",
    ]

    print("训练参数（快速模式）:")
    print("  - 数据源：通达信")
    print("  - 时间范围：2024-01-01 至 2026-06-09")
    print("  - ETF 数量：每个指数族 2 只")
    print("  - 模型：LSTM")
    print("  - 训练轮数：20 (减少以加快速度)")
    print("  - 序列长度：10")
    print()

    result = main(args)

    if result == 0:
        print("\n" + "=" * 80)
        print("✅ 快速训练完成！")
        print("=" * 80)
        print("\n下一步：")
        print("  1. 查看模型：models/dl_models/")
        print("  2. 查看对比：models/dl_models/model_comparison.csv")
        print("  3. 训练更多模型：")
        print("     python scripts/train_dl_models_tdx.py \\")
        print("       --models lstm bilstm attention_lstm lstm_transformer")

    return result


if __name__ == "__main__":
    try:
        sys.exit(quick_train_demo())
    except KeyboardInterrupt:
        print("\n\n已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
