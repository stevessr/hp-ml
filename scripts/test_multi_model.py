#!/usr/bin/env python3
"""快速测试多模型训练流程"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hp_ml.multi_model_train import main


if __name__ == "__main__":
    # 快速测试配置：较短历史，较少模型
    test_args = [
        "--start", "20230101",
        "--end", "20240601",
        "--horizon", "5",
        "--max-etfs-per-index", "2",
        "--models", "ridge", "hgb", "enhanced_rf",
        "--train-ratio", "0.6",
        "--val-ratio", "0.2",
        "--test-ratio", "0.2",
        "--top-k", "2",
        "--min-pred", "0.0",
        "--transaction-cost", "0.001",
    ]

    print("=" * 80)
    print("多模型训练快速测试")
    print("=" * 80)

    main(test_args)

    print("\n" + "=" * 80)
    print("测试完成！检查 reports/ 目录查看结果")
    print("=" * 80)
