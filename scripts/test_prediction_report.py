#!/usr/bin/env python3
"""
测试预测报告生成功能

演示如何使用 prediction_report 模块生成完整的预测报告
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from hp_ml.prediction_report import generate_prediction_report


def generate_sample_predictions():
    """生成模拟预测数据用于演示"""
    np.random.seed(42)

    n_samples = 500
    n_stocks = 5

    data = []

    for stock_idx in range(n_stocks):
        code = f"{600000 + stock_idx:06d}"

        for i in range(n_samples // n_stocks):
            # 生成实际值（带趋势 + 噪声）
            trend = 0.0001 * i
            actual = trend + np.random.normal(0, 0.02)

            # 生成预测值（实际值 + 偏差 + 噪声）
            prediction = actual + np.random.normal(0, 0.01) + 0.002

            date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)

            data.append(
                {
                    "date": date,
                    "code": code,
                    "actual_return": actual,
                    "predicted_return": prediction,
                }
            )

    return pd.DataFrame(data)


def main():
    print("=" * 80)
    print("测试预测报告生成功能")
    print("=" * 80)

    # 生成模拟数据
    print("\n[1/3] 生成模拟预测数据...")
    predictions_df = generate_sample_predictions()
    print(f"  生成 {len(predictions_df)} 条预测记录")
    print(f"  涵盖 {predictions_df['code'].nunique()} 只股票")

    # 显示数据样本
    print("\n数据样本:")
    print(predictions_df.head(10))

    # 生成报告
    print("\n[2/3] 生成预测报告...")
    output_files = generate_prediction_report(
        predictions_df=predictions_df,
        prediction_col="predicted_return",
        actual_col="actual_return",
        model_name="demo_model",
        output_dir="reports/prediction_demo",
        date_col="date",
        code_col="code",
    )

    # 显示生成的文件
    print("\n[3/3] 报告生成完成！")
    print("\n生成的文件:")
    for file_type, file_path in output_files.items():
        print(f"  {file_type:15s}: {file_path}")

    print("\n" + "=" * 80)
    print("✓ 测试完成！")
    print(f"请查看报告: {output_files['report']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
