#!/usr/bin/env python
"""演示使用通达信数据源进行训练

这个脚本展示如何使用通达信数据源拉取数据并训练模型。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_tdx_training():
    """演示通达信数据源训练"""
    print("=" * 60)
    print("演示：使用通达信数据源训练模型")
    print("=" * 60)
    print()

    # 导入模块
    from hp_ml.train import main as train_main

    # 构建参数
    args = [
        "--start", "20240101",
        "--end", "20260609",
        "--horizon", "5",
        "--test-days", "60",
        "--max-etfs-per-index", "2",
        "--model", "ridge",
        "--data-source", "tdx",  # 使用通达信数据源
        "--model-out", "models/csi_broad_etf_model_tdx.joblib",
    ]

    print("训练参数:")
    print(f"  数据源: 通达信 (TDX)")
    print(f"  时间范围: 20240101 - 20260609")
    print(f"  预测周期: 5 日")
    print(f"  测试集: 最近 60 天")
    print(f"  每个指数族: 前 2 只 ETF")
    print(f"  模型类型: Ridge 回归")
    print()

    print("开始训练...")
    print("-" * 60)

    try:
        train_main(args)
        print()
        print("=" * 60)
        print("✓ 训练完成！")
        print("=" * 60)
        print()
        print("模型文件: models/csi_broad_etf_model_tdx.joblib")
        print("预测结果: reports/latest_predictions.csv")
        print("训练报告: reports/training_summary.md")
        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("✗ 训练失败")
        print("=" * 60)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_comparison():
    """对比东方财富和通达信数据源"""
    print("\n" + "=" * 60)
    print("对比：东方财富 vs 通达信数据源")
    print("=" * 60)
    print()

    from hp_ml.data_sources import fetch_etf_history
    from hp_ml.tdx_data_source import fetch_etf_history_tdx

    test_code = "159300"  # 沪深300ETF

    print(f"测试代码: {test_code} (沪深300ETF)")
    print()

    # 东方财富
    print("1. 东方财富数据源:")
    try:
        df_em = fetch_etf_history(test_code)
        print(f"   ✓ 获取 {len(df_em)} 条记录")
        print(f"   时间范围: {df_em['date'].min()} 至 {df_em['date'].max()}")
        print(f"   最新收盘: {df_em['close'].iloc[-1]:.3f}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")

    print()

    # 通达信
    print("2. 通达信数据源:")
    try:
        df_tdx = fetch_etf_history_tdx(f"sz{test_code}", cache=False)
        print(f"   ✓ 获取 {len(df_tdx)} 条记录")
        print(f"   时间范围: {df_tdx['date'].min()} 至 {df_tdx['date'].max()}")
        print(f"   最新收盘: {df_tdx['close'].iloc[-1]:.3f}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("通达信数据源集成演示")
    print("=" * 60)
    print()

    # 演示1: 数据对比
    demo_comparison()

    print()
    input("按 Enter 继续训练演示（或 Ctrl+C 退出）...")

    # 演示2: 训练模型
    success = demo_tdx_training()

    if success:
        print("\n" + "=" * 60)
        print("使用说明")
        print("=" * 60)
        print()
        print("1. 使用通达信数据源训练:")
        print("   python -m hp_ml.train --data-source tdx")
        print()
        print("2. 使用东方财富数据源训练 (默认):")
        print("   python -m hp_ml.train")
        print()
        print("3. 完整参数示例:")
        print("   python -m hp_ml.train \\")
        print("     --data-source tdx \\")
        print("     --start 20200101 \\")
        print("     --horizon 5 \\")
        print("     --model ridge")
        print()

    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n已取消")
        sys.exit(1)
