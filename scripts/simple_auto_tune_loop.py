#!/usr/bin/env python3
"""简化版自动化测试循环 - 使用现有ETF预测数据

直接使用已有的ETF预测，只循环测试不同的策略配置
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

# 项目根目录
ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT))

from hp_ml.auto_tune_baseline import (
    auto_tune_until_baseline,
    write_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="简化版自动测试循环")
    parser.add_argument("--start", default="2024-11-01", help="回测起始日期")
    parser.add_argument("--end", default="2026-01-28", help="回测结束日期")
    parser.add_argument("--min-excess", type=float, default=0.0, help="最小超额收益")
    parser.add_argument("--max-model-configs", type=int, default=3, help="最多测试多少个模型配置")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="手续费率")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🎯 简化版自动测试循环")
    print("="*80)
    print(f"📅 回测期间: {args.start} ~ {args.end}")
    print(f"🎚️  最小超额收益: {args.min_excess:.2%}")
    print(f"🔬 最多模型配置: {args.max_model_configs}")
    print("="*80 + "\n")

    # 检查必需的数据
    data_dir = ROOT / "data" / "topic2_broad_base"  # 使用topic2目录，包含正确格式的daily.csv文件
    sentiment_path = ROOT / "data" / "alternative_data" / "google_trends_sentiment.csv"

    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print(f"   尝试查找其他位置...")
        # 尝试其他可能的位置
        alt_dirs = [
            REPORTS / "lite_etf_history",
            ROOT / "data" / "topic2_broad_base",
        ]
        for alt_dir in alt_dirs:
            if alt_dir.exists():
                data_dir = alt_dir
                print(f"✅ 找到数据目录: {data_dir}")
                break
        else:
            return 1

    if not sentiment_path.exists():
        print(f"❌ 情绪数据不存在: {sentiment_path}")
        print(f"   尝试查找其他位置...")
        # 尝试其他可能的位置
        alt_sentiment = [
            REPORTS / "composite_sentiment_lite.csv",
            ROOT / "data" / "processed" / "sentiment.csv",
        ]
        for alt_sent in alt_sentiment:
            if alt_sent.exists():
                sentiment_path = alt_sent
                print(f"✅ 找到情绪数据: {sentiment_path}")
                break
        else:
            return 1

    print(f"✅ 数据目录: {data_dir}")
    print(f"✅ 情绪数据: {sentiment_path}\n")

    try:
        print("🚀 开始自动调优...\n")

        result = auto_tune_until_baseline(
            data_dir=data_dir,
            sentiment_path=sentiment_path,
            start=args.start,
            end=args.end,
            min_excess=args.min_excess,
            max_model_configs=args.max_model_configs,
            fee_rate=args.fee_rate,
        )

        print("\n" + "="*80)
        if result.success:
            print("🎊 成功！找到跑赢baseline的策略！")
        else:
            print("😔 未找到跑赢baseline的策略")
        print("="*80)

        print(f"\n📊 最佳策略指标:")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # 保存输出
        daily_out = REPORTS / "ml_auto_tune_until_baseline.csv"
        trials_out = REPORTS / "ml_auto_tune_trials.csv"
        summary_out = REPORTS / "ml_auto_tune_summary.json"

        write_outputs(
            result,
            daily_out=daily_out,
            trials_out=trials_out,
            summary_out=summary_out,
            start=args.start,
            end=args.end,
            data_dir=data_dir,
            sentiment_path=sentiment_path,
        )

        print(f"\n💾 输出文件:")
        print(f"  日度收益: {daily_out}")
        print(f"  所有试验: {trials_out}")
        print(f"  摘要: {summary_out}\n")

        # 检查最终性能
        if daily_out.exists():
            df = pd.read_csv(daily_out)
            if not df.empty:
                last = df.iloc[-1]
                strategy_cum = float(last.get("Strategy", 0))
                benchmark_cum = float(last.get("Benchmark", 0))
                excess = strategy_cum - benchmark_cum

                print(f"🏁 最终结果:")
                print(f"  策略累计收益: {strategy_cum:.4%}")
                print(f"  基准累计收益: {benchmark_cum:.4%}")
                print(f"  超额收益: {excess:.4%}")
                print(f"  {'✅ 跑赢baseline!' if excess >= args.min_excess else '❌ 未跑赢baseline'}\n")

                return 0 if excess >= args.min_excess else 1

        return 0 if result.success else 1

    except Exception as exc:
        print(f"\n❌ 错误: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
