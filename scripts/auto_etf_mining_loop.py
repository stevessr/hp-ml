#!/usr/bin/env python3
"""自动ETF股票挖掘与测试循环 - 直到跑赢baseline

这个脚本会循环执行：
1. 运行lite_train获取ETF预测
2. 使用stock_deep_dive深挖看涨股票
3. 运行auto_tune_baseline测试策略
4. 检查是否跑赢baseline
5. 如果没跑赢，调整参数并重试

目标：找到一个跑赢baseline的ETF股票挖掘配置
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# 项目根目录
ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
DATA = ROOT / "data"

# 循环配置空间
LITE_TRAIN_CONFIGS = [
    {"max_etfs_per_index": 3, "min_amount": 500_000_000, "test_days": 60, "auto_tune": True},
    {"max_etfs_per_index": 4, "min_amount": 300_000_000, "test_days": 90, "auto_tune": True},
    {"max_etfs_per_index": 5, "min_amount": 200_000_000, "test_days": 120, "auto_tune": True},
    {"max_etfs_per_index": 2, "min_amount": 400_000_000, "test_days": 45, "auto_tune": False},
]

DEEP_DIVE_CONFIGS = [
    {"top_etfs": 8, "stocks_per_family": 25, "max_candidates": 80, "top_stocks": 30},
    {"top_etfs": 10, "stocks_per_family": 30, "max_candidates": 100, "top_stocks": 40},
    {"top_etfs": 12, "stocks_per_family": 35, "max_candidates": 120, "top_stocks": 50},
    {"top_etfs": 6, "stocks_per_family": 20, "max_candidates": 60, "top_stocks": 20},
]


def run_command(cmd: list[str], desc: str, timeout: int = 600) -> tuple[bool, str]:
    """运行命令并返回是否成功"""
    print(f"\n{'='*80}")
    print(f"🚀 {desc}")
    print(f"📝 命令: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode == 0:
            print(f"✅ {desc} - 成功")
            return True, result.stdout
        else:
            print(f"❌ {desc} - 失败")
            print(f"错误输出:\n{result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        print(f"⏰ {desc} - 超时")
        return False, "Timeout"
    except Exception as exc:
        print(f"💥 {desc} - 异常: {exc}")
        return False, str(exc)


def check_baseline_performance(report_path: Path, min_excess: float = 0.0) -> tuple[bool, dict[str, Any]]:
    """检查策略是否跑赢baseline"""
    if not report_path.exists():
        return False, {"error": "report not found"}

    try:
        df = pd.read_csv(report_path)
        if df.empty:
            return False, {"error": "empty report"}

        # 获取最后一行的累计收益
        last = df.iloc[-1]
        strategy_cum = float(last.get("Strategy", last.get("strategy_return_raw", 0)))
        benchmark_cum = float(last.get("Benchmark", last.get("Benchmark", 0)))
        excess = strategy_cum - benchmark_cum

        metrics = {
            "strategy_cumulative": strategy_cum,
            "benchmark_cumulative": benchmark_cum,
            "excess_return": excess,
            "success": excess >= min_excess,
            "total_rows": len(df),
        }

        print(f"\n📊 性能指标:")
        print(f"  策略累计收益: {strategy_cum:.4%}")
        print(f"  基准累计收益: {benchmark_cum:.4%}")
        print(f"  超额收益: {excess:.4%}")
        print(f"  {'🎉 跑赢baseline!' if metrics['success'] else '😔 未跑赢baseline'}")

        return metrics["success"], metrics

    except Exception as exc:
        print(f"❌ 解析报告失败: {exc}")
        return False, {"error": str(exc)}


def run_lite_train(config: dict[str, Any], force: bool = False) -> tuple[bool, Path]:
    """运行lite_train生成ETF预测"""
    cmd = [
        sys.executable, "-m", "hp_ml.lite_train",
        "--top-n", str(config["top_n"]),
        "--min-amount", str(config["min_amount"]),
        "--lookback", str(config["lookback"]),
    ]
    if force:
        cmd.append("--force")

    success, _ = run_command(
        cmd,
        f"运行lite_train (top_n={config['top_n']}, min_amount={config['min_amount']}, lookback={config['lookback']})",
        timeout=900,
    )

    pred_path = REPORTS / "latest_predictions_lite.csv"
    return success and pred_path.exists(), pred_path


def run_deep_dive(config: dict[str, Any], force: bool = False) -> tuple[bool, Path]:
    """运行stock_deep_dive挖掘看涨股票"""
    cmd = [
        sys.executable, "-m", "hp_ml.stock_deep_dive",
        "--top-etfs", str(config["top_etfs"]),
        "--stocks-per-family", str(config["stocks_per_family"]),
        "--max-candidates", str(config["max_candidates"]),
        "--top-stocks", str(config["top_stocks"]),
    ]
    if force:
        cmd.append("--force")

    success, _ = run_command(
        cmd,
        f"运行deep_dive (top_etfs={config['top_etfs']}, stocks={config['top_stocks']})",
        timeout=1200,
    )

    output_path = REPORTS / "bullish_stock_deep_dive.csv"
    return success and output_path.exists(), output_path


def run_auto_tune_baseline(start: str, end: str, min_excess: float = 0.0) -> tuple[bool, Path]:
    """运行auto_tune_baseline测试策略"""
    cmd = [
        sys.executable, "-m", "hp_ml.auto_tune_baseline",
        "--start", start,
        "--end", end,
        "--min-excess", str(min_excess),
        "--max-model-configs", "1",
    ]

    success, _ = run_command(
        cmd,
        f"运行auto_tune_baseline (期间={start}~{end}, 最小超额={min_excess:.2%})",
        timeout=1800,
    )

    report_path = REPORTS / "ml_auto_tune_until_baseline.csv"
    return success and report_path.exists(), report_path


def save_iteration_log(
    iteration: int,
    lite_config: dict[str, Any],
    dive_config: dict[str, Any],
    metrics: dict[str, Any],
    log_path: Path,
) -> None:
    """保存迭代日志"""
    log_entry = {
        "iteration": iteration,
        "timestamp": dt.datetime.now().isoformat(),
        "lite_train_config": lite_config,
        "deep_dive_config": dive_config,
        "metrics": metrics,
    }

    # 追加到JSON日志
    logs = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                logs = json.load(f)
        except Exception:
            logs = []

    logs.append(log_entry)

    with open(log_path, "w") as f:
        json.dumps(logs, f, indent=2, ensure_ascii=False)

    print(f"💾 迭代日志已保存: {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="自动ETF股票挖掘与测试循环")
    parser.add_argument("--start", default="2024-11-01", help="回测起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-28", help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--min-excess", type=float, default=0.0, help="最小超额收益阈值")
    parser.add_argument("--max-iterations", type=int, default=20, help="最大迭代次数")
    parser.add_argument("--force", action="store_true", help="强制刷新所有缓存")
    parser.add_argument("--log-file", default="reports/auto_mining_loop_log.json", help="迭代日志文件")

    args = parser.parse_args()

    log_path = ROOT / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("🎯 自动ETF股票挖掘与测试循环启动")
    print("="*80)
    print(f"📅 回测期间: {args.start} ~ {args.end}")
    print(f"🎚️  最小超额收益: {args.min_excess:.2%}")
    print(f"🔄 最大迭代次数: {args.max_iterations}")
    print(f"💾 日志文件: {log_path}")
    print("="*80 + "\n")

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'#'*80}")
        print(f"🔁 迭代 {iteration}/{args.max_iterations}")
        print(f"{'#'*80}\n")

        # 选择配置（循环遍历配置空间）
        lite_idx = (iteration - 1) % len(LITE_TRAIN_CONFIGS)
        dive_idx = (iteration - 1) % len(DEEP_DIVE_CONFIGS)

        lite_config = LITE_TRAIN_CONFIGS[lite_idx]
        dive_config = DEEP_DIVE_CONFIGS[dive_idx]

        print(f"📋 本轮配置:")
        print(f"  Lite Train: {lite_config}")
        print(f"  Deep Dive: {dive_config}\n")

        # 1. 运行lite_train
        lite_success, pred_path = run_lite_train(lite_config, force=args.force or iteration == 1)
        if not lite_success:
            print("⚠️  lite_train失败，跳过本轮")
            continue

        # 2. 运行deep_dive
        dive_success, dive_path = run_deep_dive(dive_config, force=args.force or iteration == 1)
        if not dive_success:
            print("⚠️  deep_dive失败，跳过本轮")
            continue

        # 3. 运行auto_tune_baseline
        tune_success, report_path = run_auto_tune_baseline(args.start, args.end, args.min_excess)
        if not tune_success:
            print("⚠️  auto_tune_baseline失败，跳过本轮")
            continue

        # 4. 检查是否跑赢baseline
        success, metrics = check_baseline_performance(report_path, args.min_excess)

        # 保存迭代日志
        save_iteration_log(iteration, lite_config, dive_config, metrics, log_path)

        if success:
            print(f"\n{'='*80}")
            print("🎊🎊🎊 成功！找到跑赢baseline的配置！")
            print(f"{'='*80}")
            print(f"✅ 迭代次数: {iteration}")
            print(f"✅ Lite Train配置: {lite_config}")
            print(f"✅ Deep Dive配置: {dive_config}")
            print(f"✅ 策略累计收益: {metrics['strategy_cumulative']:.4%}")
            print(f"✅ 基准累计收益: {metrics['benchmark_cumulative']:.4%}")
            print(f"✅ 超额收益: {metrics['excess_return']:.4%}")
            print(f"{'='*80}\n")

            # 保存成功配置
            success_config = {
                "success": True,
                "iteration": iteration,
                "timestamp": dt.datetime.now().isoformat(),
                "lite_train_config": lite_config,
                "deep_dive_config": dive_config,
                "metrics": metrics,
                "backtest_period": {"start": args.start, "end": args.end},
            }

            success_path = REPORTS / "auto_mining_success_config.json"
            with open(success_path, "w") as f:
                json.dump(success_config, f, indent=2, ensure_ascii=False)

            print(f"💾 成功配置已保存: {success_path}\n")
            return 0

        print(f"\n💪 继续下一轮迭代...\n")
        time.sleep(2)  # 稍作停顿避免过载

    print(f"\n{'='*80}")
    print(f"😔 达到最大迭代次数({args.max_iterations})，未找到跑赢baseline的配置")
    print(f"{'='*80}")
    print(f"💡 建议:")
    print(f"  1. 增加 --max-iterations")
    print(f"  2. 调整配置空间 LITE_TRAIN_CONFIGS 和 DEEP_DIVE_CONFIGS")
    print(f"  3. 降低 --min-excess 阈值")
    print(f"  4. 检查日志文件: {log_path}")
    print(f"{'='*80}\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
