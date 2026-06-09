#!/usr/bin/env python3
"""时间序列交叉验证调优脚本 - 测试模型鲁棒性

使用多个时间切分点，每次用过去作为训练集，未来作为测试集
评估模型在不同时间段的稳定性和泛化能力
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.lite_train import train_ridge, evaluate, backtest_purchase_strategy
from hp_ml.config import PROCESSED_DIR, REPORTS_DIR
from hp_ml.charts import write_horizontal_bar_chart


def generate_time_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_days: int = 180,
    test_days: int = 60,
) -> list[tuple[str, str, str]]:
    """生成时间序列交叉验证的切分点"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    min_date = df["date"].min()
    max_date = df["date"].max()
    total_days = (max_date - min_date).days

    if total_days < min_train_days + test_days:
        raise ValueError(f"数据不足：总天数 {total_days}，需要至少 {min_train_days + test_days} 天")

    splits = []
    available_days = total_days - min_train_days - test_days
    interval = available_days // (n_splits - 1) if n_splits > 1 else 0

    for i in range(n_splits):
        train_start = min_date
        train_end_offset = min_train_days + i * interval
        train_end = min_date + timedelta(days=train_end_offset)
        test_end = train_end + timedelta(days=test_days)

        if test_end > max_date:
            test_end = max_date

        splits.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))

    return splits


def evaluate_time_split(
    df: list[dict[str, Any]],
    feature_cols: list[str],
    target_col: str,
    train_start: str,
    train_end: str,
    test_end: str,
    l2: float = 1.0,
    top_k: int = 3,
    min_pred: float = 0.0,
    round_trip_cost_bps: float = 100.0,
    horizon: int = 5,
) -> dict[str, Any]:
    """在单个时间切分上评估模型"""
    train_df = [r for r in df if train_start <= str(r.get('date', '')) <= train_end]
    test_df = [r for r in df if train_end < str(r.get('date', '')) <= test_end]

    if len(train_df) == 0 or len(test_df) == 0:
        return {"error": "数据切分后为空", "train_rows": len(train_df), "test_rows": len(test_df)}

    try:
        model = train_ridge(train_df, feature_cols, target_col, l2=l2)
    except Exception as e:
        return {"error": f"训练失败：{e}"}

    train_metrics, _ = evaluate(train_df, model, target_col)
    test_metrics, test_preds = evaluate(test_df, model, target_col)
    strategy_metrics, _, _ = backtest_purchase_strategy(
        test_preds, target_col, top_k=top_k, min_pred=min_pred,
        round_trip_cost_bps=round_trip_cost_bps, horizon=horizon
    )

    return {
        "train_start": train_start,
        "train_end": train_end,
        "test_end": test_end,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_directional_accuracy": train_metrics.get("directional_accuracy", 0),
        "train_spearman_ic": train_metrics.get("spearman_ic_by_date", 0),
        "train_rmse": train_metrics.get("rmse", 0),
        "test_directional_accuracy": test_metrics.get("directional_accuracy", 0),
        "test_spearman_ic": test_metrics.get("spearman_ic_by_date", 0),
        "test_rmse": test_metrics.get("rmse", 0),
        "strategy_cumulative_return": strategy_metrics.get("strategy_cumulative_return", 0),
        "strategy_sharpe_like": strategy_metrics.get("sharpe_like", 0),
        "strategy_max_drawdown": strategy_metrics.get("strategy_max_drawdown", 0),
        "strategy_win_rate": strategy_metrics.get("trade_win_rate", 0),
        "strategy_trade_count": strategy_metrics.get("trade_count", 0),
    }


def run_time_series_cv(
    panel_path: Path,
    target_col: str = "fwd_ret_5",
    n_splits: int = 5,
    min_train_days: int = 180,
    test_days: int = 60,
    l2_grid: list[float] | None = None,
    top_k_grid: list[int] | None = None,
    min_pred_grid: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """运行时间序列交叉验证调优"""
    print("📊 加载训练面板数据...")
    panel_df = pd.read_csv(panel_path)
    print(f"  ✓ 数据行数：{len(panel_df)}")
    print(f"  ✓ 日期范围：{panel_df['date'].min()} ~ {panel_df['date'].max()}")

    exclude_cols = {'date', 'code', 'name', 'family_id', 'close', target_col, 'is_trainable', 'is_future'}
    feature_cols = [c for c in panel_df.columns if c not in exclude_cols and not c.startswith('fwd_')]
    print(f"  ✓ 特征数量：{len(feature_cols)}")

    df = panel_df.to_dict('records')

    print(f"\n⏱️  生成 {n_splits} 个时间切分...")
    splits = generate_time_splits(panel_df, n_splits, min_train_days, test_days)

    for i, (train_start, train_end, test_end) in enumerate(splits, 1):
        print(f"  切分 {i}: 训练 [{train_start} ~ {train_end}], 测试 [{train_end} ~ {test_end}]")

    if l2_grid is None:
        l2_grid = [0.1, 1.0, 10.0, 100.0]
    if top_k_grid is None:
        top_k_grid = [3, 5, 10]
    if min_pred_grid is None:
        min_pred_grid = [0.0, 0.005, 0.01]

    print("\n🔍 超参数搜索空间：")
    print(f"  L2 正则化：{l2_grid}")
    print(f"  Top K: {top_k_grid}")
    print(f"  最小预测：{min_pred_grid}")

    results = []
    total = len(splits) * len(l2_grid) * len(top_k_grid) * len(min_pred_grid)
    current = 0

    print(f"\n🚀 开始评估 {total} 个配置...\n")

    for split_idx, (train_start, train_end, test_end) in enumerate(splits, 1):
        print(f"📅 切分 {split_idx}/{n_splits}: {train_start} ~ {train_end} -> {test_end}")

        for l2 in l2_grid:
            for top_k in top_k_grid:
                for min_pred in min_pred_grid:
                    current += 1
                    result = evaluate_time_split(
                        df, feature_cols, target_col, train_start, train_end, test_end,
                        l2, top_k, min_pred
                    )

                    if "error" not in result:
                        result.update({"split_idx": split_idx, "l2": l2, "top_k": top_k, "min_pred": min_pred})
                        results.append(result)

                    if current % 10 == 0:
                        print(f"  进度：{current}/{total} ({current/total*100:.1f}%)")

    print(f"\n✅ 评估完成！共 {len(results)} 个有效结果\n")

    results_df = pd.DataFrame(results)
    summary = {
        "total_splits": n_splits,
        "total_configs": len(results),
        "avg_test_directional_accuracy": results_df["test_directional_accuracy"].mean(),
        "avg_test_spearman_ic": results_df["test_spearman_ic"].mean(),
        "std_test_directional_accuracy": results_df["test_directional_accuracy"].std(),
        "std_test_spearman_ic": results_df["test_spearman_ic"].std(),
        "avg_strategy_return": results_df["strategy_cumulative_return"].mean(),
        "std_strategy_return": results_df["strategy_cumulative_return"].std(),
        "best_by_accuracy": results_df.loc[results_df["test_directional_accuracy"].idxmax()].to_dict(),
        "best_by_return": results_df.loc[results_df["strategy_cumulative_return"].idxmax()].to_dict(),
    }

    return results_df, summary


def create_visualizations(results_df: pd.DataFrame, output_dir: Path) -> None:
    """创建可视化图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    split_accuracy = results_df.groupby("split_idx")["test_directional_accuracy"].mean()
    write_horizontal_bar_chart(
        output_dir / "split_accuracy.svg",
        title="不同时间切分的测试集准确率",
        subtitle="评估模型在不同时期的表现",
        rows=[(f"切分 {idx}", acc) for idx, acc in split_accuracy.items()],
        value_kind="pct",
    )

    l2_accuracy = results_df.groupby("l2")["test_directional_accuracy"].mean()
    write_horizontal_bar_chart(
        output_dir / "l2_accuracy.svg",
        title="L2 正则化参数对准确率的影响",
        subtitle="平均测试集方向准确率",
        rows=[(f"L2={l2}", acc) for l2, acc in l2_accuracy.items()],
        value_kind="pct",
    )

    print(f"📊 可视化图表已保存到：{output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="时间序列交叉验证调优")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "time_series_cv_results.csv")
    parser.add_argument("--summary", type=Path, default=REPORTS_DIR / "time_series_cv_summary.json")
    parser.add_argument("--charts", type=Path, default=REPORTS_DIR / "charts" / "time_series_cv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--target", default="fwd_ret_5")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🎯 时间序列交叉验证 - 模型鲁棒性测试")
    print("="*80)
    print(f"📁 训练面板：{args.panel}")
    print(f"🔢 切分数量：{args.n_splits}")
    print(f"📅 训练天数：{args.min_train_days}")
    print(f"📅 测试天数：{args.test_days}")
    print("="*80 + "\n")

    if not args.panel.exists():
        print(f"❌ 训练面板文件不存在：{args.panel}")
        return 1

    try:
        results_df, summary = run_time_series_cv(
            panel_path=args.panel,
            target_col=args.target,
            n_splits=args.n_splits,
            min_train_days=args.min_train_days,
            test_days=args.test_days,
        )

        results_df.to_csv(args.output, index=False)
        print(f"💾 详细结果已保存：{args.output}")

        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 汇总统计已保存：{args.summary}")

        create_visualizations(results_df, args.charts)

        print("\n" + "="*80)
        print("📊 鲁棒性分析结果")
        print("="*80)
        print(f"\n【平均表现】")
        print(f"  测试集方向准确率：{summary['avg_test_directional_accuracy']:.4f}")
        print(f"  测试集 Spearman IC: {summary['avg_test_spearman_ic']:.4f}")
        print(f"  策略平均收益：{summary['avg_strategy_return']:.4f}")

        print(f"\n【稳定性指标】")
        print(f"  准确率标准差：{summary['std_test_directional_accuracy']:.4f}")
        print(f"  IC 标准差：{summary['std_test_spearman_ic']:.4f}")

        best_acc = summary['best_by_accuracy']
        print(f"\n【最佳准确率配置】")
        print(f"  L2={best_acc['l2']}, Top_K={best_acc['top_k']}, Min_Pred={best_acc['min_pred']}")
        print(f"  测试集准确率：{best_acc['test_directional_accuracy']:.4f}")

        print("\n" + "="*80)
        print("✅ 时间序列交叉验证完成！")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
