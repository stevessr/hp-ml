#!/usr/bin/env python3
"""增强的时间序列交叉验证 - 用于验证高级模型是否达到 80% 准确度"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from datetime import timedelta

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR
from hp_ml.charts import write_horizontal_bar_chart


def generate_time_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_days: int = 180,
    test_days: int = 30,
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
        train_end = min_date + timedelta(days=min_train_days + i * interval)
        test_end = train_end + timedelta(days=test_days)

        if test_end > max_date:
            test_end = max_date

        splits.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))

    return splits


def evaluate_split(
    model_package: dict[str, Any],
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_end: str,
    feature_cols: list[str],
    target_col: str,
) -> dict[str, Any]:
    """在单个时间切分上评估模型"""
    model = model_package['model']
    model_type = model_package['model_type']

    # 切分数据
    train_df = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
    test_df = df[(df['date'] > train_end) & (df['date'] <= test_end)]

    if len(train_df) == 0 or len(test_df) == 0:
        return {
            "error": "数据切分后为空",
            "train_rows": len(train_df),
            "test_rows": len(test_df)
        }

    # 准备特征
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df[target_col].values

    # 预测
    try:
        if model_type == "lightgbm":
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
        elif model_type == "xgboost":
            import xgboost as xgb
            dtrain = xgb.DMatrix(X_train)
            dtest = xgb.DMatrix(X_test)
            y_pred_train = model.predict(dtrain)
            y_pred_test = model.predict(dtest)
        else:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
    except Exception as e:
        return {"error": f"预测失败：{e}"}

    # 计算指标
    train_accuracy = np.mean((y_train >= 0) == (y_pred_train >= 0))
    test_accuracy = np.mean((y_test >= 0) == (y_pred_test >= 0))

    train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))

    # Spearman IC（按日期分组）
    test_df_copy = test_df.copy()
    test_df_copy['prediction'] = y_pred_test

    daily_ic = []
    for date, group in test_df_copy.groupby('date'):
        if len(group) >= 3:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(group[target_col], group['prediction'])
            if not np.isnan(ic):
                daily_ic.append(ic)

    mean_ic = np.mean(daily_ic) if daily_ic else 0

    return {
        "train_start": train_start,
        "train_end": train_end,
        "test_end": test_end,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "test_spearman_ic": float(mean_ic),
    }


def run_enhanced_cv(
    model_path: Path,
    panel_path: Path,
    target_col: str = "fwd_ret_5",
    n_splits: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """运行增强的时间序列交叉验证"""
    print("📦 加载模型...")
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)

    print(f"  ✓ 模型类型：{model_package['model_type']}")
    print(f"  ✓ 特征数量：{len(model_package['feature_cols'])}")

    print("\n📊 加载数据...")
    panel_df = pd.read_csv(panel_path)
    panel_df = panel_df[panel_df[target_col].notna()]
    print(f"  ✓ 数据行数：{len(panel_df)}")
    print(f"  ✓ 日期范围：{panel_df['date'].min()} ~ {panel_df['date'].max()}")

    print(f"\n⏱️ 生成 {n_splits} 个时间切分...")
    splits = generate_time_splits(panel_df, n_splits)

    for i, (train_start, train_end, test_end) in enumerate(splits, 1):
        print(f"  切分 {i}: 训练 [{train_start} ~ {train_end}], 测试 [{train_end} ~ {test_end}]")

    print("\n🚀 开始评估...\n")
    results = []

    for split_idx, (train_start, train_end, test_end) in enumerate(splits, 1):
        print(f"📅 切分 {split_idx}/{n_splits}: {train_start} ~ {test_end}")

        result = evaluate_split(
            model_package,
            panel_df,
            train_start,
            train_end,
            test_end,
            model_package['feature_cols'],
            target_col,
        )

        if "error" not in result:
            result['split_idx'] = split_idx
            results.append(result)
            print(f"  ✓ 测试准确率：{result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
        else:
            print(f"  ✗ {result['error']}")

    print(f"\n✅ 评估完成！共 {len(results)} 个有效结果\n")

    # 汇总统计
    test_accuracies = [r['test_accuracy'] for r in results]
    summary = {
        "model_type": model_package['model_type'],
        "n_splits": len(results),
        "mean_test_accuracy": float(np.mean(test_accuracies)),
        "std_test_accuracy": float(np.std(test_accuracies)),
        "min_test_accuracy": float(np.min(test_accuracies)),
        "max_test_accuracy": float(np.max(test_accuracies)),
        "median_test_accuracy": float(np.median(test_accuracies)),
        "target_achieved": float(np.mean(test_accuracies)) >= 0.80,
    }

    return results, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="增强的时间序列交叉验证")
    parser.add_argument("--model", type=Path, required=True, help="模型文件路径")
    parser.add_argument("--panel", type=Path, default=PROCESSED_DIR / "training_panel_lite.csv")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "enhanced_cv_results.json")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--target", default="fwd_ret_5")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("🎯 增强的时间序列交叉验证")
    print("="*80)
    print(f"📦 模型：{args.model}")
    print(f"📁 数据：{args.panel}")
    print(f"🔢 切分数：{args.n_splits}")
    print("="*80 + "\n")

    if not args.model.exists():
        print(f"❌ 模型文件不存在：{args.model}")
        return 1

    if not args.panel.exists():
        print(f"❌ 数据文件不存在：{args.panel}")
        return 1

    try:
        results, summary = run_enhanced_cv(
            model_path=args.model,
            panel_path=args.panel,
            target_col=args.target,
            n_splits=args.n_splits,
        )

        # 保存结果
        output_data = {
            "summary": summary,
            "splits": results,
        }

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"💾 结果已保存：{args.output}")

        # 显示汇总
        print("\n" + "="*80)
        print("📊 交叉验证结果")
        print("="*80)
        print(f"\n模型类型：{summary['model_type']}")
        print(f"切分数量：{summary['n_splits']}")
        print(f"\n测试集准确率：")
        print(f"  均值：{summary['mean_test_accuracy']:.4f} ({summary['mean_test_accuracy']*100:.2f}%)")
        print(f"  标准差：{summary['std_test_accuracy']:.4f}")
        print(f"  最小值：{summary['min_test_accuracy']:.4f} ({summary['min_test_accuracy']*100:.2f}%)")
        print(f"  最大值：{summary['max_test_accuracy']:.4f} ({summary['max_test_accuracy']*100:.2f}%)")
        print(f"  中位数：{summary['median_test_accuracy']:.4f} ({summary['median_test_accuracy']*100:.2f}%)")

        if summary['target_achieved']:
            print(f"\n🎉 ✅ 目标达成！平均准确率 >= 80%")
        else:
            gap = 0.80 - summary['mean_test_accuracy']
            print(f"\n⚠️ 未达标：距离 80% 目标还差 {gap*100:.2f}%")

        print("="*80 + "\n")

        return 0 if summary['target_achieved'] else 1

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
