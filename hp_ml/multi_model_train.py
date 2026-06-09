"""多模型训练与对比主脚本"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .backtest import backtest_strategy, compare_models_backtest
from .config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .data_pipeline import time_series_split
from .data_sources import fetch_many_histories, today_yyyymmdd
from .features import build_feature_panel
from .model import evaluate_predictions, make_model
from .models_extended import make_extended_model
from .prediction_report import generate_prediction_report
from .reporting import make_training_summary, write_json
from .universe import discover_broad_etfs
from .visualization import create_all_visualizations, generate_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多模型训练与对比")
    parser.add_argument("--start", default="20180101", help="历史数据起始日期 YYYYMMDD")
    parser.add_argument("--end", default=today_yyyymmdd(), help="历史数据结束日期 YYYYMMDD")
    parser.add_argument("--horizon", type=int, default=5, help="预测未来 N 个交易日收益")
    parser.add_argument("--max-etfs-per-index", type=int, default=3, help="每个指数族保留前 N 只 ETF")
    parser.add_argument("--min-amount", type=float, default=0.0, help="最低成交额筛选")

    parser.add_argument(
        "--models",
        nargs="+",
        default=["hgb", "ridge", "enhanced_rf"],
        help="训练的模型列表，可选：hgb, ridge, rf, enhanced_rf, prophet, lstm"
    )

    parser.add_argument("--train-ratio", type=float, default=0.6, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="测试集比例")

    parser.add_argument("--top-k", type=int, default=3, help="回测时每次买入前 K 只")
    parser.add_argument("--min-pred", type=float, default=0.0, help="回测买入阈值")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="交易成本")

    parser.add_argument("--include-enhanced", action="store_true", help="包含增强指数 ETF")
    parser.add_argument("--include-style", action="store_true", help="包含风格/主题 ETF")
    parser.add_argument("--force", action="store_true", help="强制刷新缓存")
    parser.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", "none", "raw"], help="复权方式")

    return parser


def train_single_model(
    model_type: str,
    split_data: Any,
    feature_cols: list[str],
    target_col: str,
    random_state: int = 42,
) -> tuple[Any, dict, pd.DataFrame]:
    """训练单个模型并返回结果"""

    print(f"\n训练 {model_type.upper()} 模型...")

    # 选择模型
    if model_type in ["hgb", "ridge", "rf"]:
        model = make_model(model_type=model_type, random_state=random_state)
    else:
        model = make_extended_model(model_type=model_type, random_state=random_state)

    # 训练
    x_train = split_data.train[feature_cols]
    y_train = split_data.train[target_col].astype(float)
    model.fit(x_train, y_train)

    # 预测
    test_predictions = split_data.test.copy()
    test_predictions["prediction"] = model.predict(test_predictions[feature_cols])

    # 评估
    metrics = evaluate_predictions(test_predictions, pred_col="prediction", target_col=target_col)
    metrics["model_type"] = model_type

    return model, metrics, test_predictions


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # 1. 发现 ETF 候选池
    print("发现中证宽基 ETF 候选池...")
    universe = discover_broad_etfs(
        max_per_family=args.max_etfs_per_index,
        min_amount=args.min_amount,
        include_enhanced=args.include_enhanced,
        include_style=args.include_style,
        force=args.force,
    )
    universe_path = REPORTS_DIR / "latest_candidates.csv"
    universe.to_csv(universe_path, index=False)
    print(f"发现 {len(universe)} 只候选 ETF")

    # 2. 拉取历史数据
    print(f"\n拉取历史数据 ({args.start} - {args.end})...")
    codes = universe["code"].astype(str).str.zfill(6).tolist()
    histories = fetch_many_histories(codes, start=args.start, end=args.end, adjust=args.adjust, force=args.force)

    if not histories:
        raise RuntimeError("未获取到任何 ETF 历史数据")
    print(f"成功拉取 {len(histories)} 只 ETF 历史数据")

    # 3. 构建特征面板
    print("\n构建特征面板...")
    panel, feature_cols, target_col = build_feature_panel(histories, universe=universe, horizon=args.horizon)
    panel_path = PROCESSED_DIR / "training_panel.csv"
    panel.to_csv(panel_path, index=False)
    print(f"特征面板：{len(panel)} 行，{len(feature_cols)} 个特征")

    # 4. 数据划分
    print("\n划分训练集/验证集/测试集...")
    split_data = time_series_split(
        panel,
        target_col=target_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(f"训练集：{len(split_data.train)} 行 ({split_data.train_dates[0].date()} ~ {split_data.train_dates[1].date()})")
    print(f"验证集：{len(split_data.val)} 行 ({split_data.val_dates[0].date()} ~ {split_data.val_dates[1].date()})")
    print(f"测试集：{len(split_data.test)} 行 ({split_data.test_dates[0].date()} ~ {split_data.test_dates[1].date()})")

    # 5. 训练多个模型
    models_results = {}
    all_predictions = {}

    for model_type in args.models:
        try:
            model, metrics, predictions = train_single_model(
                model_type,
                split_data,
                feature_cols,
                target_col,
            )

            models_results[model_type] = {
                "model": model,
                "metrics": metrics,
                "predictions": predictions,
            }

            all_predictions[model_type] = predictions

            print(f"\n{model_type.upper()} 模型评估：")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  方向准确率：{metrics['directional_accuracy']:.2%}")
            print(f"  Spearman IC: {metrics.get('spearman_ic_by_date', 0):.4f}")

        except Exception as e:
            print(f"\n{model_type.upper()} 模型训练失败：{e}")
            continue

    if not models_results:
        raise RuntimeError("所有模型训练均失败")

    # 6. 回测对比
    print("\n\n执行回测对比...")
    backtest_comparison = compare_models_backtest(
        all_predictions,
        target_col=target_col,
        top_k=args.top_k,
        min_pred_threshold=args.min_pred,
        transaction_cost=args.transaction_cost,
    )

    print("\n回测结果排名：")
    print(backtest_comparison.to_string(index=False))

    # 7. 保存结果
    print("\n保存模型和报告...")

    # 保存所有模型
    for model_type, result in models_results.items():
        model_path = MODELS_DIR / f"model_{model_type}.joblib"
        joblib.dump({
            "model": result["model"],
            "feature_cols": feature_cols,
            "target_col": target_col,
            "horizon": args.horizon,
            "metrics": result["metrics"],
        }, model_path)
        print(f"  {model_type}: {model_path}")

    # 保存回测对比
    backtest_path = REPORTS_DIR / "backtest_comparison.csv"
    backtest_comparison.to_csv(backtest_path, index=False)

    # 保存各模型预测
    for model_type, predictions in all_predictions.items():
        pred_path = REPORTS_DIR / f"predictions_{model_type}.csv"
        predictions.to_csv(pred_path, index=False)

    # 保存汇总 JSON
    summary = {
        "training_date": pd.Timestamp.now().isoformat(),
        "data_range": {
            "start": args.start,
            "end": args.end,
            "train_dates": [str(d.date()) for d in split_data.train_dates],
            "val_dates": [str(d.date()) for d in split_data.val_dates],
            "test_dates": [str(d.date()) for d in split_data.test_dates],
        },
        "models": {
            model_type: result["metrics"]
            for model_type, result in models_results.items()
        },
        "backtest": backtest_comparison.to_dict(orient="records"),
        "best_model": backtest_comparison.iloc[0]["model"],
    }

    summary_path = REPORTS_DIR / "multi_model_summary.json"
    write_json(summary_path, summary)

    # 8. 生成可视化图表和报告
    print("\n生成可视化图表...")
    charts_dir = REPORTS_DIR / "charts"
    models_with_features = {
        model_type: {
            "model": result["model"],
            "feature_cols": feature_cols,
            "metrics": result["metrics"],
        }
        for model_type, result in models_results.items()
    }

    create_all_visualizations(
        backtest_comparison,
        models_with_features,
        all_predictions,
        target_col,
        charts_dir,
        top_k=args.top_k,
        transaction_cost=args.transaction_cost,
    )

    # 生成 Markdown 报告
    print("\n生成 Markdown 报告...")
    models_metrics = {model_type: result["metrics"] for model_type, result in models_results.items()}
    generate_markdown_report(
        backtest_comparison,
        models_metrics,
        REPORTS_DIR / "multi_model_report.md"
    )

    # 9. 生成预测报告（预测 vs 实际对比）
    print("\n生成预测详细报告...")
    prediction_reports_dir = REPORTS_DIR / "prediction_reports"

    for model_type, predictions in all_predictions.items():
        print(f"  生成 {model_type} 预测报告...")
        try:
            report_files = generate_prediction_report(
                predictions_df=predictions,
                prediction_col="prediction",
                actual_col=target_col,
                model_name=model_type,
                output_dir=prediction_reports_dir,
                date_col="date" if "date" in predictions.columns else None,
                code_col="code" if "code" in predictions.columns else None,
            )
            print(f"    ✓ 报告已生成：{report_files.get('report', 'N/A')}")
        except Exception as e:
            print(f"    ✗ 报告生成失败：{e}")

    print(f"\n训练完成！最佳模型：{summary['best_model']}")
    print(f"报告保存至：{REPORTS_DIR}")
    print(f"图表保存至：{charts_dir}")
    print(f"预测报告保存至：{prediction_reports_dir}")


if __name__ == "__main__":
    main()
