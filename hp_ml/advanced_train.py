"""AI 增强的自动化训练系统：整合所有高级功能"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .ai_assistant import AIAssistedEvolution
from .auto_tuning import HyperparameterTuner
from .backtest import compare_models_backtest
from .config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .data_pipeline import time_series_split
from .data_sources import fetch_many_histories, today_yyyymmdd
from .ensemble import create_ensemble
from .feature_learning import FeatureSelector, TechnicalIndicatorGenerator
from .features import build_feature_panel
from .model import make_model
from .prediction_report import generate_prediction_report
from .reporting import write_json
from .universe import discover_broad_etfs
from .visualization import create_all_visualizations, generate_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 增强的自动化训练系统")

    # 基础参数
    parser.add_argument("--start", default="20200101", help="历史数据起始日期")
    parser.add_argument("--end", default=today_yyyymmdd(), help="历史数据结束日期")
    parser.add_argument("--horizon", type=int, default=5, help="预测未来天数")
    parser.add_argument("--max-etfs-per-index", type=int, default=3, help="每指数族保留 ETF 数")

    # 模型选择
    parser.add_argument(
        "--base-models",
        nargs="+",
        default=["ridge", "hgb", "rf"],
        help="基础模型列表"
    )

    # 高级功能开关
    parser.add_argument("--enable-auto-tuning", action="store_true", help="启用自动超参数调优")
    parser.add_argument("--enable-ensemble", action="store_true", help="启用模型集成")
    parser.add_argument("--enable-feature-learning", action="store_true", help="启用特征自动学习")
    parser.add_argument("--enable-ai-assistant", action="store_true", help="启用 AI 辅助")

    # 超参数调优
    parser.add_argument("--tuning-trials", type=int, default=30, help="超参数搜索次数")

    # 集成学习
    parser.add_argument(
        "--ensemble-type",
        default="stacking",
        choices=["stacking", "blending", "weighted"],
        help="集成方法"
    )

    # 特征学习
    parser.add_argument("--auto-features", action="store_true", help="自动生成技术指标")
    parser.add_argument("--feature-selection", default="importance", help="特征选择方法")
    parser.add_argument("--n-features", type=int, default=None, help="保留特征数")

    # AI 辅助
    parser.add_argument("--anthropic-api-key", default=None, help="Anthropic API Key")

    # 其他
    parser.add_argument("--force", action="store_true", help="强制刷新缓存")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print("=" * 80)
    print("AI 增强的自动化训练系统")
    print("=" * 80)

    # ========== 步骤 1: 发现 ETF 池 ==========
    print("\n[1/8] 发现 ETF 候选池...")
    universe = discover_broad_etfs(
        max_per_family=args.max_etfs_per_index,
        force=args.force,
    )
    print(f"  发现 {len(universe)} 只候选 ETF")

    # ========== 步骤 2: 拉取历史数据 ==========
    print(f"\n[2/8] 拉取历史数据 ({args.start} - {args.end})...")
    codes = universe["code"].astype(str).str.zfill(6).tolist()
    histories = fetch_many_histories(codes, start=args.start, end=args.end, force=args.force)
    print(f"  成功拉取 {len(histories)} 只 ETF 历史数据")

    # ========== 步骤 3: 特征工程 ==========
    print("\n[3/8] 构建特征面板...")
    panel, feature_cols, target_col = build_feature_panel(histories, universe=universe, horizon=args.horizon)

    # 自动生成技术指标
    if args.auto_features:
        print("  生成技术指标...")
        try:
            indicator_gen = TechnicalIndicatorGenerator()
            panel_with_ta = []

            for code, group in panel.groupby("code"):
                group_ta = indicator_gen.generate(group)
                panel_with_ta.append(group_ta)

            panel = pd.concat(panel_with_ta, ignore_index=True)
            new_features = indicator_gen.generated_features_
            feature_cols.extend([f for f in new_features if f in panel.columns])
            print(f"  新增 {len(new_features)} 个技术指标特征")

        except Exception as e:
            print(f"  技术指标生成失败：{e}")

    print(f"  特征面板：{len(panel)} 行，{len(feature_cols)} 个特征")

    # ========== 步骤 4: 数据划分 ==========
    print("\n[4/8] 划分训练集/验证集/测试集...")
    split = time_series_split(panel, target_col=target_col)
    print(f"  训练集：{len(split.train)} 行")
    print(f"  验证集：{len(split.val)} 行")
    print(f"  测试集：{len(split.test)} 行")

    # ========== 步骤 5: 特征选择 ==========
    selected_features = feature_cols
    if args.enable_feature_learning and args.n_features:
        print(f"\n[5/8] 自动特征选择（保留 {args.n_features} 个）...")
        try:
            selector = FeatureSelector(
                method=args.feature_selection,
                n_features=args.n_features,
            )

            X_train = split.train[feature_cols].fillna(0)
            y_train = split.train[target_col]

            selector.fit(X_train, y_train)
            selected_features = selector.selected_features_

            print(f"  选择了 {len(selected_features)} 个特征")
            importance_df = selector.get_feature_importance()
            print(f"  Top 5 特征：{', '.join(importance_df.head(5)['feature'].tolist())}")

        except Exception as e:
            print(f"  特征选择失败：{e}")
            selected_features = feature_cols
    else:
        print("\n[5/8] 使用所有特征")

    # ========== 步骤 6: 训练基础模型 ==========
    print(f"\n[6/8] 训练基础模型（{len(args.base_models)} 个）...")

    base_models_trained = {}
    all_predictions = {}

    for model_type in args.base_models:
        print(f"\n  训练 {model_type.upper()} 模型...")

        # 自动超参数调优
        if args.enable_auto_tuning:
            print(f"    执行超参数优化（{args.tuning_trials} 次试验）...")
            try:
                tuner = HyperparameterTuner(
                    model_type=model_type,
                    n_trials=args.tuning_trials,
                    cv_folds=3,
                )

                X_train = split.train[selected_features].fillna(0)
                y_train = split.train[target_col]

                best_params = tuner.optimize(X_train, y_train)
                print(f"    最佳参数：{best_params}")

                # 使用最佳参数创建模型
                from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
                from sklearn.linear_model import Ridge

                if model_type == "ridge":
                    model = Ridge(**best_params)
                elif model_type == "hgb":
                    model = HistGradientBoostingRegressor(**best_params, random_state=42)
                elif model_type == "rf":
                    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
                else:
                    model = make_model(model_type)

            except Exception as e:
                print(f"    超参数优化失败：{e}，使用默认参数")
                model = make_model(model_type)
        else:
            model = make_model(model_type)

        # 训练模型
        X_train = split.train[selected_features].fillna(0)
        y_train = split.train[target_col]
        model.fit(X_train, y_train)

        # 预测
        test_predictions = split.test.copy()
        X_test = test_predictions[selected_features].fillna(0)
        test_predictions["prediction"] = model.predict(X_test)

        base_models_trained[model_type] = model
        all_predictions[model_type] = test_predictions

        print(f"    ✓ {model_type} 训练完成")

    # ========== 步骤 7: 模型集成 ==========
    if args.enable_ensemble and len(base_models_trained) > 1:
        print(f"\n[7/8] 构建集成模型（{args.ensemble_type}）...")

        try:
            base_models_list = list(base_models_trained.values())
            ensemble_model = create_ensemble(
                ensemble_type=args.ensemble_type,
                base_models=base_models_list,
                cv_folds=3,
                random_state=42,
            )

            X_train = split.train[selected_features].fillna(0)
            y_train = split.train[target_col]
            ensemble_model.fit(X_train, y_train)

            test_predictions_ensemble = split.test.copy()
            X_test = test_predictions_ensemble[selected_features].fillna(0)
            test_predictions_ensemble["prediction"] = ensemble_model.predict(X_test)

            base_models_trained["ensemble"] = ensemble_model
            all_predictions["ensemble"] = test_predictions_ensemble

            print(f"  ✓ 集成模型训练完成")

        except Exception as e:
            print(f"  集成模型失败：{e}")
    else:
        print("\n[7/8] 跳过集成学习")

    # ========== 步骤 8: AI 辅助分析（可选）==========
    ai_insights = {}
    if args.enable_ai_assistant and args.anthropic_api_key:
        print("\n[8/8] AI 辅助分析...")

        try:
            ai_assistant = AIAssistedEvolution(api_key=args.anthropic_api_key)

            # 分析性能
            from .model import evaluate_predictions

            for model_name, predictions in list(all_predictions.items())[:3]:  # 只分析前 3 个
                print(f"  分析 {model_name} 模型...")
                metrics = evaluate_predictions(predictions, "prediction", target_col)

                insights = ai_assistant.analyze_model_performance(
                    model_type=model_name,
                    metrics=metrics,
                )

                ai_insights[model_name] = insights
                print(f"    诊断：{insights.get('diagnosis', 'N/A')[:100]}...")

            # 特征建议
            print("  获取特征建议...")
            feature_suggestions = ai_assistant.suggest_features(
                existing_features=selected_features,
                data_description="中证宽基 ETF 日线数据",
                target_description=f"未来 {args.horizon} 日收益率",
            )

            ai_insights["feature_suggestions"] = feature_suggestions

        except Exception as e:
            print(f"  AI 辅助分析失败：{e}")
    else:
        print("\n[8/8] 跳过 AI 辅助分析")

    # ========== 保存结果 ==========
    print("\n保存模型和报告...")

    # 保存模型
    for model_name, model in base_models_trained.items():
        model_path = MODELS_DIR / f"advanced_model_{model_name}.joblib"
        joblib.dump({
            "model": model,
            "feature_cols": selected_features,
            "target_col": target_col,
            "horizon": args.horizon,
        }, model_path)

    # 回测对比
    backtest_comparison = compare_models_backtest(
        all_predictions,
        target_col=target_col,
        top_k=3,
        transaction_cost=0.001,
    )

    backtest_path = REPORTS_DIR / "advanced_backtest_comparison.csv"
    backtest_comparison.to_csv(backtest_path, index=False)

    # 保存 AI 洞察
    if ai_insights:
        ai_path = REPORTS_DIR / "ai_insights.json"
        write_json(ai_path, ai_insights)

    # 生成报告
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "config": {
            "base_models": args.base_models,
            "auto_tuning": args.enable_auto_tuning,
            "ensemble": args.enable_ensemble,
            "feature_learning": args.enable_feature_learning,
            "ai_assistant": args.enable_ai_assistant,
        },
        "features": {
            "total": len(feature_cols),
            "selected": len(selected_features),
        },
        "models": list(base_models_trained.keys()),
        "best_model": backtest_comparison.iloc[0]["model"],
    }

    summary_path = REPORTS_DIR / "advanced_training_summary.json"
    write_json(summary_path, summary)

    # 生成预测报告
    print("\n生成预测详细报告...")
    prediction_reports_dir = REPORTS_DIR / "advanced_prediction_reports"

    for model_name, predictions in all_predictions.items():
        print(f"  生成 {model_name} 预测报告...")
        try:
            report_files = generate_prediction_report(
                predictions_df=predictions,
                prediction_col="prediction",
                actual_col=target_col,
                model_name=f"advanced_{model_name}",
                output_dir=prediction_reports_dir,
                date_col="date" if "date" in predictions.columns else None,
                code_col="code" if "code" in predictions.columns else None,
            )
            print(f"    ✓ {report_files.get('report', 'N/A').name}")
        except Exception as e:
            print(f"    ✗ 失败：{e}")

    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"最佳模型：{summary['best_model']}")
    print(f"报告保存至：{REPORTS_DIR}")
    print(f"预测报告保存至：{prediction_reports_dir}")
    print("=" * 80)

    # 显示回测结果
    print("\n回测结果排名：")
    print(backtest_comparison[["model", "annualized_return", "sharpe_ratio", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
