#!/usr/bin/env python3
"""演示如何通过代码调用 HP-ML CLI 的各个功能模块

本脚本展示了在非交互式环境（如自动化脚本、测试环境）中如何直接调用
CLI 的底层功能，而不使用交互式菜单。
"""
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_train_ridge():
    """演示：训练 Ridge 模型"""
    print("\n" + "="*60)
    print("演示 1: 训练 Ridge 模型（快速示例）")
    print("="*60 + "\n")

    from hp_ml.train import main as train_main

    # 配置参数
    argv = [
        "--start", "20200101",
        "--horizon", "5",
        "--max-etfs-per-index", "2",
        "--test-days", "120",
        "--model", "ridge",
        "--data-source", "eastmoney",
    ]

    print("参数配置:")
    print(f"  起始日期: 20200101")
    print(f"  预测周期: 5 天")
    print(f"  每族 ETF 数: 2")
    print(f"  测试集天数: 120")
    print(f"  模型类型: Ridge")
    print(f"  数据源: 东方财富\n")

    # 执行训练
    train_main(argv)

    print("\n✅ Ridge 模型训练完成！")
    print("输出文件:")
    print("  - models/csi_broad_etf_model.joblib")
    print("  - reports/latest_predictions.csv")
    print("  - reports/training_summary.md")


def demo_train_multi_models():
    """演示：训练多个模型并对比"""
    print("\n" + "="*60)
    print("演示 2: 训练多个模型并对比")
    print("="*60 + "\n")

    from hp_ml.multi_model_train import main as multi_train_main

    # 配置参数
    argv = [
        "--start", "20200101",
        "--horizon", "5",
        "--max-etfs-per-index", "2",
        "--models", "ridge", "hgb",  # 只训练两个模型以节省时间
        "--top-k", "3",
    ]

    print("参数配置:")
    print(f"  起始日期: 20200101")
    print(f"  预测周期: 5 天")
    print(f"  每族 ETF 数: 2")
    print(f"  模型类型: Ridge, HGB")
    print(f"  回测 Top-K: 3\n")

    # 执行训练
    multi_train_main(argv)

    print("\n✅ 多模型训练完成！")
    print("输出文件:")
    print("  - models/model_ridge.joblib")
    print("  - models/model_hgb.joblib")
    print("  - reports/backtest_comparison.csv")
    print("  - reports/multi_model_summary.json")


def demo_export_tdx():
    """演示：导出通达信公式"""
    print("\n" + "="*60)
    print("演示 3: 导出通达信公式")
    print("="*60 + "\n")

    from hp_ml.export_tdx import main as export_main
    from hp_ml.config import MODELS_DIR

    # 检查模型是否存在
    model_path = MODELS_DIR / "csi_broad_etf_model.joblib"
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 demo_train_ridge() 训练模型")
        return

    # 配置参数
    argv = [
        "--model", str(model_path),
        "--out", "reports/tdx_formulas",
        "--all-families",
    ]

    print("参数配置:")
    print(f"  模型文件: {model_path}")
    print(f"  输出目录: reports/tdx_formulas")
    print(f"  导出模式: 所有指数族\n")

    # 执行导出
    export_main(argv)

    print("\n✅ 通达信公式导出完成！")
    print("输出文件:")
    print("  - reports/tdx_formulas/HPML_宽基_ETF_CSI_300.tdx")
    print("  - reports/tdx_formulas/HPML_宽基_ETF_CSI_500.tdx")
    print("  - reports/tdx_formulas/HPML_宽基_ETF_CSI_1000.tdx")
    print("  - ...")


def demo_backtest():
    """演示：回测评估"""
    print("\n" + "="*60)
    print("演示 4: 回测评估")
    print("="*60 + "\n")

    from hp_ml.config import REPORTS_DIR
    from hp_ml.backtest import backtest_strategy
    import pandas as pd

    # 检查预测文件是否存在
    pred_path = REPORTS_DIR / "holdout_predictions.csv"
    if not pred_path.exists():
        print(f"❌ 预测文件不存在: {pred_path}")
        print("请先运行 demo_train_ridge() 训练模型")
        return

    print("加载预测文件...")
    predictions = pd.read_csv(pred_path)

    # 找到目标列
    target_col = None
    for col in predictions.columns:
        if "forward_return" in col or "target" in col:
            target_col = col
            break

    if target_col is None:
        print("❌ 未找到目标列")
        return

    # 配置回测参数
    top_k = 3
    min_pred = 0.0
    transaction_cost = 0.001

    print("\n回测参数:")
    print(f"  Top-K: {top_k}")
    print(f"  最低预测阈值: {min_pred}")
    print(f"  交易成本: {transaction_cost:.2%}\n")

    # 执行回测
    print("执行回测...")
    metrics, trades = backtest_strategy(
        predictions,
        pred_col="prediction",
        target_col=target_col,
        top_k=top_k,
        min_pred_threshold=min_pred,
        transaction_cost=transaction_cost,
    )

    print("\n📈 回测结果:")
    print(f"  总收益率: {metrics.total_return:.2%}")
    print(f"  年化收益率: {metrics.annualized_return:.2%}")
    print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
    print(f"  索提诺比率: {metrics.sortino_ratio:.2f}")
    print(f"  最大回撤: {metrics.max_drawdown:.2%}")
    print(f"  卡玛比率: {metrics.calmar_ratio:.2f}")
    print(f"  胜率: {metrics.win_rate:.2%}")
    print(f"  盈亏比: {metrics.profit_factor:.2f}")
    print(f"  总交易次数: {metrics.total_trades}")
    print(f"  盈利次数: {metrics.winning_trades}")
    print(f"  亏损次数: {metrics.losing_trades}")

    # 保存结果
    backtest_path = REPORTS_DIR / "backtest_demo.csv"
    trades.to_csv(backtest_path, index=False)
    print(f"\n💾 回测明细已保存: {backtest_path}")


def demo_compare_models():
    """演示：模型对比"""
    print("\n" + "="*60)
    print("演示 5: 模型对比")
    print("="*60 + "\n")

    from hp_ml.config import REPORTS_DIR
    from hp_ml.backtest import compare_models_backtest
    import pandas as pd

    # 查找预测文件
    pred_files = list(REPORTS_DIR.glob("predictions_*.csv"))

    if len(pred_files) < 2:
        print(f"❌ 找到的预测文件少于 2 个 (找到 {len(pred_files)} 个)")
        print("请先运行 demo_train_multi_models() 训练多个模型")
        return

    print(f"找到 {len(pred_files)} 个模型的预测文件:")
    for f in pred_files:
        print(f"  - {f.name}")

    # 加载所有预测
    all_predictions = {}
    target_col = None

    for pred_file in pred_files:
        model_name = pred_file.stem.replace("predictions_", "")
        df = pd.read_csv(pred_file)

        if target_col is None:
            for col in df.columns:
                if "forward_return" in col or "target" in col:
                    target_col = col
                    break

        all_predictions[model_name] = df

    if not all_predictions or target_col is None:
        print("❌ 加载预测文件失败")
        return

    print(f"\n对比 {len(all_predictions)} 个模型...")

    # 执行对比
    comparison = compare_models_backtest(
        all_predictions,
        target_col=target_col,
        top_k=3,
        min_pred_threshold=0.0,
        transaction_cost=0.001,
    )

    print("\n📊 模型对比结果:\n")
    print(comparison.to_string(index=False))

    # 保存结果
    comparison_path = REPORTS_DIR / "model_comparison_demo.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\n💾 对比结果已保存: {comparison_path}")


def demo_full_workflow():
    """演示：完整流程"""
    print("\n" + "="*60)
    print("演示 6: 完整流程 (训练 → 回测 → 导出)")
    print("="*60 + "\n")

    print("第一步: 训练 Ridge 模型")
    demo_train_ridge()

    print("\n第二步: 回测评估")
    demo_backtest()

    print("\n第三步: 导出通达信公式")
    demo_export_tdx()

    print("\n" + "="*60)
    print("🎉 完整流程演示完成！")
    print("="*60)


def main():
    """主函数"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        HP-ML CLI 演示脚本                                 ║
║                                                           ║
║  展示如何通过代码调用 CLI 的各个功能模块                  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    if len(sys.argv) > 1:
        demo_name = sys.argv[1]
        demos = {
            "train": demo_train_ridge,
            "multi": demo_train_multi_models,
            "export": demo_export_tdx,
            "backtest": demo_backtest,
            "compare": demo_compare_models,
            "full": demo_full_workflow,
        }

        if demo_name in demos:
            demos[demo_name]()
        else:
            print(f"❌ 未知演示: {demo_name}")
            print(f"可用演示: {', '.join(demos.keys())}")
    else:
        print("使用方法:")
        print("  python demo_cli_usage.py <demo_name>")
        print("\n可用演示:")
        print("  train     - 训练 Ridge 模型")
        print("  multi     - 训练多个模型并对比")
        print("  export    - 导出通达信公式")
        print("  backtest  - 回测评估")
        print("  compare   - 模型对比")
        print("  full      - 完整流程")
        print("\n示例:")
        print("  python demo_cli_usage.py train")
        print("  python demo_cli_usage.py full")


if __name__ == "__main__":
    main()
