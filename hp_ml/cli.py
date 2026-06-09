"""交互式 CLI：模型训练、导出、回测、对比的统一入口"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import questionary
    from questionary import Style
except ImportError:
    print("错误：缺少 questionary 依赖")
    print("请运行：pip install questionary")
    sys.exit(1)


# 自定义样式
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#f44336 bold'),
    ('pointer', 'fg:#673ab7 bold'),
    ('highlighted', 'fg:#673ab7 bold'),
    ('selected', 'fg:#cc5454'),
    ('separator', 'fg:#cc5454'),
    ('instruction', ''),
    ('text', ''),
])


# 模型配置
MODEL_CONFIGS = {
    "Ridge 岭回归（快速，推荐）": {
        "key": "ridge",
        "description": "线性模型，训练快速，支持精确导出通达信公式",
        "module": "train",
        "export_support": "精确",
    },
    "HGB 梯度提升树": {
        "key": "hgb",
        "description": "集成学习，性能优秀，支持近似导出",
        "module": "train",
        "export_support": "近似",
    },
    "随机森林": {
        "key": "rf",
        "description": "集成学习，稳定性好，支持近似导出",
        "module": "train",
        "export_support": "近似",
    },
    "增强随机森林": {
        "key": "enhanced_rf",
        "description": "特征增强的随机森林，性能更优",
        "module": "multi_model",
        "export_support": "近似",
    },
    "Prophet 时间序列": {
        "key": "prophet",
        "description": "Facebook 时间序列模型，适合趋势预测",
        "module": "multi_model",
        "export_support": "知识蒸馏",
    },
    "LSTM 深度学习": {
        "key": "lstm",
        "description": "深度学习模型，捕捉时序依赖",
        "module": "multi_model",
        "export_support": "知识蒸馏",
    },
    "多模型对比": {
        "key": "multi",
        "description": "训练多个模型并生成对比报告",
        "module": "multi_model",
        "export_support": "各自方式",
    },
}


# 操作配置
OPERATION_CONFIGS = {
    "训练模型": {
        "key": "train",
        "description": "训练新模型或更新现有模型",
        "icon": "🎯",
    },
    "导出到通达信": {
        "key": "export",
        "description": "将模型导出为通达信公式文件",
        "icon": "📤",
    },
    "回测评估": {
        "key": "backtest",
        "description": "对训练好的模型进行回测评估",
        "icon": "📊",
    },
    "模型对比": {
        "key": "compare",
        "description": "对比多个模型的性能指标",
        "icon": "⚖️",
    },
    "完整流程": {
        "key": "full",
        "description": "训练→回测→导出→对比 一条龙",
        "icon": "🚀",
    },
}


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        HP-ML 交互式 CLI - ETF 量化训练工具                ║
║                                                           ║
║  支持模型：Ridge, HGB, RF, Prophet, LSTM                  ║
║  功能：训练 | 导出通达信 | 回测 | 对比                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def select_model() -> dict[str, Any]:
    """选择模型类型"""
    choices = [
        f"{name} - {config['description']}"
        for name, config in MODEL_CONFIGS.items()
    ]

    answer = questionary.select(
        "请选择模型类型：",
        choices=choices,
        style=custom_style,
    ).ask()

    if answer is None:
        sys.exit(0)

    # 提取模型名称
    model_name = answer.split(" - ")[0]
    return MODEL_CONFIGS[model_name]


def select_operation() -> dict[str, Any]:
    """选择操作类型"""
    choices = [
        f"{config['icon']} {name} - {config['description']}"
        for name, config in OPERATION_CONFIGS.items()
    ]

    answer = questionary.select(
        "请选择操作类型：",
        choices=choices,
        style=custom_style,
    ).ask()

    if answer is None:
        sys.exit(0)

    # 提取操作名称（去掉 icon）
    operation_name = answer.split(" - ")[0].split(" ", 1)[1]
    return OPERATION_CONFIGS[operation_name]


def get_training_params() -> dict[str, Any]:
    """获取训练参数"""
    print("\n⚙️  配置训练参数（按 Enter 使用默认值）\n")

    start_date = questionary.text(
        "历史数据起始日期 (YYYYMMDD)：",
        default="20180101",
        style=custom_style,
    ).ask()

    horizon = questionary.text(
        "预测未来 N 个交易日收益：",
        default="5",
        style=custom_style,
    ).ask()

    max_etfs = questionary.text(
        "每个指数族保留前 N 只 ETF：",
        default="3",
        style=custom_style,
    ).ask()

    test_days = questionary.text(
        "测试集交易日数：",
        default="252",
        style=custom_style,
    ).ask()

    data_source = questionary.select(
        "数据源：",
        choices=["东方财富 (默认)", "通达信 (更稳定)"],
        style=custom_style,
    ).ask()

    return {
        "start": start_date,
        "horizon": int(horizon),
        "max_etfs_per_index": int(max_etfs),
        "test_days": int(test_days),
        "data_source": "tdx" if "通达信" in data_source else "eastmoney",
    }


def get_export_params() -> dict[str, Any]:
    """获取导出参数"""
    print("\n📤 配置导出参数\n")

    from .config import MODELS_DIR

    # 列出可用模型
    model_files = list(MODELS_DIR.glob("*.pkl")) + list(MODELS_DIR.glob("*.joblib"))

    if not model_files:
        print("❌ 未找到训练好的模型文件")
        print("请先运行训练操作")
        sys.exit(1)

    model_choices = [str(f.name) for f in model_files]
    model_file = questionary.select(
        "选择要导出的模型：",
        choices=model_choices,
        style=custom_style,
    ).ask()

    export_mode = questionary.select(
        "导出模式：",
        choices=["所有指数族（批量导出）", "单个指数族"],
        style=custom_style,
    ).ask()

    family_id = None
    if "单个" in export_mode:
        family_id = questionary.select(
            "选择指数族：",
            choices=["CSI_300", "CSI_500", "CSI_1000", "CSI_2000", "CSI_800", "CSI_A500"],
            style=custom_style,
        ).ask()

    return {
        "model": str(MODELS_DIR / model_file),
        "all_families": "所有" in export_mode,
        "family_id": family_id,
    }


def get_backtest_params() -> dict[str, Any]:
    """获取回测参数"""
    print("\n📊 配置回测参数\n")

    top_k = questionary.text(
        "每次买入前 K 只 ETF：",
        default="3",
        style=custom_style,
    ).ask()

    min_pred = questionary.text(
        "买入预测收益阈值：",
        default="0.0",
        style=custom_style,
    ).ask()

    transaction_cost = questionary.text(
        "交易成本（完整买卖，bps）：",
        default="10",
        style=custom_style,
    ).ask()

    return {
        "top_k": int(top_k),
        "min_pred": float(min_pred),
        "transaction_cost": float(transaction_cost) / 10000,
    }


def execute_training(model_config: dict, params: dict):
    """执行训练"""
    print("\n" + "="*60)
    print(f"🎯 开始训练：{model_config['key'].upper()} 模型")
    print("="*60 + "\n")

    if model_config['key'] == 'multi':
        # 多模型训练
        from .multi_model_train import main as train_main

        models_to_train = questionary.checkbox(
            "选择要训练的模型（空格选择，Enter确认）：",
            choices=["ridge", "hgb", "rf", "enhanced_rf", "prophet", "lstm"],
            style=custom_style,
        ).ask()

        if not models_to_train:
            print("❌ 未选择任何模型")
            return

        argv = [
            "--start", params["start"],
            "--horizon", str(params["horizon"]),
            "--max-etfs-per-index", str(params["max_etfs_per_index"]),
            "--models", *models_to_train,
        ]

        train_main(argv)
    else:
        # 单模型训练
        from .train import main as train_main

        argv = [
            "--start", params["start"],
            "--horizon", str(params["horizon"]),
            "--max-etfs-per-index", str(params["max_etfs_per_index"]),
            "--test-days", str(params["test_days"]),
            "--model", model_config['key'],
            "--data-source", params["data_source"],
        ]

        train_main(argv)

    print("\n✅ 训练完成！")


def execute_export(params: dict):
    """执行导出"""
    print("\n" + "="*60)
    print("📤 开始导出到通达信")
    print("="*60 + "\n")

    from .export_tdx import main as export_main

    argv = ["--model", params["model"]]

    if params["all_families"]:
        argv.extend(["--all-families", "--out", "reports/tdx_formulas"])
    elif params["family_id"]:
        argv.extend(["--family-id", params["family_id"], "--out", f"reports/hp_ml_{params['family_id']}.tdx"])

    export_main(argv)

    print("\n✅ 导出完成！")


def execute_backtest(params: dict):
    """执行回测"""
    print("\n" + "="*60)
    print("📊 开始回测评估")
    print("="*60 + "\n")

    from .config import REPORTS_DIR
    import pandas as pd
    from .backtest import backtest_strategy

    # 查找预测文件
    pred_files = list(REPORTS_DIR.glob("*predictions*.csv"))

    if not pred_files:
        print("❌ 未找到预测文件")
        print("请先运行训练操作")
        return

    pred_choices = [f.name for f in pred_files]
    pred_file = questionary.select(
        "选择预测文件：",
        choices=pred_choices,
        style=custom_style,
    ).ask()

    predictions = pd.read_csv(REPORTS_DIR / pred_file)

    # 确定列名
    pred_col = "prediction" if "prediction" in predictions.columns else "pred"
    target_col = None
    for col in predictions.columns:
        if "forward_return" in col or "target" in col:
            target_col = col
            break

    if target_col is None:
        print("❌ 未找到目标列")
        return

    metrics, trades = backtest_strategy(
        predictions,
        pred_col=pred_col,
        target_col=target_col,
        top_k=params["top_k"],
        min_pred_threshold=params["min_pred"],
        transaction_cost=params["transaction_cost"],
    )

    print("\n📈 回测结果：")
    print(f"  总收益率：{metrics.total_return:.2%}")
    print(f"  年化收益率：{metrics.annualized_return:.2%}")
    print(f"  夏普比率：{metrics.sharpe_ratio:.2f}")
    print(f"  最大回撤：{metrics.max_drawdown:.2%}")
    print(f"  胜率：{metrics.win_rate:.2%}")
    print(f"  总交易次数：{metrics.total_trades}")

    # 保存结果
    backtest_result_path = REPORTS_DIR / "backtest_result_cli.csv"
    trades.to_csv(backtest_result_path, index=False)
    print(f"\n💾 回测明细已保存：{backtest_result_path}")


def execute_compare():
    """执行模型对比"""
    print("\n" + "="*60)
    print("⚖️  开始模型对比")
    print("="*60 + "\n")

    from .config import REPORTS_DIR
    import pandas as pd
    from .backtest import compare_models_backtest

    # 查找所有预测文件
    pred_files = list(REPORTS_DIR.glob("predictions_*.csv"))

    if len(pred_files) < 2:
        print("❌ 找到的预测文件少于 2 个")
        print("请先训练多个模型")
        return

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

    # 对比
    comparison = compare_models_backtest(
        all_predictions,
        target_col=target_col,
        top_k=3,
        min_pred_threshold=0.0,
        transaction_cost=0.001,
    )

    print("\n📊 模型对比结果：\n")
    print(comparison.to_string(index=False))

    # 保存
    comparison_path = REPORTS_DIR / "model_comparison_cli.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\n💾 对比结果已保存：{comparison_path}")


def execute_full_workflow(model_config: dict):
    """执行完整流程"""
    print("\n" + "="*60)
    print("🚀 开始完整流程：训练 → 回测 → 导出 → 对比")
    print("="*60 + "\n")

    # 1. 训练
    params = get_training_params()
    execute_training(model_config, params)

    # 2. 回测
    if questionary.confirm("是否继续回测？", default=True, style=custom_style).ask():
        backtest_params = get_backtest_params()
        execute_backtest(backtest_params)

    # 3. 导出
    if questionary.confirm("是否导出到通达信？", default=True, style=custom_style).ask():
        export_params = get_export_params()
        execute_export(export_params)

    # 4. 对比
    if questionary.confirm("是否进行模型对比？", default=False, style=custom_style).ask():
        execute_compare()

    print("\n✅ 完整流程执行完毕！")


def main():
    """主入口"""
    try:
        print_banner()

        # 选择模型
        model_config = select_model()
        print(f"\n✓ 已选择：{model_config['key'].upper()} - {model_config['description']}")

        # 选择操作
        operation_config = select_operation()
        print(f"✓ 已选择：{operation_config['key']} - {operation_config['description']}\n")

        # 执行操作
        if operation_config['key'] == 'train':
            params = get_training_params()
            execute_training(model_config, params)

        elif operation_config['key'] == 'export':
            params = get_export_params()
            execute_export(params)

        elif operation_config['key'] == 'backtest':
            params = get_backtest_params()
            execute_backtest(params)

        elif operation_config['key'] == 'compare':
            execute_compare()

        elif operation_config['key'] == 'full':
            execute_full_workflow(model_config)

        print("\n" + "="*60)
        print("🎉 操作完成！感谢使用 HP-ML CLI")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  操作已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
