#!/usr/bin/env python3
"""测试 CLI 模块导入和基本功能"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """测试所有必要模块是否能正确导入"""
    print("测试模块导入...")

    errors = []

    # 测试 CLI 主模块
    try:
        from hp_ml import cli
        print("✓ hp_ml.cli 导入成功")
    except Exception as e:
        errors.append(f"✗ hp_ml.cli 导入失败: {e}")

    # 测试 questionary
    try:
        import questionary
        print("✓ questionary 导入成功")
    except Exception as e:
        errors.append(f"✗ questionary 导入失败: {e}")

    # 测试核心模块
    try:
        from hp_ml import train
        print("✓ hp_ml.train 导入成功")
    except Exception as e:
        errors.append(f"✗ hp_ml.train 导入失败: {e}")

    try:
        from hp_ml import multi_model_train
        print("✓ hp_ml.multi_model_train 导入成功")
    except Exception as e:
        errors.append(f"✗ hp_ml.multi_model_train 导入失败: {e}")

    try:
        from hp_ml import export_tdx
        print("✓ hp_ml.export_tdx 导入成功")
    except Exception as e:
        errors.append(f"✗ hp_ml.export_tdx 导入失败: {e}")

    try:
        from hp_ml import backtest
        print("✓ hp_ml.backtest 导入成功")
    except Exception as e:
        errors.append(f"✗ hp_ml.backtest 导入失败: {e}")

    if errors:
        print("\n❌ 导入测试失败：")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("\n✅ 所有模块导入成功！")
        return True


def test_cli_structure():
    """测试 CLI 结构和配置"""
    print("\n测试 CLI 结构...")

    try:
        from hp_ml.cli import MODEL_CONFIGS, OPERATION_CONFIGS

        print(f"✓ 模型配置数量: {len(MODEL_CONFIGS)}")
        print(f"✓ 操作配置数量: {len(OPERATION_CONFIGS)}")

        print("\n支持的模型:")
        for name, config in MODEL_CONFIGS.items():
            print(f"  - {name}: {config['key']}")

        print("\n支持的操作:")
        for name, config in OPERATION_CONFIGS.items():
            print(f"  - {name}: {config['key']}")

        return True
    except Exception as e:
        print(f"❌ CLI 结构测试失败: {e}")
        return False


def test_function_availability():
    """测试 CLI 各个功能函数是否可用"""
    print("\n测试功能函数...")

    try:
        from hp_ml.cli import (
            print_banner,
            select_model,
            select_operation,
            get_training_params,
            get_export_params,
            get_backtest_params,
        )

        print("✓ print_banner 函数可用")
        print("✓ select_model 函数可用")
        print("✓ select_operation 函数可用")
        print("✓ get_training_params 函数可用")
        print("✓ get_export_params 函数可用")
        print("✓ get_backtest_params 函数可用")

        # 测试打印横幅
        print("\n测试横幅显示:")
        print_banner()

        return True
    except Exception as e:
        print(f"❌ 功能函数测试失败: {e}")
        return False


def test_directories():
    """测试必要的目录是否存在"""
    print("\n测试目录结构...")

    from hp_ml.config import MODELS_DIR, REPORTS_DIR, PROCESSED_DIR

    dirs = {
        "模型目录": MODELS_DIR,
        "报告目录": REPORTS_DIR,
        "处理数据目录": PROCESSED_DIR,
    }

    for name, path in dirs.items():
        if path.exists():
            print(f"✓ {name} 存在: {path}")
        else:
            print(f"⚠ {name} 不存在，将自动创建: {path}")
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {name} 已创建")

    return True


def test_dependencies():
    """测试所有依赖是否已安装"""
    print("\n测试依赖包...")

    dependencies = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "joblib": "joblib",
        "questionary": "questionary",
        "prompt_toolkit": "prompt_toolkit",
        "matplotlib": "matplotlib",
    }

    missing = []

    for display_name, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✓ {display_name} 已安装")
        except ImportError:
            print(f"✗ {display_name} 未安装")
            missing.append(display_name)

    if missing:
        print(f"\n⚠ 缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False

    return True


def test_main_entry():
    """测试 __main__.py 入口"""
    print("\n测试 __main__.py 入口...")

    try:
        import hp_ml.__main__
        print("✓ hp_ml.__main__ 导入成功")
        print("✓ 可以使用 python -m hp_ml 启动")
        return True
    except Exception as e:
        print(f"❌ __main__.py 入口测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("HP-ML CLI 模块测试")
    print("="*60 + "\n")

    results = {
        "模块导入": test_imports(),
        "CLI 结构": test_cli_structure(),
        "功能函数": test_function_availability(),
        "目录结构": test_directories(),
        "依赖检查": test_dependencies(),
        "主入口": test_main_entry(),
    }

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60 + "\n")

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有测试通过！CLI 已就绪")
        print("\n可以运行以下命令启动 CLI：")
        print("  make cli")
        print("  python -m hp_ml.cli")
        print("  python -m hp_ml")
    else:
        print("⚠ 部分测试失败，请检查上述错误")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
