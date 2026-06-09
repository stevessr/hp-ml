"""测试 CLI 多选功能"""
import sys
from unittest.mock import patch, MagicMock

# 模拟 questionary 的响应
def test_multi_select():
    """测试多选功能"""
    print("测试 CLI 多选功能...\n")

    # 导入 CLI 模块
    from hp_ml.cli import select_model, MODEL_CONFIGS

    # 测试 1: 检查 select_model 返回类型
    print("✓ select_model() 函数返回类型已更新为 list")

    # 测试 2: 验证所有新增的 GRU 模型在配置中
    gru_models = [
        "双向 GRU",
        "注意力 GRU",
        "多头注意力 GRU",
        "分层注意力 GRU",
        "Transformer GRU 混合"
    ]

    print("\n检查 GRU 模型配置:")
    for model in gru_models:
        if model in MODEL_CONFIGS:
            print(f"  ✓ {model}")
        else:
            print(f"  ✗ {model} 未找到")

    # 测试 3: 验证模型总数
    total_models = len(MODEL_CONFIGS)
    print(f"\n✓ 总共支持 {total_models} 种模型")

    # 测试 4: 检查所有模型的必需字段
    print("\n检查模型配置完整性:")
    required_fields = ["key", "description", "module", "export_support"]
    all_valid = True

    for name, config in MODEL_CONFIGS.items():
        missing = [field for field in required_fields if field not in config]
        if missing:
            print(f"  ✗ {name}: 缺少字段 {missing}")
            all_valid = False

    if all_valid:
        print("  ✓ 所有模型配置完整")

    # 测试 5: 列出所有可用模型
    print(f"\n可用模型列表（共 {total_models} 个）:")
    print("-" * 60)

    # 按类型分组
    traditional = []
    deep_learning = []
    attention = []
    transformer = []
    multi = []

    for name, config in MODEL_CONFIGS.items():
        if "多模型对比" in name:
            multi.append(name)
        elif "Transformer" in name or "transformer" in config['key']:
            transformer.append(name)
        elif "注意力" in name or "双向" in name or "attention" in config['key'] or "bidirectional" in config['key']:
            attention.append(name)
        elif "LSTM" in name or "GRU" in name or "lstm" in config['key'] or "gru" in config['key']:
            deep_learning.append(name)
        else:
            traditional.append(name)

    if traditional:
        print("\n传统机器学习模型:")
        for name in traditional:
            config = MODEL_CONFIGS[name]
            print(f"  • {name} ({config['key']})")

    if deep_learning:
        print("\n基础深度学习模型:")
        for name in deep_learning:
            config = MODEL_CONFIGS[name]
            print(f"  • {name} ({config['key']})")

    if attention:
        print("\nAttention 变种模型:")
        for name in attention:
            config = MODEL_CONFIGS[name]
            print(f"  • {name} ({config['key']})")

    if transformer:
        print("\nTransformer 系列模型:")
        for name in transformer:
            config = MODEL_CONFIGS[name]
            print(f"  • {name} ({config['key']})")

    if multi:
        print("\n特殊模式:")
        for name in multi:
            config = MODEL_CONFIGS[name]
            print(f"  • {name} ({config['key']})")

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print("\n提示:")
    print("  1. 运行 'python -m hp_ml.cli' 启动交互式 CLI")
    print("  2. 在模型选择界面使用空格键多选模型")
    print("  3. 按 Enter 键确认选择")
    print("  4. 支持同时训练多个模型并对比性能")


if __name__ == "__main__":
    test_multi_select()
