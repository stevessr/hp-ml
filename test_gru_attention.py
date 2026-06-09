"""测试 GRU Attention 变种模型"""
import numpy as np
import pandas as pd
from hp_ml.models_extended import (
    AttentionGRUModel,
    MultiHeadAttentionGRU,
    BidirectionalGRUModel,
    HierarchicalAttentionGRU,
    make_extended_model
)

def test_model_creation():
    """测试模型创建"""
    print("测试模型创建...")

    models = {
        "attention_gru": "注意力 GRU",
        "multihead_attention_gru": "多头注意力 GRU",
        "bigru": "双向 GRU",
        "hierarchical_attention_gru": "分层注意力 GRU",
        "gru_transformer": "GRU-Transformer 混合"
    }

    for key, name in models.items():
        try:
            model = make_extended_model(key, seq_length=10, units=32, epochs=1)
            print(f"✓ {name} 创建成功")
        except Exception as e:
            print(f"✗ {name} 创建失败: {e}")

    print()

def test_model_training():
    """测试模型训练（使用小数据集）"""
    print("测试模型训练...")

    # 创建小型测试数据
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # 模拟时序数据
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")
    codes = ["TEST"] * n_samples

    data = {
        "code": codes,
        "date": dates.strftime("%Y%m%d").tolist(),
    }

    # 添加特征
    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(n_samples)

    X = pd.DataFrame(data)
    y = pd.Series(np.random.randn(n_samples), name="target")

    # 测试每个模型
    models = {
        "attention_gru": AttentionGRUModel(seq_length=10, units=16, epochs=2, batch_size=16),
        "multihead_gru": MultiHeadAttentionGRU(seq_length=10, units=16, epochs=2, batch_size=16),
        "bigru": BidirectionalGRUModel(seq_length=10, units=16, epochs=2, batch_size=16),
        "hierarchical_gru": HierarchicalAttentionGRU(seq_length=10, units=16, epochs=2, batch_size=16),
    }

    for name, model in models.items():
        try:
            print(f"  训练 {name}...", end=" ")
            model.fit(X, y)
            predictions = model.predict(X)
            print(f"✓ 预测形状: {predictions.shape}")
        except Exception as e:
            print(f"✗ 失败: {e}")

    print()

def test_cli_config():
    """测试 CLI 配置"""
    print("测试 CLI 配置...")

    from hp_ml.cli import MODEL_CONFIGS

    gru_models = [
        "双向 GRU",
        "注意力 GRU",
        "多头注意力 GRU",
        "分层注意力 GRU",
        "Transformer GRU 混合"
    ]

    for model_name in gru_models:
        if model_name in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_name]
            print(f"✓ {model_name}: key={config['key']}, module={config['module']}")
        else:
            print(f"✗ {model_name} 未在 CLI 配置中找到")

    print()

if __name__ == "__main__":
    print("="*60)
    print("GRU Attention 变种模型集成测试")
    print("="*60)
    print()

    test_model_creation()
    test_model_training()
    test_cli_config()

    print("="*60)
    print("测试完成！")
    print("="*60)
