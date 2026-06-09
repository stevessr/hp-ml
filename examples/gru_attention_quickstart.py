"""
GRU Attention 变种模型快速启动示例
演示如何使用新增的 GRU 变种模型
"""
from hp_ml.models_extended import make_extended_model
import numpy as np
import pandas as pd


def create_sample_data(n_samples=200, n_features=10, seq_length=20):
    """创建示例时序数据"""
    print("创建示例数据...")

    dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")
    codes = ["ETF_001"] * n_samples

    data = {
        "code": codes,
        "date": dates.strftime("%Y%m%d").tolist(),
    }

    # 添加特征
    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(n_samples) * 0.1 + np.sin(np.linspace(0, 4*np.pi, n_samples))

    X = pd.DataFrame(data)
    y = pd.Series(np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.randn(n_samples) * 0.05, name="target")

    print(f"  数据形状: X={X.shape}, y={y.shape}")
    return X, y


def train_and_evaluate_model(model_type, model_name, X, y):
    """训练和评估单个模型"""
    print(f"\n{'='*60}")
    print(f"模型: {model_name} ({model_type})")
    print(f"{'='*60}")

    try:
        # 创建模型
        print("1. 创建模型...")
        model = make_extended_model(
            model_type,
            seq_length=20,
            units=32,
            epochs=10,
            batch_size=16,
            early_stopping_patience=5
        )
        print("   ✓ 模型创建成功")

        # 训练模型
        print("2. 训练模型...")
        model.fit(X, y)
        print("   ✓ 训练完成")

        # 预测
        print("3. 生成预测...")
        predictions = model.predict(X)
        print(f"   ✓ 预测形状: {predictions.shape}")

        # 简单评估
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # 只评估有效预测（排除前面的填充零）
        valid_mask = predictions != 0
        if valid_mask.sum() > 0:
            valid_y = y[valid_mask]
            valid_pred = predictions[valid_mask]

            mse = mean_squared_error(valid_y, valid_pred)
            mae = mean_absolute_error(valid_y, valid_pred)
            r2 = r2_score(valid_y, valid_pred)

            print("\n4. 评估结果:")
            print(f"   MSE:  {mse:.6f}")
            print(f"   MAE:  {mae:.6f}")
            print(f"   R²:   {r2:.6f}")

        return True

    except Exception as e:
        print(f"   ✗ 错误: {e}")
        return False


def main():
    """主函数"""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                                                           ║")
    print("║       GRU Attention 变种模型 - 快速启动示例              ║")
    print("║                                                           ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # 创建示例数据
    X, y = create_sample_data(n_samples=200, n_features=10)

    # 定义要测试的模型
    models = [
        ("gru", "基础 GRU"),
        ("bigru", "双向 GRU"),
        ("attention_gru", "注意力 GRU"),
        ("multihead_attention_gru", "多头注意力 GRU"),
        ("hierarchical_attention_gru", "分层注意力 GRU"),
    ]

    # 训练和评估每个模型
    results = {}
    for model_type, model_name in models:
        success = train_and_evaluate_model(model_type, model_name, X, y)
        results[model_name] = "✓ 成功" if success else "✗ 失败"

    # 打印总结
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    for model_name, status in results.items():
        print(f"  {model_name:30s} {status}")

    print("\n" + "="*60)
    print("所有模型测试完成！")
    print("="*60)
    print("\n提示:")
    print("  1. 使用 'python -m hp_ml.cli' 启动交互式界面")
    print("  2. 查看 'GRU_ATTENTION_MODELS.md' 了解详细文档")
    print("  3. 使用实际数据进行训练以获得更好的效果")
    print()


if __name__ == "__main__":
    main()
