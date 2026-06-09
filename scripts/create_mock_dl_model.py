"""创建模拟的深度学习模型用于测试 TDX 导出"""
import pickle
from pathlib import Path

# 创建模拟的 LSTM 模型 artifact
def create_mock_lstm_artifact():
    """创建一个模拟的 LSTM 模型 artifact 用于测试"""

    # 模拟 LSTM 模型类
    class MockLSTMModel:
        def __init__(self):
            self.model_type = "LSTMModel"

        def __class__(self):
            return type('LSTMModel', (), {})

    mock_model = MockLSTMModel()

    # 创建 artifact
    artifact = {
        "model": mock_model,
        "feature_cols": [
            "ret_1", "ret_5", "ret_20", "vol_20", "ma_gap_5_20",
            "drawdown_20", "amount_log", "turnover_rate",
            "family_CSI_300", "family_CSI_500"
        ],
        "target_col": "fwd_ret_5",
        "horizon": 5,
        "created_at": "2026-06-09T15:00:00"
    }

    return artifact, mock_model

# 创建并保存模拟模型
artifact, model = create_mock_lstm_artifact()

# 修复 model 的类名
model.__class__ = type('LSTMModel', (), {})

output_path = Path("models/model_lstm_test.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("wb") as f:
    pickle.dump(artifact, f)

print(f"✅ 已创建模拟 LSTM 模型：{output_path}")
print(f"   特征数：{len(artifact['feature_cols'])}")
print(f"   目标：{artifact['target_col']}")
