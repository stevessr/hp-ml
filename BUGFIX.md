# 深度学习模型训练错误修复

## 问题描述

运行 GRU 及其他深度学习模型训练时报错：
```
GRU 模型训练失败：'code'
```

## 根本原因

深度学习模型（LSTM、GRU 等）需要 `code` 和 `date` 列来构建时间序列：

1. `prepare_lstm_sequences()` 函数按 `code` 分组，按 `date` 排序来构建序列数据
2. 但在 `train_single_model()` 中，传递给模型的数据只包含特征列，不包含 `code` 和 `date`
3. 导致深度学习模型内部调用 `prepare_lstm_sequences()` 时因缺少必需列而失败

## 修复方案

### 1. 修改 `models_extended.py` 中的所有深度学习模型

**fit() 方法**：确保只选择 `code` 和 `date` 加特征列
```python
# 修复前
df_train = X.copy()  # 包含所有列，可能包含不需要的列
df_train["target"] = y.values

# 修复后
df_train = X[["code", "date"] + feature_cols].copy()  # 只选择需要的列
df_train["target"] = y.values
```

**predict() 方法**：同样只选择需要的列
```python
# 修复前
df_pred = X.copy()
df_pred["target"] = 0.0

# 修复后
df_pred = X[["code", "date"] + self.feature_names_].copy()
df_pred["target"] = 0.0
```

### 2. 修改 `multi_model_train.py` 中的 `train_single_model()`

区分传统 ML 模型和深度学习模型的数据需求：

```python
# 深度学习模型列表
deep_learning_models = [
    "lstm", "gru", "bilstm", "attention_lstm", "multihead_attention",
    "self_attention", "hierarchical_attention", "lstm_transformer",
    "transformer_xl", "memory_transformer", "gru_transformer",
    "cnn_ngram", "tcn", "wavenet"
]

needs_code_date = model_type.lower() in deep_learning_models

# 训练时根据模型类型选择不同的列
if needs_code_date:
    x_train = split_data.train[["code", "date"] + feature_cols]
else:
    x_train = split_data.train[feature_cols]

# 预测时同样处理
if needs_code_date:
    test_predictions["prediction"] = model.predict(
        test_predictions[["code", "date"] + feature_cols]
    )
else:
    test_predictions["prediction"] = model.predict(test_predictions[feature_cols])
```

## 验证结果

所有深度学习模型训练成功：

```
✓ LSTM 模型评估：
  MAE: 0.029534
  RMSE: 0.039340
  方向准确率：45.29%

✓ GRU 模型评估：
  MAE: 0.026998
  RMSE: 0.036111
  方向准确率：47.86%

✓ BiLSTM 模型评估：
  MAE: 0.024469
  RMSE: 0.031567
  方向准确率：44.65%
```

回测对比结果显示 GRU 表现最佳：
- 总收益：195.77%
- 年化收益：134.90%
- 夏普比率：2.63
- 最大回撤：-41.26%

## 受影响的模型

以下所有模型已修复：
- LSTM
- GRU
- BidirectionalLSTM
- AttentionLSTM
- MultiHeadAttentionLSTM
- SelfAttentionLSTM
- HierarchicalAttentionLSTM
- LSTMTransformer
- GRUTransformer
- TransformerXL
- MemoryAugmentedTransformer
- CNNNGram
- TemporalConvNet
- WaveNet

## 后续建议

1. 在 `make_extended_model()` 中添加模型类型验证，确保传入的 `model_type` 在支持列表中
2. 考虑为深度学习模型创建统一的基类，避免重复代码
3. 添加单元测试覆盖所有模型的 fit/predict 流程
