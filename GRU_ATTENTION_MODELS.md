# GRU Attention 变种模型文档

本文档描述了新增的 GRU 及其 Attention 变种模型。

## 新增模型列表

### 1. 双向 GRU (Bidirectional GRU)
- **模型键**: `bigru` 或 `bidirectional_gru`
- **类名**: `BidirectionalGRUModel`
- **描述**: 双向 GRU 模型，同时捕捉前向和后向的时序依赖关系
- **优势**: 
  - 能够同时利用过去和未来的信息
  - 比单向 GRU 有更强的特征提取能力
  - 比双向 LSTM 计算效率更高

### 2. 注意力 GRU (Attention GRU)
- **模型键**: `attention_gru` 或 `gru_attention`
- **类名**: `AttentionGRUModel`
- **描述**: 带单头注意力机制的 GRU 模型
- **优势**:
  - 自动学习时序中的重要时间步
  - 提高模型对关键信息的关注度
  - 相比 LSTM 版本更轻量、更快

### 3. 多头注意力 GRU (Multi-Head Attention GRU)
- **模型键**: `multihead_attention_gru` 或 `multihead_gru`
- **类名**: `MultiHeadAttentionGRU`
- **描述**: 使用多头注意力机制的 GRU 模型
- **优势**:
  - 多个注意力头可以关注不同的特征维度
  - 提供更丰富的特征表示
  - 残差连接和层归一化提升训练稳定性

### 4. 分层注意力 GRU (Hierarchical Attention GRU)
- **模型键**: `hierarchical_attention_gru` 或 `hierarchical_gru`
- **类名**: `HierarchicalAttentionGRU`
- **描述**: 具有双层注意力机制的 GRU 模型，分别捕捉局部和全局模式
- **优势**:
  - 第一层关注短期局部模式
  - 第二层关注长期全局趋势
  - 同时捕捉多尺度时序特征

### 5. GRU-Transformer 混合 (GRU-Transformer Hybrid)
- **模型键**: `gru_transformer` 或 `transformer_gru`
- **类名**: `GRUTransformer`
- **描述**: GRU 编码器 + Transformer 块的混合架构
- **优势**:
  - GRU 提取局部时序特征
  - Transformer 捕捉长距离依赖
  - 结合两者的优势，性能更强
  - 比 LSTM-Transformer 更高效

## 模型对比

| 模型 | 参数量 | 训练速度 | 推理速度 | 内存占用 | 适用场景 |
|------|--------|----------|----------|----------|----------|
| GRU | 低 | 快 | 快 | 低 | 基础时序预测 |
| 双向 GRU | 中 | 中 | 中 | 中 | 需要双向信息 |
| 注意力 GRU | 中 | 中 | 中 | 中 | 重要时间步识别 |
| 多头注意力 GRU | 中高 | 中慢 | 中 | 中高 | 复杂特征关系 |
| 分层注意力 GRU | 高 | 慢 | 中慢 | 高 | 多尺度模式 |
| GRU-Transformer | 高 | 慢 | 中慢 | 高 | 长序列依赖 |

## 与 LSTM 变种对比

GRU 变种相比对应的 LSTM 变种有以下特点：

### 优势
1. **计算效率更高**: GRU 只有 2 个门（重置门和更新门），而 LSTM 有 3 个门（输入门、遗忘门、输出门）
2. **参数量更少**: 通常比 LSTM 少 25-30% 的参数
3. **训练速度更快**: 每个 epoch 的训练时间通常快 20-30%
4. **更不容易过拟合**: 参数少意味着在小数据集上表现可能更好

### 劣势
1. **记忆能力稍弱**: 在极长序列上，LSTM 的分离记忆单元可能表现更好
2. **表达能力**: 对于某些复杂任务，LSTM 的额外门控可能提供更细粒度的控制

## 使用示例

### 通过 CLI 使用

```bash
# 启动交互式 CLI
python -m hp_ml.cli

# 选择模型时，可以看到以下 GRU 变种：
# - GRU 深度学习
# - 双向 GRU
# - 注意力 GRU
# - 多头注意力 GRU
# - 分层注意力 GRU
# - Transformer GRU 混合
```

### 通过代码直接使用

```python
from hp_ml.models_extended import (
    AttentionGRUModel,
    MultiHeadAttentionGRU,
    BidirectionalGRUModel,
    HierarchicalAttentionGRU,
    GRUTransformer
)

# 创建注意力 GRU 模型
model = AttentionGRUModel(
    seq_length=20,
    units=64,
    dropout=0.2,
    learning_rate=0.001,
    epochs=50
)

# 训练
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)
```

### 通过 make_extended_model 工厂函数

```python
from hp_ml.models_extended import make_extended_model

# 创建模型
model = make_extended_model(
    "attention_gru",
    seq_length=20,
    units=64
)

# 其他模型类型
models = {
    "bigru": "双向 GRU",
    "attention_gru": "注意力 GRU",
    "multihead_attention_gru": "多头注意力 GRU",
    "hierarchical_attention_gru": "分层注意力 GRU",
    "gru_transformer": "GRU-Transformer 混合"
}
```

## 多模型对比训练

在 CLI 的"多模型对比"选项中，现在可以选择训练所有 GRU 变种并进行性能对比：

```bash
python -m hp_ml.cli

# 选择 "多模型对比"
# 勾选想要对比的模型：
# [ ] gru
# [ ] bigru
# [ ] attention_gru
# [ ] multihead_attention_gru
# [ ] hierarchical_attention_gru
# [ ] gru_transformer
```

## 模型参数说明

所有 GRU 变种模型共享以下基础参数：

- `seq_length`: 输入序列长度（默认 20）
- `units`: GRU 单元数量（默认 64）
- `dropout`: Dropout 率（默认 0.2）
- `learning_rate`: 学习率（默认 0.001）
- `epochs`: 训练轮数（默认 50）
- `batch_size`: 批次大小（默认 32）
- `early_stopping_patience`: 早停耐心值（默认 10）

特定模型的额外参数：

### MultiHeadAttentionGRU
- `num_heads`: 注意力头数量（默认 4）

### GRUTransformer
- `gru_units`: GRU 单元数（默认 64）
- `num_heads`: 注意力头数（默认 4）
- `ff_dim`: 前馈网络维度（默认 128）
- `num_transformer_blocks`: Transformer 块数量（默认 2）

## 性能建议

1. **小数据集**: 优先使用基础 GRU 或注意力 GRU，避免过拟合
2. **中等数据集**: 可以尝试多头注意力 GRU 或双向 GRU
3. **大数据集**: 分层注意力 GRU 或 GRU-Transformer 表现更好
4. **长序列**: GRU-Transformer 专门设计用于处理长序列依赖
5. **快速迭代**: 基础 GRU 或双向 GRU 训练最快

## 导出支持

所有 GRU 变种模型都支持通过**知识蒸馏**方式导出到通达信公式：

1. 使用 GRU 变种训练得到高精度模型
2. 使用知识蒸馏将复杂模型的知识转移到简单的线性模型
3. 导出线性模型到通达信公式

## 更新日志

### 2024-06-09
- 新增 5 个 GRU Attention 变种模型
- 更新 CLI 配置，支持新模型选择
- 更新模型工厂函数 `make_extended_model`
- 更新多模型对比训练流程
- 添加模型集成测试

## 注意事项

1. 所有深度学习模型都需要安装 TensorFlow
2. 建议使用 GPU 加速训练（如果可用）
3. 首次训练会自动下载和安装必要的依赖
4. 模型训练需要足够的内存（建议至少 8GB RAM）
5. 对于生产环境，建议使用验证集进行超参数调优

## 技术参考

这些模型的实现基于以下论文和技术：

1. **GRU**: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder"
2. **Attention Mechanism**: Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate"
3. **Multi-Head Attention**: Vaswani et al., "Attention Is All You Need"
4. **Hierarchical Attention**: Yang et al., "Hierarchical Attention Networks for Document Classification"
5. **Transformer**: Vaswani et al., "Attention Is All You Need"
