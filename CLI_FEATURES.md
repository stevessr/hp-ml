# CLI 新功能说明

## 🎯 最新更新

### 1. 修复 PROPHET 模型训练 ✅

**问题**：PROPHET 模型训练时提示 "数据必须包含 'code' 和 'date' 列"

**解决方案**：
- 修改了 `multi_model_train.py`，将 "prophet" 添加到需要 code/date 列的模型列表
- Prophet 作为时间序列模型，需要按 ETF code 分组训练独立模型

**验证**：
```bash
python test_prophet_fix.py
```

---

### 2. 配置记忆功能 💾

**功能**：CLI 会自动记住你上次的选择，下次使用时作为默认值

**配置文件位置**：
```
~/.hp_ml_cli_config.json
```

**记忆的内容**：

#### 训练参数
- 历史数据起始日期 (默认: 20180101)
- 预测未来 N 个交易日收益 (默认: 5)
- 每个指数族保留前 N 只 ETF (默认: 3)
- 测试集交易日数 (默认: 252)
- 数据源选择 (东方财富/通达信)

#### 回测参数
- 每次买入前 K 只 ETF (默认: 3)
- 买入预测收益阈值 (默认: 0.0)
- 交易成本 bps (默认: 10)

#### 导出参数
- 导出模式（所有指数族/单个指数族）
- 选择的指数族

#### 多模型训练
- 上次选择的模型列表

**使用示例**：
```bash
# 第一次运行，输入参数
make cli

# 第二次运行，直接按 Enter 使用上次的参数
make cli
```

---

### 3. 多模型训练优化 🚀

**问题**：选择多个模型训练时，每个模型都会重复获取 ETF 数据，浪费时间

**解决方案**：
- 当选择多个模型时，自动使用 `multi_model_train.py` 批量训练
- 所有模型共享同一份数据（ETF 候选池、历史数据、特征面板）
- 只在最开始获取一次数据

**效果对比**：

| 场景 | 旧逻辑 | 新逻辑 |
|------|--------|--------|
| 训练 3 个模型 | 获取数据 3 次 | 获取数据 1 次 |
| 总耗时 | ~15 分钟 | ~5 分钟 |

---

### 4. 模型自动保存 💾

**保存位置**：
```
models/model_{model_type}.joblib
```

**保存内容**：
- 训练好的模型对象
- 特征列列表
- 目标列名称
- 预测周期
- 评估指标

**加载模型**：
```python
import joblib

# 加载模型
model_data = joblib.load('models/model_prophet.joblib')
model = model_data['model']
feature_cols = model_data['feature_cols']
metrics = model_data['metrics']

print(f"模型性能：MAE = {metrics['mae']:.4f}")
```

---

## 📝 使用流程

### 方式 1：交互式 CLI（推荐）
```bash
make cli
```

### 方式 2：直接指定模型训练
```bash
# 单模型训练
python -m hp_ml.multi_model_train --models prophet

# 多模型批量训练
python -m hp_ml.multi_model_train --models prophet lstm gru
```

---

## 🎨 支持的模型

### 传统机器学习
- `ridge` - 岭回归（快速，推荐）
- `hgb` - 梯度提升树
- `rf` - 随机森林
- `enhanced_rf` - 增强随机森林

### 时间序列
- `prophet` - Facebook Prophet（已修复）

### 深度学习 - LSTM 系列
- `lstm` - LSTM 深度学习
- `bilstm` - 双向 LSTM
- `attention_lstm` - 注意力 LSTM
- `multihead_attention` - 多头注意力 LSTM
- `hierarchical_attention` - 分层注意力 LSTM

### 深度学习 - GRU 系列
- `gru` - GRU 深度学习
- `bigru` - 双向 GRU
- `attention_gru` - 注意力 GRU
- `multihead_attention_gru` - 多头注意力 GRU
- `hierarchical_attention_gru` - 分层注意力 GRU

### 深度学习 - 混合模型
- `lstm_transformer` - LSTM-Transformer 混合
- `gru_transformer` - GRU-Transformer 混合
- `transformer_xl` - Transformer-XL

---

## 🐛 故障排除

### Prophet 训练失败
```
错误：数据必须包含 'code' 和 'date' 列
```

**解决**：已在本次更新中修复，请重新运行

### 配置文件损坏
```bash
# 删除配置文件重置
rm ~/.hp_ml_cli_config.json
```

### 模型加载失败
```bash
# 检查模型文件是否存在
ls -lh models/model_*.joblib

# 重新训练模型
make cli
```

---

## 📊 性能提示

1. **使用通达信数据源**：比东方财富更稳定
2. **批量训练模型**：一次性选择多个模型，共享数据
3. **调整预测周期**：horizon=5（短期）vs horizon=20（中期）
4. **合理选择模型数量**：3-5 个模型通常足够

---

## 🔄 更新日志

### v1.1.0 (2024-06-09)
- ✅ 修复 PROPHET 模型训练失败
- ✅ 新增配置记忆功能
- ✅ 优化多模型训练流程
- ✅ 自动保存训练好的模型

### v1.0.0 (初始版本)
- 基础 CLI 交互功能
- 多模型训练支持
- 回测评估功能
- 通达信公式导出
