# 模型导出到通达信指南

## 📊 支持的模型类型

### ✅ 直接导出（精确公式）
**岭回归 (Ridge)**
- 导出方式：线性公式
- 精度：100%（完全等价）
- 公式层数：1 层
- 推荐使用场景：追求精度和稳定性

**使用方法**：
```bash
# 训练 Ridge 模型
python -m hp_ml.multi_model_train --models ridge

# 导出所有指数族
python -m hp_ml.export_tdx \
  --model models/model_ridge.joblib \
  --out reports/tdx_formulas \
  --all-families

# 或导出单个指数族
python -m hp_ml.export_tdx \
  --model models/model_ridge.joblib \
  --family-id CSI_300 \
  --out reports/ridge_CSI_300.tdx
```

---

### 🔄 知识蒸馏导出（多层近似）
**Prophet / LSTM / GRU / 深度学习模型**
- 导出方式：多层神经网络近似
- 精度：60-70%
- 公式层数：4 层（特征层 + 2个隐藏层 + 输出层）
- 原理：使用简单神经网络模拟复杂模型的行为

**支持的模型**：
- Prophet 时间序列
- LSTM / BiLSTM / Attention LSTM
- GRU / BiGRU / Attention GRU
- LSTM-Transformer / GRU-Transformer
- 所有深度学习模型

**使用方法**：
```bash
# 1. 训练 Prophet 模型
python -m hp_ml.multi_model_train --models prophet

# 2. 导出到通达信（自动使用知识蒸馏）
python -m hp_ml.export_tdx \
  --model models/model_prophet.joblib \
  --out reports/tdx_formulas \
  --all-families
```

**生成的文件**：
```
reports/tdx_formulas/
├── prophet_CSI_300.tdx      # 中证300
├── prophet_CSI_500.tdx      # 中证500
├── prophet_CSI_1000.tdx     # 中证1000
├── prophet_CSI_2000.tdx     # 中证2000
└── ...
```

---

### ⚠️ 不支持的模型
**随机森林 (RF) / 梯度提升树 (HGB) / XGBoost**
- 原因：树模型无法用数学公式精确表示
- 替代方案：使用知识蒸馏（精度会降低）

---

## 🎯 通达信公式导入步骤

### 方式1：单层公式（Ridge 模型）
1. 打开通达信软件
2. 功能 → 公式管理器 → 技术指标公式 → 新建
3. 复制 `.tdx` 文件内容到公式编辑器
4. 点击"确定"保存
5. 在K线图上调用公式名称（如 "HPML 宽基 ETF_CSI_300"）

### 方式2：多层公式（Prophet/LSTM 模型）
**必须按顺序创建 4 个公式**：

#### 第1步：创建特征层
```
公式名称：HPML 宽基 ETF_CSI_300_特征层
```
- 复制 `.tdx` 文件中 "公式1: 特征层" 部分
- 计算所有基础技术指标

#### 第2步：创建隐藏层1
```
公式名称：HPML 宽基 ETF_CSI_300_隐藏层1
```
- 复制 "公式2: 隐藏层1" 部分
- 引用特征层的输出变量（如 FR1, FSD5 等）

#### 第3步：创建隐藏层2
```
公式名称：HPML 宽基 ETF_CSI_300_隐藏层2
```
- 复制 "公式3: 隐藏层2" 部分
- 引用隐藏层1的输出变量（如 H2U0, H2U1 等）

#### 第4步：创建输出层
```
公式名称：HPML 宽基 ETF_CSI_300_输出层
```
- 复制 "公式4: 输出层" 部分
- 引用隐藏层2的输出变量（如 H3U0, H3U1 等）
- 这是最终使用的公式

**⚠️ 注意事项**：
- 公式必须严格按顺序创建
- 如果修改某一层，需要重新创建后续所有层
- 建议先用 Ridge 单层公式测试，确认流程正常后再使用多层公式

---

## 📈 使用公式进行选股

### 在通达信中调用公式
1. 打开任意 ETF 的 K 线图
2. 输入公式名称，如：`HPML 宽基 ETF_CSI_300_输出层`
3. 观察指标值：
   - `HPMLSCORE`：预测未来 5 日收益率
   - `HPMLBUY`：买入信号（1=买入，0=不买入）

### 选股策略示例
```
# 条件选股公式
SCORE:=HPML宽基ETF_CSI_300_输出层.HPMLSCORE;
BUY:=SCORE>0.01;  {预测收益大于1%}
BUY;
```

### 排名选股（推荐）
1. 工具 → 综合排名
2. 添加自定义指标：`HPML宽基ETF_CSI_300_输出层.HPMLSCORE`
3. 按 HPMLSCORE 降序排列
4. 选择前 3-5 只 ETF 买入

---

## 🔧 高级配置

### 调整买入阈值
```bash
python -m hp_ml.export_tdx \
  --model models/model_prophet.joblib \
  --signal-threshold 0.01 \
  --all-families
```
- `--signal-threshold 0.01`：只有预测收益 >1% 时才显示买入信号

### 简化公式（剔除小系数）
```bash
python -m hp_ml.export_tdx \
  --model models/model_ridge.joblib \
  --min-abs-coef 0.01 \
  --all-families
```
- `--min-abs-coef 0.01`：剔除绝对值 <0.01 的特征系数
- 可以让公式更简洁，略微降低精度

### 指定输出编码
```bash
python -m hp_ml.export_tdx \
  --model models/model_ridge.joblib \
  --encoding gbk \
  --all-families
```
- `--encoding gbk`：使用 GBK 编码（老版通达信）
- 默认 UTF-8（新版通达信推荐）

---

## 📊 精度对比

| 模型类型 | 导出方式 | 精度 | 公式复杂度 | 推荐场景 |
|---------|---------|------|-----------|---------|
| Ridge | 线性公式 | 100% | 简单（1层） | 稳定性优先 |
| Prophet | 多层近似 | 60-70% | 中等（4层） | 捕捉趋势 |
| LSTM | 多层近似 | 60-70% | 中等（4层） | 捕捉时序依赖 |
| HGB/RF | 不支持 | N/A | N/A | 仅用于 Python 回测 |

---

## 🎨 完整工作流示例

### 场景：训练 Prophet 模型并导出到通达信

```bash
# 1. 训练模型
python -m hp_ml.multi_model_train --models prophet

# 2. 导出所有指数族的公式
python -m hp_ml.export_tdx \
  --model models/model_prophet.joblib \
  --out reports/tdx_formulas \
  --all-families

# 3. 查看生成的文件
ls reports/tdx_formulas/
# prophet_CSI_300.tdx
# prophet_CSI_500.tdx
# prophet_CSI_1000.tdx
# ...

# 4. 在通达信中按顺序导入 4 个公式
# 5. 使用输出层公式进行选股
```

---

## ❓ 常见问题

### Q1: 多层公式太复杂，能用单层吗？
**A**: 可以训练 Ridge 模型，导出的是单层线性公式，精度 100%：
```bash
python -m hp_ml.multi_model_train --models ridge
python -m hp_ml.export_tdx --model models/model_ridge.joblib --all-families
```

### Q2: 为什么 Prophet 精度只有 60-70%？
**A**: Prophet 是时间序列模型，内部使用了傅里叶变换、趋势分解等复杂算法，无法用简单公式精确表示。我们用多层神经网络"模拟"它的行为，因此有精度损失。

### Q3: 能导出 LSTM 模型吗？
**A**: 可以！LSTM 也是通过多层近似导出：
```bash
python -m hp_ml.multi_model_train --models lstm
python -m hp_ml.export_tdx --model models/model_lstm.joblib --all-families
```

### Q4: 公式在通达信中提示"变量未定义"？
**A**: 确保按顺序创建所有层：
1. 特征层（定义 FR1, FSD5 等）
2. 隐藏层1（引用特征层变量）
3. 隐藏层2（引用隐藏层1变量）
4. 输出层（引用隐藏层2变量）

### Q5: 不同指数族的公式有什么区别？
**A**: 主要区别在 `family_XXX` 变量：
- `FfamilyCSI300:=1` - 中证300 公式
- `FfamilyCSI500:=1` - 中证500 公式
- 其他指数族设为 0

---

## 📝 总结

- **追求精度**：使用 Ridge 模型（100% 精度，单层公式）
- **捕捉趋势**：使用 Prophet 模型（60-70% 精度，多层公式）
- **时序依赖**：使用 LSTM/GRU（60-70% 精度，多层公式）
- **快速验证**：先用 Ridge，确认流程后再尝试复杂模型

**推荐组合策略**：
1. 用 Prophet/LSTM 训练获取高质量预测
2. 在 Python 中进行回测验证
3. 将最优模型导出到通达信
4. 在通达信中进行实盘跟踪
