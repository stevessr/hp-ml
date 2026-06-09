# 快速开始指南

## 🚀 5 分钟上手 HP-ML

### 前置要求
- Python 3.8+
- 已安装依赖：`pip install -r requirements.txt`

---

## 📖 基础流程

### 1️⃣ 训练你的第一个模型

```bash
# 方式1：使用交互式 CLI（推荐）
make cli

# 方式2：直接命令行
python -m hp_ml.multi_model_train --models prophet
```

**训练完成后**：
- ✅ 模型保存在：`models/model_prophet.joblib`
- ✅ 报告保存在：`reports/`
- ✅ 配置自动记忆到：`~/.hp_ml_cli_config.json`

---

### 2️⃣ 导出到通达信

```bash
# 导出所有指数族
python -m hp_ml.export_tdx \
  --model models/model_prophet.joblib \
  --out reports/tdx_formulas \
  --all-families
```

**生成文件**：
```
reports/tdx_formulas/
├── HPML_宽基_ETF_CSI_300.tdx
├── HPML_宽基_ETF_CSI_500.tdx
└── ...
```

---

### 3️⃣ 转换为 TN6 格式（可选）

```bash
# 批量转换为二进制格式
python -m hp_ml.export_tn6 --batch reports/tdx_formulas
```

**生成文件**：
```
reports/tdx_formulas/
├── HPML_宽基_ETF_CSI_300.tn6
├── HPML_宽基_ETF_CSI_500.tn6
└── ...
```

---

### 4️⃣ 在通达信中使用

#### 方式A：导入 TN6 文件（推荐）
1. 打开通达信
2. 功能 → 公式管理器 → 技术指标公式
3. 点击"导入公式"
4. 选择 `.tn6` 文件
5. 确认导入 ✅

#### 方式B：手动创建 TDX 公式
1. 打开 `.tdx` 文件
2. 复制公式内容
3. 在通达信中新建公式
4. 粘贴并保存

---

## 🎯 常用命令

### 训练模型
```bash
# Ridge（精确导出，100%精度）
python -m hp_ml.multi_model_train --models ridge

# Prophet（时间序列，60-70%精度）
python -m hp_ml.multi_model_train --models prophet

# LSTM（深度学习，60-70%精度）
python -m hp_ml.multi_model_train --models lstm

# 多模型对比
python -m hp_ml.multi_model_train --models ridge prophet lstm gru
```

### 导出公式
```bash
# 导出所有指数族
python -m hp_ml.export_tdx --model models/model_xxx.joblib --all-families

# 导出单个指数族
python -m hp_ml.export_tdx --model models/model_xxx.joblib --family-id CSI_300

# 转换为 TN6
python -m hp_ml.export_tn6 --batch reports/tdx_formulas
```

### 测试工具
```bash
# 测试 Prophet 修复
python test_prophet_fix.py

# 测试通达信导出
python test_tdx_export.py prophet
```

---

## 📚 进阶文档

| 文档 | 说明 |
|------|------|
| [CLI_FEATURES.md](./CLI_FEATURES.md) | CLI 新功能详细说明 |
| [TDX_EXPORT_GUIDE.md](./TDX_EXPORT_GUIDE.md) | 通达信导出完整指南 |
| [TN6_EXPORT_GUIDE.md](./TN6_EXPORT_GUIDE.md) | TN6 格式导出指南 |
| [UPDATES.md](./UPDATES.md) | 最新更新内容 |

---

## ❓ 快速问答

### Q: 哪个模型最好？
**A**: 取决于你的需求：
- **Ridge**：精度最高（100%），导出最简单，推荐首选
- **Prophet**：捕捉趋势，适合时间序列预测
- **LSTM/GRU**：捕捉时序依赖，适合复杂模式

### Q: 配置会自动保存吗？
**A**: 是的！所有配置自动保存到 `~/.hp_ml_cli_config.json`，下次使用时自动加载

### Q: TDX 和 TN6 有什么区别？
**A**: 
- **TDX**：文本格式，需要复制粘贴到通达信
- **TN6**：二进制格式，可以直接导入，更方便

### Q: 多层公式太复杂怎么办？
**A**: 使用 Ridge 模型，导出的是单层公式，100% 精度

---

## 🆘 遇到问题？

1. 查看文档：`cat CLI_FEATURES.md`
2. 运行测试：`python test_prophet_fix.py`
3. 重置配置：`rm ~/.hp_ml_cli_config.json`

---

**祝使用愉快！** 🎉
