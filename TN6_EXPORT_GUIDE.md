# TN6 格式导出指南

## 🎯 什么是 TN6 格式？

TN6 是通达信指标公式的二进制文件格式，可以直接在通达信软件中导入使用，相比 .tdx 文本格式更加方便。

## ✨ 功能特点

- ✅ 自动转换 .tdx 文本公式为 .tn6 二进制格式
- ✅ 支持批量转换整个目录
- ✅ 使用 GBK 编码，完美兼容通达信
- ✅ 包含公式元数据（名称、类型、版本）
- ✅ 自动生成校验和

## 📦 使用方法

### 方式1：单个文件转换

```bash
# 转换单个 .tdx 文件为 .tn6
python -m hp_ml.export_tn6 reports/tdx_formulas/prophet_CSI_300.tdx

# 指定输出路径
python -m hp_ml.export_tn6 \
  reports/tdx_formulas/prophet_CSI_300.tdx \
  -o output/prophet_CSI_300.tn6
```

### 方式2：批量转换目录

```bash
# 批量转换目录下所有 .tdx 文件
python -m hp_ml.export_tn6 --batch reports/tdx_formulas

# 指定输出目录
python -m hp_ml.export_tn6 \
  --batch reports/tdx_formulas \
  -o output/tn6_files
```

### 方式3：指定公式类型

```bash
# 技术指标公式（默认）
python -m hp_ml.export_tn6 formula.tdx -t 1

# 条件选股公式
python -m hp_ml.export_tn6 formula.tdx -t 2

# 专家系统公式
python -m hp_ml.export_tn6 formula.tdx -t 3
```

## 🚀 完整工作流

### 步骤1：训练模型
```bash
python -m hp_ml.multi_model_train --models prophet
```

### 步骤2：导出为 .tdx 格式
```bash
python -m hp_ml.export_tdx \
  --model models/model_prophet.joblib \
  --out reports/tdx_formulas \
  --all-families
```

### 步骤3：转换为 .tn6 格式
```bash
python -m hp_ml.export_tn6 \
  --batch reports/tdx_formulas
```

### 步骤4：在通达信中导入
1. 打开通达信软件
2. 功能 → 公式管理器 → 技术指标公式
3. 点击"导入公式"按钮
4. 选择 .tn6 文件
5. 确认导入

## 📊 文件格式对比

| 格式 | 类型 | 编码 | 导入方式 | 优点 | 缺点 |
|------|------|------|----------|------|------|
| .tdx | 文本 | UTF-8/GBK | 复制粘贴 | 可读性好 | 需要手动创建 |
| .tn6 | 二进制 | GBK | 直接导入 | 一键导入 | 不可读 |
| .tne | 二进制 | GBK | 批量导入 | 多个公式 | 格式复杂 |

## 💡 Python 脚本使用

```python
from pathlib import Path
from hp_ml.export_tn6 import TN6Exporter, convert_all_tdx_to_tn6

# 方式1：使用导出器
exporter = TN6Exporter()

# 转换单个文件
exporter.export_tdx_formula_as_tn6(
    tdx_file='reports/tdx_formulas/prophet_CSI_300.tdx',
    output_file='output/prophet_CSI_300.tn6',
    formula_type=1  # 1=技术指标
)

# 方式2：批量转换
convert_all_tdx_to_tn6(
    input_dir='reports/tdx_formulas',
    output_dir='output/tn6_files'
)

# 方式3：直接从代码生成
formula_code = """
FR1:=CLOSE/REF(CLOSE,1)-1;
HPMLSCORE:FR1*100,COLORWHITE;
"""

tn6_data = exporter.export_formula(
    formula_name='测试公式',
    formula_code=formula_code,
    formula_type=1,
    output_path='output/test.tn6'
)
```

## 🔧 TN6 文件结构

```
TN6 文件结构：
┌─────────────────────────────┐
│ 文件头标识: "TN6\x00" (4B)  │
├─────────────────────────────┤
│ 版本号: int (4B)             │
├─────────────────────────────┤
│ 公式类型: int (4B)           │
│   1 = 技术指标               │
│   2 = 条件选股               │
│   3 = 专家系统               │
├─────────────────────────────┤
│ 公式名称长度: int (4B)       │
├─────────────────────────────┤
│ 公式名称: GBK编码字符串      │
├─────────────────────────────┤
│ 公式代码长度: int (4B)       │
├─────────────────────────────┤
│ 公式代码: GBK编码字符串      │
├─────────────────────────────┤
│ 校验和: int (4B)             │
└─────────────────────────────┘
```

## ❓ 常见问题

### Q1: TN6 和 TDX 有什么区别？
**A**: 
- **TDX**：文本格式，可以用记事本打开，需要复制粘贴到通达信
- **TN6**：二进制格式，可以直接在通达信中点击"导入"按钮一键导入

### Q2: 通达信提示"文件格式不正确"？
**A**: 可能原因：
1. 通达信版本太旧，不支持此格式
2. 文件编码问题（已使用 GBK，应该兼容）
3. 可以尝试使用 .tdx 文本格式作为替代方案

### Q3: 如何验证 TN6 文件是否正确？
**A**: 
```bash
# 查看文件头（应该看到 "TN6" 标识）
hexdump -C file.tn6 | head -5

# 查看文件大小（应该 > 100 字节）
ls -lh file.tn6
```

### Q4: 能否直接生成 TN6，跳过 TDX？
**A**: 可以！使用 Python API：
```python
from hp_ml.export_tn6 import TN6Exporter

exporter = TN6Exporter()
exporter.export_formula(
    formula_name='我的公式',
    formula_code='公式代码...',
    output_path='output.tn6'
)
```

### Q5: 多层公式如何导出为 TN6？
**A**: 每一层单独导出为一个 TN6 文件：
```bash
# 先导出为 .tdx
python -m hp_ml.export_tdx --model models/model_prophet.joblib --all-families

# 再批量转换为 .tn6
python -m hp_ml.export_tn6 --batch reports/tdx_formulas
```

## 📝 实战案例

### 案例1：导出 Prophet 模型为 TN6

```bash
# 1. 训练 Prophet 模型
python -m hp_ml.multi_model_train --models prophet

# 2. 导出为 TDX 文本格式
python -m hp_ml.export_tdx \
  --model models/model_prophet.joblib \
  --out reports/tdx_formulas \
  --all-families

# 3. 批量转换为 TN6 二进制格式
python -m hp_ml.export_tn6 --batch reports/tdx_formulas

# 4. 查看生成的文件
ls -lh reports/tdx_formulas/*.tn6
```

### 案例2：只导出中证300的 TN6

```bash
# 1. 导出中证300的 TDX
python -m hp_ml.export_tdx \
  --model models/model_prophet.joblib \
  --family-id CSI_300 \
  --out reports/prophet_CSI_300.tdx

# 2. 转换为 TN6
python -m hp_ml.export_tn6 \
  reports/prophet_CSI_300.tdx \
  -o reports/prophet_CSI_300.tn6
```

### 案例3：导出为条件选股公式

```bash
# 假设你有一个条件选股公式文本
python -m hp_ml.export_tn6 \
  my_stock_selection.tdx \
  -o my_stock_selection.tn6 \
  -t 2  # 类型2 = 条件选股
```

## 🎨 高级用法

### 自定义导出器

```python
from hp_ml.export_tn6 import TN6Exporter

class CustomTN6Exporter(TN6Exporter):
    """自定义 TN6 导出器"""
    
    def __init__(self):
        super().__init__()
        self.version = 2  # 使用版本2
        
    def add_metadata(self, formula_name, formula_code):
        """添加额外的元数据"""
        metadata = f"{{创建时间: {datetime.now()}}}\n"
        return metadata + formula_code

# 使用自定义导出器
exporter = CustomTN6Exporter()
exporter.export_tdx_formula_as_tn6('input.tdx', 'output.tn6')
```

## 📖 相关文档

- [TDX 导出指南](./TDX_EXPORT_GUIDE.md)
- [CLI 功能说明](./CLI_FEATURES.md)
- [通达信公式教程](https://github.com/wgwang/rstock-docs)

## ⚠️ 免责声明

此 TN6 格式实现是基于社区经验和逆向分析，非官方格式规范。
如果通达信软件无法识别生成的 TN6 文件，请使用 .tdx 文本格式作为替代方案。

---

**提示**：如果你在使用过程中遇到问题，欢迎提交 Issue 或 Pull Request！
