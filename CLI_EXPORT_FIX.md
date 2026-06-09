# CLI 导出功能修复说明

## 问题描述

在使用 CLI 导出功能时，出现"二次选择模型"的问题：

1. **第一步**：用户选择模型类型（Ridge、HGB、Enhanced RF 等）
2. **第二步**：选择操作类型为"导出到通达信"
3. **第三步**：系统又要求用户从模型文件列表中选择（❌ 冗余！）

用户体验不佳，因为在第一步已经明确了要导出哪个模型。

## 修复方案

### 核心逻辑

修改 `get_export_params()` 函数，增加自动匹配功能：

```python
def get_export_params(selected_models: list[dict] = None) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Args:
        selected_models: 用户第一步选择的模型配置列表，如果提供则自动匹配对应的模型文件
    
    Returns:
        单个模型：返回 dict
        多个模型：返回 list[dict]
    """
```

### 匹配规则

对于每个用户选择的模型类型，系统会自动查找对应的模型文件：

- `enhanced_rf` → `model_enhanced_rf.joblib` ✓
- `ridge` → `model_ridge.pkl` ✓
- `lstm` → `model_lstm.joblib` ✓

匹配条件：`model_key in file_name` 或 `file_name.replace("model_", "") == model_key`

### 执行流程

#### 情况 1：成功匹配
```
用户选择：Enhanced RF
↓
系统输出：
根据您选择的 1 个模型类型，查找对应的模型文件...
✓ ENHANCED_RF → model_enhanced_rf.joblib
↓
直接配置导出参数（指数族、导出模式等）
```

#### 情况 2：未匹配到
```
用户选择：某个新模型
↓
系统输出：
⚠️ NEW_MODEL → 未找到对应的模型文件
未找到匹配的模型文件，请手动选择：
[显示所有可用模型文件供用户选择]
```

### 支持多模型导出

如果用户第一步选择了多个模型（例如：Ridge + HGB + Enhanced RF），系统会：

1. 自动匹配所有模型对应的文件
2. 统一配置导出参数（导出模式、指数族等）
3. 逐个导出每个模型

```python
# 修改后的调用逻辑
elif operation_config['key'] == 'export':
    params = get_export_params(model_configs)  # 传入第一步选择的模型
    if isinstance(params, list):  # 多个模型
        for i, param in enumerate(params, 1):
            print(f"导出模型 {i}/{len(params)}")
            execute_export(param)
    else:  # 单个模型
        execute_export(params)
```

## 测试验证

### 测试脚本
```bash
python test_cli_export_fix.py
```

### 测试结果
```
找到 22 个模型文件
用户选择了 1 个模型类型：ENHANCED_RF
执行自动匹配：
✓ ENHANCED_RF → model_enhanced_rf.joblib
匹配结果：成功匹配 1 个模型
```

## 修改文件

- `hp_ml/cli.py:318-423` - 修改 `get_export_params()` 函数
- `hp_ml/cli.py:749-760` - 修改导出操作调用逻辑
- `hp_ml/cli.py:789-800` - 修改完整流程中的导出逻辑
- `hp_ml/cli.py:692-699` - 修改单个模型完整流程的导出逻辑

## 预期效果

### 修复前
```
? 请选择模型类型： Enhanced RF ✓
? 请选择操作类型： 导出到通达信 ✓
? 选择要导出的模型： [需要再次从列表中选择] ❌
```

### 修复后
```
? 请选择模型类型： Enhanced RF ✓
? 请选择操作类型： 导出到通达信 ✓

根据您选择的 1 个模型类型，查找对应的模型文件...
✓ ENHANCED_RF → model_enhanced_rf.joblib

? 导出模式： 所有指数族（批量导出） ✓
```

用户只需选择一次模型，系统自动找到对应文件，直接进入导出配置！✅

## 兼容性

- ✅ 向后兼容：如果不传入 `selected_models` 参数，仍使用旧逻辑（手动选择）
- ✅ 降级策略：如果自动匹配失败，自动回退到手动选择模式
- ✅ 多模型支持：支持批量导出多个模型
