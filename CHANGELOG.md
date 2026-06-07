# 更新日志 (CHANGELOG)

所有值得注意的项目变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [2.0.0] - 2026-06-07

### 新增 🎉

#### 核心模块
- **多模型训练框架** (`hp_ml/multi_model_train.py`)
  - 支持同时训练和对比多个模型
  - 自动生成对比报告和可视化图表
  - 统一的模型评估和回测流程

- **扩展模型库** (`hp_ml/models_extended.py`)
  - `ProphetWrapper`: Facebook Prophet 时间序列模型
  - `LSTMModel`: LSTM 深度学习模型
  - `EnhancedRandomForest`: 增强随机森林（含特征重要性）
  - `make_extended_model()`: 统一模型创建接口

- **数据划分管道** (`hp_ml/data_pipeline.py`)
  - `time_series_split()`: 训练/验证/测试标准划分
  - `expanding_window_cv()`: 扩展窗口交叉验证
  - `prepare_lstm_sequences()`: LSTM 序列数据准备

- **回测框架** (`hp_ml/backtest.py`)
  - `backtest_strategy()`: 单策略回测
  - `compare_models_backtest()`: 多模型对比回测
  - `BacktestMetrics`: 标准化回测指标数据类
  - 完整的风险指标计算（夏普、索提诺、卡玛、最大回撤等）

- **可视化系统** (`hp_ml/visualization.py`)
  - `plot_model_comparison_metrics()`: 模型指标对比图
  - `plot_equity_curves()`: 多模型权益曲线对比
  - `plot_feature_importance()`: 特征重要性可视化
  - `plot_prediction_distribution()`: 预测分布对比图
  - `generate_markdown_report()`: 自动生成 Markdown 报告
  - `create_all_visualizations()`: 一键生成所有图表

#### 文档系统
- **快速入门指南** (`docs/QUICKSTART.md`)
  - 5 分钟上手教程
  - 典型使用场景示例
  - 常见问题解答

- **多模型详细文档** (`docs/MULTI_MODEL.md`)
  - 各模型详细介绍和使用方法
  - 完整 API 参考
  - 最佳实践建议

- **算法公式文档** (`docs/FORMULAS.md`)
  - 系统架构图
  - 特征工程公式推导
  - 模型算法数学原理
  - 回测指标计算公式

- **功能总结文档** (`docs/SUMMARY.md`)
  - 新增模块概览
  - 模型对比表格
  - 输出文件结构说明

- **文档索引** (`docs/README.md`)
  - 完整文档导航
  - 推荐阅读路径
  - 快速链接

#### 测试与示例
- **单元测试** 
  - `tests/test_data_pipeline.py`: 数据管道测试
  - `tests/test_backtest.py`: 回测框架测试

- **示例代码**
  - `examples/complete_workflow.py`: 完整工作流示例
  - `scripts/test_multi_model.py`: 快速测试脚本

#### 构建工具
- **Makefile 新增命令**
  - `make train-multi`: 多模型训练
  - `make test-multi`: 快速测试

### 改进 ⚡

- **依赖管理**
  - 新增 `prophet>=1.1` 支持时间序列模型
  - 新增 `tensorflow>=2.15` 和 `keras>=3.0` 支持深度学习
  - 所有依赖更新到最新稳定版本

- **README 更新**
  - 新增多模型功能说明
  - 更新快速开始示例
  - 添加多模型训练命令

### 向后兼容 ✅

- 原有 `hp_ml.train` 完全保留，功能不变
- 原有 `hp_ml.lite_train` 完全保留，功能不变
- 所有现有 API 保持稳定
- 原有输出文件格式不变

## [1.x.x] - 历史版本

### 特性
- 单模型训练（HGB、Ridge、RandomForest）
- ETF 自动发现和筛选
- 基础特征工程
- 时间序列交叉验证
- 通达信公式导出
- 自动参数调优
- 基础回测功能

---

## 版本说明

### 语义化版本规则

- **主版本号 (MAJOR)**: 不兼容的 API 变更
- **次版本号 (MINOR)**: 向后兼容的功能新增
- **修订号 (PATCH)**: 向后兼容的问题修正

### 变更类型

- `新增`: 新功能
- `改进`: 现有功能的改进
- `修复`: Bug 修复
- `变更`: 不向后兼容的变更
- `废弃`: 即将移除的功能
- `移除`: 已移除的功能
- `安全`: 安全性相关的修复

---

## 未发布 [Unreleased]

### 计划新增
- [ ] 模型集成学习（Stacking/Blending）
- [ ] 更多技术指标特征
- [ ] 实时预测 API
- [ ] 交互式仪表板
- [ ] 风险管理模块（仓位控制、止损）

### 计划改进
- [ ] 性能优化（并行计算、缓存策略）
- [ ] 更多可视化图表类型
- [ ] 支持更多数据源
- [ ] 增强错误处理和日志

---

## 贡献指南

更新此文件时，请遵循以下格式：

```markdown
## [版本号] - YYYY-MM-DD

### 新增
- **功能名称**: 详细描述

### 改进
- **改进内容**: 详细描述

### 修复
- **Bug 描述**: 修复方法

### 变更
- **API 变更**: 影响说明
```

---

## 链接

- [项目仓库](https://github.com/your-repo/hp-ml)
- [问题追踪](https://github.com/your-repo/hp-ml/issues)
- [文档主页](docs/README.md)
