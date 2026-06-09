# hp-ml：中证宽基指数 ETF 自动挖掘、拉取数据与训练

这个项目把”自动搜索 → 挖掘候选 ETF → 拉取历史行情 → 特征工程 → 训练模型 → 输出排序报告”串成一条可运行流水线，重点覆盖中证/沪深宽基指数 ETF（沪深 300、中证 500、中证 1000、中证 2000、中证 800、中证 A500、中证 A50、中证 A100，以及中证 200/700/全指/流通/A 股等有 ETF 时自动纳入的宽基族）。

**🆕 新增功能**：
- **🎯 交互式 CLI**：可视化菜单系统，一键选择模型类型、训练、导出、回测、对比（推荐使用）
- **🚀 GRU Attention 变种**：新增 5 个 GRU 深度学习模型（双向、注意力、多头注意力、分层注意力、GRU-Transformer），训练速度比 LSTM 快 20-30%，参数量少 25-30%
- **通达信数据源支持**：可选择使用通达信服务器直接拉取数据，作为东方财富API的可靠备选方案
- **多模型训练与对比**：支持Prophet、LSTM、GRU、增强随机森林等，完整的数据划分管道，统一回测框架，自动生成可视化报告。详见 [多模型文档](docs/MULTI_MODEL.md)

> 说明：输出仅用于量化研究和模型验证，不构成投资建议。

## 快速开始

### 🎯 方式一：交互式 CLI（推荐，新手友好）

```bash
# 依赖安装
make setup

# 启动交互式 CLI
make cli
# 或
python -m hp_ml.cli
# 或
python -m hp_ml
```

**CLI 功能**：
- ✨ 可视化菜单选择模型类型（Ridge、HGB、RF、LSTM、GRU、Transformer 等 18+ 种模型）
- 🚀 新增 GRU Attention 变种：双向 GRU、注意力 GRU、多头注意力 GRU、分层注意力 GRU、GRU-Transformer
- 🎯 选择操作：训练、导出通达信、回测、模型对比、完整流程
- ⚙️ 交互式配置参数（数据源、日期范围、预测周期等）
- 📊 自动生成报告和图表
- 🚀 支持一键完整流程：训练→回测→导出→对比

### 📝 方式二：命令行模式（高级用户）

```bash
# 依赖安装
make setup

# 只发现候选 ETF
make discover

# 拉取 2018 至今数据并训练 5 日前瞻收益模型（默认使用东方财富数据源）
make train

# 🆕 使用通达信数据源训练（更稳定，直连行情服务器）
python -m hp_ml.train --data-source tdx

# 🆕 多模型训练与对比（Ridge, HGB, 随机森林）
make train-multi

# 更快的短历史 smoke run
make quick

# 将 ETF 看涨信号下钻到相关成分股，补成交额/换手/股东占比
make stock-deep-dive

# 滚动模型自动调优，搜索到策略累计收益超过等权 baseline 后输出报告
make auto-beat-baseline

# 将轻量 Ridge 模型导出为通达信公式文本
make export-tdx
```

也可以直接运行：

```bash
.venv/bin/python -m hp_ml.train \
  --start 20180101 \
  --max-etfs-per-index 3 \
  --horizon 5 \
  --test-days 252
```

## 流水线做了什么

1. **自动搜索 ETF 池**：扫描东方财富 ETF 行情/基金代码列表，不靠手写固定清单。
2. **聚焦中证宽基**：用名称模式识别沪深 300/中证 500/中证 1000/中证 2000/中证 800/中证 A500/中证 A50/中证 A100，以及中证 200/700/全指/流通/A 股等宽基族。
3. **剔除非纯宽基**：默认排除行业、主题、风格、跨境、债券、货币和指数增强产品；可用 `--include-enhanced` / `--include-style` 放开。
4. **拉取历史行情**：对筛出的 ETF 拉取前复权日线 K 线，并缓存到 `data/raw/history/`。
5. **设计训练样本**：生成动量、波动率、均线乖离、回撤、成交额冲击、换手率、振幅、月份周期、指数族哑变量等特征。
6. **监督学习训练**：默认训练 `HistGradientBoostingRegressor`，预测未来 `--horizon` 个交易日收益。
7. **时间切分验证**：按交易日切分训练/测试，输出 MAE、RMSE、方向准确率、按日 Spearman IC、Top1 相对全池收益等指标。

## 关键输出

- `reports/latest_candidates.csv`：自动发现和排序后的中证宽基 ETF 候选池。
- `reports/downloaded_csi_broad_etfs.csv`：`scripts/download.py` 全量下载后的 ETF 历史行情覆盖汇总。
- `data/raw/etf_spot_YYYYMMDD.csv`：全 ETF 现货行情快照。
- `data/raw/history/*.csv`：候选 ETF 历史 K 线缓存。
- `data/processed/training_panel.csv`：完整模型训练面板。
- `data/processed/historical_dataset_lite.csv`：有未来收益标签的历史数据集。
- `data/processed/train_dataset_lite.csv` / `test_dataset_lite.csv`：按时间切分的训练集与测试/回测集。
- `data/processed/future_dataset_lite.csv`：未来收益尚未发生的未标注推理数据集。
- `models/csi_broad_etf_model.joblib`：模型、特征列、目标、候选池和指标的 joblib 包。
- `reports/latest_predictions.csv`：最新日期每只候选 ETF 的模型预测排序。
- `reports/buy_signals_lite.csv`：按购买策略生成的当前买入/空仓信号。
- `reports/model_parameter_tuning_lite.csv` / `strategy_parameter_tuning_lite.csv`：自动参数调优明细。
- `reports/ml_auto_tune_until_baseline.csv`：自动调优后跑赢等权 baseline 的逐日策略/基准曲线。
- `reports/ml_auto_tune_trials.csv` / `ml_auto_tune_summary.json`：每组模型/策略参数的搜索结果与最佳配置摘要。
- `reports/related_stocks_lite.csv`：每个宽基指数族对应的更多相关股票/指数成分股。
- `reports/bullish_stock_deep_dive.csv`：从最新 ETF 信号下钻出的看涨相关股票，含成交额、换手、近期动量、主力净流入、前十大流通/总股东占比和机制标签。
- `reports/bullish_stock_deep_dive_shareholders.csv`：看涨股票逐个十大流通股东/十大股东明细。
- `reports/bullish_stock_deep_dive_holder_history.csv`：多报告期十大流通股东/十大股东历史明细，含是否个人股东、机构股东、基金/社保/QFII 类股东标记。
- `reports/bullish_stock_deep_dive_individual_holder_changes.csv`：个人股东逐期变迁，追踪新进、退出、增持、减持和排名变化。
- `reports/bullish_stock_deep_dive.md` / `.json`：自动深挖摘要、异常记录和数据源说明。
- `reports/shareholder_composition/`：每家看涨公司的股东成分 Markdown 报告和索引，报告内用饼图展示十大流通股东/十大股东占比，适合继续做人工复核或交付材料。
- `reports/long_history_coverage_lite.csv`：长起点采集后每只 ETF 的实际历史覆盖、全期收益和最大回撤。
- `reports/multi_scale_etf_metrics_lite.csv`：ETF × 持有周期的平均收益、胜率、波动、类 Sharpe、最佳/最差收益。
- `reports/multi_scale_family_metrics_lite.csv`：指数族/全池 × 持有周期的跨时间尺度聚合统计。
- `reports/multi_scale_period_metrics_lite.csv`：年份 × 指数族 × 持有周期的阶段统计。
- `reports/strategy_backtest_daily_lite.csv`：每日滚动购买策略回测结果。
- `reports/strategy_backtest_trades_lite.csv`：逐笔买入信号、实际未来收益、扣费后收益和胜负。
- `reports/training_metrics.json`：机器可读训练指标。
- `reports/training_summary.md`：中文训练摘要报告。
- `reports/charts/*.svg`：预测排序、指数族覆盖、验证指标、holdout 累计曲线、自动调参和相关股票暴露图。
- `reports/tdx_formulas/*.tdx`：按指数族导出的通达信公式文本，可复制到通达信公式编辑器。

## 常用参数

```bash
.venv/bin/python -m hp_ml.lite_train --help
```

- `--start 20180101`：历史数据开始日期。
- `--end YYYYMMDD`：历史数据结束日期，默认今天。
- `--horizon 5`：预测未来 N 个交易日收益。
- `--max-etfs-per-index 3`：每个宽基指数族按成交额保留前 N 只 ETF。
- `--min-amount 10000000`：按当日成交额过滤低流动性产品。
- `--model hgb|rf|ridge`：完整版 `hp_ml.train` 可选择梯度提升、随机森林或岭回归；轻量版固定为岭回归。
- `--auto-tune` / `--no-auto-tune`：启用或关闭自动参数调优；默认启用。
- `--tune-l2-grid 0.3,1,3,10,30`：岭回归正则强度候选。
- `--tune-top-k-grid 1,2,3,4,5`：购买策略 TopK 候选。
- `--tune-min-pred-grid -0.005,0,0.005,0.01,0.02`：买入阈值候选。
- `--strategy-top-k 3`：未自动调参时，每个决策日买入预测排序前 K 只。
- `--strategy-min-pred 0`：未自动调参时，买入所需最低预测未来收益。
- `--round-trip-cost-bps 10`：回测中每次完整买卖的成本/滑点，单位 bps。
- `--related-stocks-per-index 30`：每个指数族拉取多少只相关股票/成分股。
- `--scale-horizons 5,20,60,120,250`：长历史跨时间尺度分析使用的持有周期。
- `--force`：忽略缓存，重新拉取行情。

全量下载所有发现到的中证宽基相关 ETF 历史日线：

```bash
.venv/bin/python scripts/download.py \
  --start 20050101 \
  --end 20260603 \
  --include-enhanced
```

该脚本会把日线 CSV 写到 `data/raw/history/`，并把覆盖范围汇总到 `reports/downloaded_csi_broad_etfs.csv`。默认不包含红利、低波、成长、行业、跨境等风格/主题产品；需要时可另加 `--include-style`。


### 长历史和跨时间尺度分析

要采集更多、更久的中证宽基 ETF 数据并输出跨时间尺度分析，可以把起点提前到早期 ETF 上市前，并提高每个指数族保留数量：

```bash
.venv/bin/python -m hp_ml.lite_train \
  --start 20050101 \
  --max-etfs-per-index 5 \
  --horizon 5 \
  --test-days 504 \
  --scale-horizons 5,20,60,120,250 \
  --related-stocks-per-index 30
```

流水线会按实际上市日期保留可用历史，输出：

- `reports/long_history_coverage_lite.csv`
- `reports/multi_scale_etf_metrics_lite.csv`
- `reports/multi_scale_family_metrics_lite.csv`
- `reports/multi_scale_period_metrics_lite.csv`
- `reports/charts/multi_scale_horizon_return_lite.svg`
- `reports/charts/multi_scale_win_rate_lite.svg`
- `reports/charts/long_history_coverage_lite.svg`
- `reports/charts/multi_scale_period_return_lite.svg`



## 自动参数调优

轻量流水线默认开启自动调优：

1. 在训练集内部再切出靠后的验证期。
2. 网格搜索 Ridge `l2`，按方向准确率、IC、Top1 相对收益和 RMSE 的综合分数选择模型参数。
3. 在验证期搜索购买策略的 `TopK` 和最低预测收益阈值。
4. 用最佳参数在完整训练集上重训，再在测试/回测集上评估。

调优明细会写入：

- `reports/model_parameter_tuning_lite.csv`
- `reports/strategy_parameter_tuning_lite.csv`
- `reports/charts/model_l2_tuning_lite.svg`
- `reports/charts/strategy_parameter_tuning_lite.svg`

### 自动调优直到超越 baseline

如果目标是让模型直接根据历史数据搜索参数，直到策略累计收益超过等权 ETF baseline，可以运行：

```bash
.venv/bin/python -m hp_ml.auto_tune_baseline \
  --data-dir data/topic2_broad_base \
  --sentiment-path data/alternative_data/google_trends_sentiment.csv \
  --start 2024-11-01 \
  --end 2026-06-03 \
  --max-model-configs 1
```

该命令会滚动训练 HGB 方向分类器，自动搜索 TopK、概率阈值、舆情熔断、趋势过滤和 EMA 调仓参数；当最佳配置的累计收益超过等权 baseline 时，写出逐日曲线、trial 明细和 JSON 摘要。

## 购买策略与回测

轻量流水线默认使用一个可解释的 Top-K 购买策略：

1. 每个决策日按模型预测未来收益排序。
2. 买入预测值不低于 `--strategy-min-pred` 的前 `--strategy-top-k` 只 ETF。
3. 等权配置，持有 `--horizon` 个交易日。
4. 回测收益扣除 `--round-trip-cost-bps` 指定的完整买卖往返成本。
5. 输出交易胜率、调仓胜率、跑赢基准概率、累计收益、最大回撤和类 Sharpe。

注意：回测使用滚动前瞻收益标签，窗口会重叠，适合研究排序信号，不等同真实账户流水。


## 相关股票/成分股扩展

为了让 ETF 研究不仅停留在基金层面，流水线会按发现到的宽基指数族拉取更多相关股票/指数成分股：

- 沪深 300、中证 500、中证 1000、中证 2000、中证 800、中证 A500、中证 A50、中证 A100 等。
- 有权重字段的指数按权重排序；无权重字段时按自由流通市值排序。
- 输出股票代码、名称、行业、地区、权重、自由流通市值、PE、涨跌幅等字段。

输出文件：

- `reports/related_stocks_lite.csv`
- `reports/charts/related_stock_top_exposure_lite.svg`

### ETF 信号下钻到看涨股票

在已经生成 `reports/latest_predictions_lite.csv` 与 `reports/related_stocks_lite.csv` 后，可以继续深挖 ETF 看涨信号背后的股票交易和股东结构：

```bash
.venv/bin/python -m hp_ml.stock_deep_dive \
  --top-etfs 8 \
  --stocks-per-family 25 \
  --max-candidates 80 \
  --top-stocks 30
```

该命令会：

1. 读取最新 ETF 预测排序，把非负预测和 TopN ETF 作为信号入口。
2. 映射到对应指数族的成分股/相关股票，并按权重、自由流通市值和 ETF 信号排序候选池。
3. 拉取 A 股现货行情与日 K，补充成交额、换手率、近 5/20/60 日收益、20 日成交额放大倍数、主力净流入等交易证据。
4. 对自动判定的看涨股票拉取东方财富股东分析，汇总前十大流通股东占比、前十大股东占比、机构/基金类股东占比和第一大股东。
5. 追加历史股东变迁分析，默认保留最近 8 个报告期，重点追踪个人股东的新进、退出、增持、减持、排名变化和个人股东占比趋势。
6. 自动给每家看涨公司制作股东成分报告，包含公司交易线索、股东结构快照、十大流通股东/十大股东明细、股东占比饼图、个人股东历史变迁和解读要点。
7. 输出 `reports/bullish_stock_deep_dive.csv`、`reports/bullish_stock_deep_dive_shareholders.csv`、`reports/bullish_stock_deep_dive_holder_history.csv`、`reports/bullish_stock_deep_dive_individual_holder_changes.csv`、`reports/bullish_stock_deep_dive.md`、`reports/shareholder_composition/index.md` 以及成交额/股东集中度/行业分布 SVG 图。

常用参数：

- `--min-etf-pred 0`：ETF 预测收益不低于该阈值会进入下钻；同时默认也保留 `--top-etfs` 只最高排序 ETF，避免风险收缩期候选过少。
- `--refresh-related`：忽略现有 `related_stocks_lite.csv`，按当前 ETF 家族重新拉取成分股。
- `--force`：忽略当天行情、K 线和股东缓存，重新访问外部数据源。
- `--company-report-dir reports/shareholder_composition`：指定每家公司股东成分报告输出目录；如只要 CSV，可用 `--no-company-reports` 关闭。
- `--holder-history-periods 8`：历史股东变迁保留的报告期数量；如只要最新股东快照，可用 `--no-holder-history` 关闭。

## 数据源

### 东方财富（默认）
- ETF 列表与行情：优先使用东方财富 ETF 行情 `push2.eastmoney.com/api/qt/clist/get`；不可用时回退到 `fund.eastmoney.com/js/fundcode_search.js`。
- ETF 历史 K 线：优先使用东方财富 K 线 `push2his.eastmoney.com/api/qt/stock/kline/get`；不可用时回退到腾讯 K 线 `web.ifzq.gtimg.cn/appstock/app/fqkline/get`。

### 🆕 通达信（备选）
- **协议实现**：基于 [tdx-go](ref/tdx-go) 实现的 TCP 协议客户端
- **服务器直连**：直接连接通达信行情服务器（端口 7709），不受 HTTP API 限流影响
- **使用方法**：添加 `--data-source tdx` 参数
- **详细文档**：[通达信使用指南](docs/TDX_USAGE.md) | [集成文档](docs/TDX_INTEGRATION.md)

**使用示例**：
```bash
# 使用通达信数据源训练
python -m hp_ml.train --data-source tdx --start 20200101

# 测试通达信连接
python -m hp_ml.tdx_client

# 运行完整测试
python scripts/test_tdx_client.py
```

**数据源对比**：
| 特性 | 东方财富 | 通达信 |
|------|---------|--------|
| 协议 | HTTP API | TCP 协议 |
| 稳定性 | 可能限流 | 更稳定 |
| 速度 | 中等 | 较快 |
| 复权 | 支持 | 原始数据 |

## 深度学习模型

项目支持多种深度学习模型，包括 LSTM、GRU 及其各种 Attention 变种。详细文档：[GRU_ATTENTION_MODELS.md](GRU_ATTENTION_MODELS.md)

### 可用模型

#### 基础序列模型
- **LSTM**: 长短期记忆网络，适合时序预测
- **GRU**: 门控循环单元，比 LSTM 更快更轻量
- **双向 LSTM**: 捕捉双向时序依赖
- **双向 GRU**: GRU 的双向版本，训练速度快

#### Attention 变种
- **注意力 LSTM**: 单头注意力机制
- **注意力 GRU**: 更高效的 GRU 注意力版本 ⭐推荐
- **多头注意力 LSTM**: 多头注意力，更强表达能力
- **多头注意力 GRU**: GRU 版本，训练更快 ⭐推荐
- **分层注意力 GRU**: 双层注意力，捕捉多尺度模式 ⭐推荐

#### 混合架构
- **LSTM-Transformer**: LSTM + Transformer 混合
- **GRU-Transformer**: GRU + Transformer 混合，更高效 ⭐推荐
- **Transformer XL**: 扩展 Transformer 模型

### GRU vs LSTM

| 特性 | GRU | LSTM |
|------|-----|------|
| 训练速度 | 快 20-30% | 基准 |
| 参数量 | 少 25-30% | 基准 |
| 内存占用 | 更低 | 基准 |
| 小数据集 | 更不易过拟合 | 容易过拟合 |
| 长序列 | 适中 | 更强 |
| 推荐场景 | 快速迭代、生产部署 | 极长序列 |

**推荐**: 优先使用 GRU 变种，特别是注意力 GRU、多头注意力 GRU 和 GRU-Transformer

### 使用示例

通过 CLI:
```bash
python -m hp_ml.cli
# 选择 "注意力 GRU" 或 "多头注意力 GRU"
```

通过代码:
```python
from hp_ml.models_extended import make_extended_model

# 创建注意力 GRU
model = make_extended_model("attention_gru", seq_length=20, units=64)

# 训练
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)
```

### 快速开始

运行示例:
```bash
python examples/gru_attention_quickstart.py
```

### 指数族定义
代码中仅用于分类；候选 ETF 仍来自实时 ETF 列表扫描。


### 无第三方依赖备用训练

如果当前环境暂时无法安装 pandas/scikit-learn，可以直接使用纯 Python 备用流水线：

```bash
.venv/bin/python -m hp_ml.lite_train --start 20180101 --max-etfs-per-index 3 --horizon 5
# 或
make train-lite
```

备用流水线同样会自动发现 ETF、拉取历史 K 线、生成特征、训练岭回归模型，并输出 `reports/*_lite.*`、`reports/charts/*_lite.svg` 和 `models/csi_broad_etf_model_lite.pkl`。

## 导出通达信公式

项目支持将训练好的模型导出为通达信公式，用于在通达信软件中进行推理预测。**所有模型类型都支持导出！**

### 支持的模型类型

| 模型类型 | 导出方式 | 精确度 | 说明 |
|---------|---------|--------|------|
| **Ridge 线性回归** | ✅ 精确导出 | 100% | 完全精确，强烈推荐 |
| **HGB 梯度提升树** | ⚠️ 近似导出 | 50-80% | 使用特征重要性近似 |
| **RandomForest 随机森林** | ⚠️ 近似导出 | 60-75% | 使用特征重要性近似 |
| **EnhancedRandomForest** | ⚠️ 近似导出 | 60-75% | 使用特征重要性近似 |
| **LSTM/GRU 深度学习** | ⚠️ 知识蒸馏导出 | 40-60% | 使用知识蒸馏近似，仅供参考 |
| **Prophet 时间序列** | ⚠️ 知识蒸馏导出 | 40-60% | 使用知识蒸馏近似，仅供参考 |
| **Transformer 系列** | ⚠️ 知识蒸馏导出 | 40-60% | 使用知识蒸馏近似，仅供参考 |

**推荐：** 如果需要在通达信中使用，建议训练 Ridge 模型以获得最高精度。深度学习模型的导出公式仅供辅助参考。

### 导出命令

**批量导出所有指数族：**

```bash
# 导出 Ridge 模型（精确，推荐）
python -m hp_ml.export_tdx \
  --model models/csi_broad_etf_model_lite.pkl \
  --out reports/tdx_formulas \
  --all-families

# 导出 HGB 模型（近似）
python -m hp_ml.export_tdx \
  --model models/csi_broad_etf_model.joblib \
  --out reports/tdx_formulas \
  --all-families

# 导出深度学习模型（知识蒸馏，低精度）
python -m hp_ml.export_tdx \
  --model models/model_lstm.joblib \
  --out reports/tdx_formulas \
  --all-families
```

**导出单个指数族：**

```bash
python -m hp_ml.export_tdx \
  --model models/csi_broad_etf_model_lite.pkl \
  --family-id CSI_300 \
  --out reports/hp_ml_csi300.tdx
```

**参数说明：**
- `--model`: 模型文件路径（.pkl 或 .joblib）
- `--out`: 输出路径（文件或目录）
- `--all-families`: 批量导出所有指数族
- `--family-id`: 导出指定指数族（如 CSI_300, CSI_500）
- `--signal-threshold`: 买入信号阈值（默认 0.0）
- `--encoding`: 文件编码（默认 utf-8，旧版通达信用 gbk）

### 公式输出

生成的通达信公式包含两个输出指标：

- **HPMLSCORE**：模型预测的未来 N 日收益分数
- **HPMLBUY**：买入信号（HPMLSCORE > threshold 时为 1，否则为 0）

### 导出示例

导出后的文件位于 `reports/tdx_formulas/` 目录：

```
reports/tdx_formulas/
├── HPML_宽基_ETF_CSI_300.tdx    # 沪深300 ETF 公式
├── HPML_宽基_ETF_CSI_500.tdx    # 中证500 ETF 公式
├── HPML_宽基_ETF_CSI_1000.tdx   # 中证1000 ETF 公式
└── ...
```

### 导出方法说明

**1. 精确导出（Ridge 模型）**
- 100% 精确，公式输出与 Python 模型完全一致
- 强烈推荐用于生产环境

**2. 近似导出（树模型）**
- 使用特征重要性权重或均匀权重近似
- 精度 50-80%，可用于推理
- 公式中包含警告信息

**3. 知识蒸馏导出（深度学习模型）**
- 使用简单模型学习深度模型的输出
- 精度 40-60%，仅供辅助参考
- 建议结合 Python 原始模型使用

### 技术说明

- ETF 换手率使用 `VOL/CAPITAL*100` 近似
- `days_since_start` 使用 `BARSCOUNT(CLOSE)-1` 近似
- 树模型近似采用特征重要性权重或均匀权重
- 深度学习模型采用知识蒸馏方法导出线性近似

## 📚 文档

- [交互式 CLI 指南](docs/CLI_GUIDE.md) - 可视化菜单系统使用指南 ⭐️ 新增
- [快速入门](docs/QUICKSTART.md) - 5 分钟快速上手
- [多模型训练](docs/MULTI_MODEL.md) - 6 种模型对比训练
- [高级功能](docs/ADVANCED_FEATURES.md) - 超参数优化、集成学习、AI 辅助
- [预测报告](docs/PREDICTION_REPORT.md) - 预测 vs 实际对比可视化
- [算法公式](docs/FORMULAS.md) - 数学公式和技术细节
- [功能总结](docs/SUMMARY.md) - 功能特性对比表

## 🎯 核心功能

### 基础功能
- ✅ 自动发现中证宽基 ETF
- ✅ 历史数据拉取和缓存
- ✅ 特征工程（动量、波动率、均线等）
- ✅ 多模型训练（Ridge、HGB、RF、Prophet、LSTM）
- ✅ 时间序列划分验证
- ✅ 回测框架

### 高级功能（v3.0+）
- ✅ 交互式 CLI（可视化菜单系统）⭐️ 新增
- ✅ 自动超参数调优（Optuna）
- ✅ 模型集成学习（Stacking/Blending）
- ✅ 技术指标特征学习（50+ 指标）
- ✅ AI 辅助模型进化（Claude API + Claude Code）
- ✅ 预测报告生成（预测 vs 实际对比）

### 可视化和报告
- ✅ 自动生成图表和 Markdown 报告
- ✅ 预测 vs 实际对比图表
- ✅ 误差分析和性能指标
- ✅ 分股票对比和综合仪表板
- ✅ 回测收益曲线

## 🚀 快速命令

```bash
# 交互式 CLI（推荐）
make cli                # 启动交互式 CLI 菜单系统 ⭐️

# 基础训练
make train              # 单模型训练
make train-multi        # 多模型对比训练

# 高级功能
make train-advanced     # 启用超参数优化和集成学习
make auto-evolve        # AI 自主进化优化

# 测试和演示
make test-prediction-report  # 测试预测报告生成
make test-multi              # 测试多模型功能

# CLI 演示脚本（非交互式环境）
python examples/demo_cli_usage.py train     # 演示训练
python examples/demo_cli_usage.py full      # 演示完整流程
```

