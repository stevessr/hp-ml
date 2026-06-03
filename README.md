# hp-ml：中证宽基指数 ETF 自动挖掘、拉取数据与训练

这个项目把“自动搜索 → 挖掘候选 ETF → 拉取历史行情 → 特征工程 → 训练模型 → 输出排序报告”串成一条可运行流水线，重点覆盖中证/沪深宽基指数 ETF（沪深300、中证500、中证1000、中证2000、中证800、中证A500、中证A50、中证100）。

> 说明：输出仅用于量化研究和模型验证，不构成投资建议。

## 快速开始

```bash
# 依赖安装
make setup

# 只发现候选 ETF
make discover

# 拉取 2018 至今数据并训练 5 日前瞻收益模型
make train

# 更快的短历史 smoke run
make quick
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
2. **聚焦中证宽基**：用名称模式识别沪深300/中证500/中证1000/中证2000/中证800/中证A500/中证A50/中证100 等宽基族。
3. **剔除非纯宽基**：默认排除行业、主题、风格、跨境、债券、货币和指数增强产品；可用 `--include-enhanced` / `--include-style` 放开。
4. **拉取历史行情**：对筛出的 ETF 拉取前复权日线 K 线，并缓存到 `data/raw/history/`。
5. **设计训练样本**：生成动量、波动率、均线乖离、回撤、成交额冲击、换手率、振幅、月份周期、指数族哑变量等特征。
6. **监督学习训练**：默认训练 `HistGradientBoostingRegressor`，预测未来 `--horizon` 个交易日收益。
7. **时间切分验证**：按交易日切分训练/测试，输出 MAE、RMSE、方向准确率、按日 Spearman IC、Top1 相对全池收益等指标。

## 关键输出

- `reports/latest_candidates.csv`：自动发现和排序后的中证宽基 ETF 候选池。
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
- `reports/related_stocks_lite.csv`：每个宽基指数族对应的更多相关股票/指数成分股。
- `reports/strategy_backtest_daily_lite.csv`：每日滚动购买策略回测结果。
- `reports/strategy_backtest_trades_lite.csv`：逐笔买入信号、实际未来收益、扣费后收益和胜负。
- `reports/training_metrics.json`：机器可读训练指标。
- `reports/training_summary.md`：中文训练摘要报告。
- `reports/charts/*.svg`：预测排序、指数族覆盖、验证指标、holdout 累计曲线、自动调参和相关股票暴露图。

## 常用参数

```bash
.venv/bin/python -m hp_ml.train --help
```

- `--start 20180101`：历史数据开始日期。
- `--end YYYYMMDD`：历史数据结束日期，默认今天。
- `--horizon 5`：预测未来 N 个交易日收益。
- `--max-etfs-per-index 3`：每个宽基指数族按成交额保留前 N 只 ETF。
- `--min-amount 10000000`：按当日成交额过滤低流动性产品。
- `--model hgb|rf|ridge`：选择梯度提升、随机森林或岭回归。
- `--auto-tune` / `--no-auto-tune`：启用或关闭自动参数调优；默认启用。
- `--tune-l2-grid 0.3,1,3,10,30`：岭回归正则强度候选。
- `--tune-top-k-grid 1,2,3,4,5`：购买策略 TopK 候选。
- `--tune-min-pred-grid -0.005,0,0.005,0.01,0.02`：买入阈值候选。
- `--strategy-top-k 3`：未自动调参时，每个决策日买入预测排序前 K 只。
- `--strategy-min-pred 0`：未自动调参时，买入所需最低预测未来收益。
- `--round-trip-cost-bps 10`：回测中每次完整买卖的成本/滑点，单位 bps。
- `--related-stocks-per-index 30`：每个指数族拉取多少只相关股票/成分股。
- `--force`：忽略缓存，重新拉取行情。



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

- 沪深300、中证500、中证1000、中证2000、中证800、中证A500、中证A50 等。
- 有权重字段的指数按权重排序；无权重字段时按自由流通市值排序。
- 输出股票代码、名称、行业、地区、权重、自由流通市值、PE、涨跌幅等字段。

输出文件：

- `reports/related_stocks_lite.csv`
- `reports/charts/related_stock_top_exposure_lite.svg`

## 数据源

- ETF 列表与行情：优先使用东方财富 ETF 行情 `push2.eastmoney.com/api/qt/clist/get`；不可用时回退到 `fund.eastmoney.com/js/fundcode_search.js`。
- ETF 历史 K 线：优先使用东方财富 K 线 `push2his.eastmoney.com/api/qt/stock/kline/get`；不可用时回退到腾讯 K 线 `web.ifzq.gtimg.cn/appstock/app/fqkline/get`。
- 指数族定义：代码中仅用于分类；候选 ETF 仍来自实时 ETF 列表扫描。


### 无第三方依赖备用训练

如果当前环境暂时无法安装 pandas/scikit-learn，可以直接使用纯 Python 备用流水线：

```bash
.venv/bin/python -m hp_ml.lite_train --start 20180101 --max-etfs-per-index 3 --horizon 5
# 或
make train-lite
```

备用流水线同样会自动发现 ETF、拉取历史 K 线、生成特征、训练岭回归模型，并输出 `reports/*_lite.*`、`reports/charts/*_lite.svg` 和 `models/csi_broad_etf_model_lite.pkl`。
