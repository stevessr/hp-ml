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
- `data/processed/training_panel.csv`：模型训练面板。
- `models/csi_broad_etf_model.joblib`：模型、特征列、目标、候选池和指标的 joblib 包。
- `reports/latest_predictions.csv`：最新日期每只候选 ETF 的模型预测排序。
- `reports/training_metrics.json`：机器可读训练指标。
- `reports/training_summary.md`：中文训练摘要报告。
- `reports/charts/*.svg`：预测排序、指数族覆盖、验证指标和 holdout 累计曲线图。

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
- `--force`：忽略缓存，重新拉取行情。

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
