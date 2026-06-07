# ETF 自动信号下钻：看涨股票成交额与股东占比深挖

- 生成时间：2026-06-07T16:18:14
- ETF 信号文件：`/home/steve/文档/vibe coding/hp-ml/reports/latest_predictions_lite.csv`
- 相关股票文件：`/home/steve/文档/vibe coding/hp-ml/reports/related_stocks_lite.csv`
- 入池 ETF 信号：3 只；股票候选池：8 只；输出看涨股票：5 只。
- 口径：成交额/换手来自东方财富 A 股行情与日 K；股东占比来自东方财富股东分析十大流通股东/十大股东。
- 说明：本报告是量化研究工件，不构成投资建议。

## 1. ETF 信号入口

|排名|日期|ETF|名称|指数族|预测5日收益|信号原因|
|---:|---|---|---|---|---:|---|
|1|2026-06-02|510500|中证500ETF南方|CSI_500|0.06%|pred>= 0.0000|
|2|2026-06-03|560510|中证A500ETF泰康|CSI_A500|0.06%|pred>= 0.0000|
|3|2026-06-03|560530|中证A500ETF摩根|CSI_A500|-0.02%|top3 ETF signal|

## 2. 看涨股票核心清单

|排名|股票|行业|ETF/指数线索|机制标签|评分|成交额|换手|5日收益|20日收益|前十大流通股东占比|机构流通占比|第一流通股东|
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|1|002281 光迅科技|通信设备|CSI_500 / 510500|ETF信号牵引;价格动量;高换手交易;流通股东集中;机构股东参与|39.515|169.78亿|9.44%|11.59%|25.93%|47.95%|47.38%|烽火科技集团有限公司 (37.37%)|
|2|600487 亨通光电|通信设备|CSI_500 / 510500|ETF信号牵引;价格动量;成交额放大;高换手交易;流通股东集中;机构股东参与|35.150|247.42亿|10.26%|24.64%|24.39%|36.69%|32.29%|亨通集团有限公司 (24.28%)|
|3|002008 大族激光|自动化设备|CSI_500 / 510500|ETF信号牵引;高换手交易;流通股东集中;机构股东参与|16.650|74.58亿|5.94%|-3.02%|4.93%|37.66%|35.15%|大族控股集团有限公司 (16.91%)|
|4|300604 长川科技|半导体|CSI_500 / 510500|ETF信号牵引;高换手交易;机构股东参与|15.162|64.67亿|6.25%|-1.88%|13.64%|23.70%|13.12%|赵轶 (7.23%)|
|5|688525 佰维存储|半导体|CSI_500 / 510500|ETF信号牵引;高换手交易;流通股东集中;机构股东参与;ROE支撑|12.267|155.08亿|10.23%|-5.00%|7.08%|35.50%|16.44%|孙成思 (17.77%)|

## 3. 机制拆解

- 行业集中：通信设备(2)，半导体(2)，自动化设备(1)。
- 交易确认：输出股票平均成交额约 142.31亿，平均换手 8.42%；成交额放大与高换手会被写入 `mechanism_tags`。
- 股东结构：前十大流通股东平均占比 36.30%；机构/基金类持股占比单独保存在 CSV，便于过滤“筹码集中 + 机构参与”的股票。
- 成交额最活跃：600487亨通光电(247.42亿)，002281光迅科技(169.78亿)，688525佰维存储(155.08亿)，002008大族激光(74.58亿)，300604长川科技(64.67亿)。
- 流通股东最集中：002281光迅科技(47.95%)，002008大族激光(37.66%)，600487亨通光电(36.69%)，688525佰维存储(35.50%)，300604长川科技(23.70%)。

## 4. 输出文件

- 看涨股票深挖表：`/home/steve/文档/vibe coding/hp-ml/reports/bullish_stock_deep_dive.csv`
- 股东明细表：`/home/steve/文档/vibe coding/hp-ml/reports/bullish_stock_deep_dive_shareholders.csv`
- 历史股东变迁表：`/home/steve/文档/vibe coding/hp-ml/reports/bullish_stock_deep_dive_holder_history.csv`
- 个人股东逐期变化表：`/home/steve/文档/vibe coding/hp-ml/reports/bullish_stock_deep_dive_individual_holder_changes.csv`
- 摘要 JSON：`/home/steve/文档/vibe coding/hp-ml/reports/bullish_stock_deep_dive.json`
- 公司股东成分报告索引：`/home/steve/文档/vibe coding/hp-ml/reports/shareholder_composition/index.md`
  - `002281` 光迅科技：`/home/steve/文档/vibe coding/hp-ml/reports/shareholder_composition/01_002281_光迅科技_shareholder_composition.md`
  - `600487` 亨通光电：`/home/steve/文档/vibe coding/hp-ml/reports/shareholder_composition/02_600487_亨通光电_shareholder_composition.md`
  - `002008` 大族激光：`/home/steve/文档/vibe coding/hp-ml/reports/shareholder_composition/03_002008_大族激光_shareholder_composition.md`
  - `300604` 长川科技：`/home/steve/文档/vibe coding/hp-ml/reports/shareholder_composition/04_300604_长川科技_shareholder_composition.md`
  - `688525` 佰维存储：`/home/steve/文档/vibe coding/hp-ml/reports/shareholder_composition/05_688525_佰维存储_shareholder_composition.md`
- 图表 score_top：`/home/steve/文档/vibe coding/hp-ml/reports/charts/bullish_stock_score_top.svg`
- 图表 amount_top：`/home/steve/文档/vibe coding/hp-ml/reports/charts/bullish_stock_amount_top.svg`
- 图表 holder_concentration：`/home/steve/文档/vibe coding/hp-ml/reports/charts/bullish_stock_holder_concentration.svg`
- 图表 industry_counts：`/home/steve/文档/vibe coding/hp-ml/reports/charts/bullish_stock_industry_counts.svg`

## 数据源与风险提示

- 行情列表：https://push2.eastmoney.com/api/qt/clist/get
- 日 K：https://push2his.eastmoney.com/api/qt/stock/kline/get
- 股东分析：https://datacenter-web.eastmoney.com/api/data/v1/get (RPT_F10_EH_FREEHOLDERS / RPT_DMSK_HOLDERS)
- 东方财富页面也声明相关信息仅用于传播更多信息，不构成投资建议；实际交易需再核对交易所公告、财报与风险承受能力。
