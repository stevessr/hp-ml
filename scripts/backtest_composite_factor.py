import os
import pandas as pd
import numpy as np

# 1. 路径设置
etf_path = "data/topic2_broad_base/510300_daily.csv"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🚀 正在启动《量价 + 舆情复合 Alpha 因子》高级量化回测引擎...")

# 2. 加载并清洗数据
etf_df = pd.read_csv(etf_path)
date_etf_col = [col for col in etf_df.columns if 'date' in col.lower()][0]
etf_df[date_etf_col] = pd.to_datetime(etf_df[date_etf_col])
etf_df.set_index(date_etf_col, inplace=True)
close_col = 'Close' if 'Close' in etf_df.columns else etf_df.columns[0]
etf_series = etf_df[close_col]
if isinstance(etf_series, pd.DataFrame):
    etf_series = etf_series.iloc[:, 0]

trends_df = pd.read_csv(trends_path)
date_trend_col = [col for col in trends_df.columns if 'date' in col.lower()][0]
trends_df[date_trend_col] = pd.to_datetime(trends_df[date_trend_col])
trends_df.set_index(date_trend_col, inplace=True)

# 3. 升频对齐：舆情数据前向填充至真实交易日
trends_daily = trends_df.reindex(etf_df.index, method='ffill')

# 4. 构建底层的单因子 (采用 Rolling Z-Score 进行无未来函数的标准化)
window = 20

# 因子 A：技术动量（偏离均线程度）
etf_ma = etf_series.rolling(window=window).mean()
raw_tech = (etf_series / etf_ma - 1).fillna(0)
f_tech = (raw_tech - raw_tech.rolling(window=window).mean()) / raw_tech.rolling(window=window).std()
f_tech = f_tech.fillna(0)

# 因子 B：搜索引擎舆情热度
raw_sent = trends_daily['A 股'].fillna(0)
f_sent = (raw_sent - raw_sent.rolling(window=window).mean()) / raw_sent.rolling(window=window).std()
f_sent = f_sent.fillna(0)

# 5. 因子线性合成（技术占 60% 权重，舆情作为反向锚定占 40% 权重）
composite_score = 0.6 * f_tech - 0.4 * f_sent

# 6. 构建策略矩阵（聚焦数据重合的黄金回测窗口：2024-11-08 起）
merged = pd.DataFrame(index=etf_df.index)
merged['Price'] = etf_series.astype(float)
merged['Daily_Return'] = merged['Price'].pct_change().fillna(0)
merged['F_Tech'] = f_tech
merged['F_Sent'] = f_sent
merged['Composite_Score'] = composite_score

# 🎯 信号生成逻辑
# 当复合 Alpha 得分 > 0 时看多（仓位为 1），否则空仓防御（仓位为 0）
merged['Signal'] = (merged['Composite_Score'] > 0).astype(int)

# 严格执行平移，规避未来函数
merged['Signal_Delayed'] = merged['Signal'].shift(1).fillna(0)

# 7. 截取重合区间，计算各方净值（确保基准与策略在同一起跑线）
test_zone = merged.loc['2024-11-08':].copy()
test_zone['B&H_Cum'] = (1 + test_zone['Daily_Return']).cumprod() - 1
test_zone['Strategy_Cum'] = (1 + test_zone['Signal_Delayed'] * test_zone['Daily_Return']).cumprod() - 1

# 8. 绩效指标精算
def calc_perf(returns, cum_series):
    total_ret = cum_series.iloc[-1]
    # 计算当前区间的实际年化收益
    ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    cum_prices = (1 + returns).cumprod()
    max_dd = ((cum_prices - cum_prices.cummax()) / cum_prices.cummax()).min()
    return total_ret, ann_ret, sharpe, max_dd

bh_tot, bh_ann, bh_sha, bh_dd = calc_perf(test_zone['Daily_Return'], test_zone['B&H_Cum'])
str_tot, str_ann, str_sha, str_dd = calc_perf(test_zone['Signal_Delayed'] * test_zone['Daily_Return'], test_zone['Strategy_Cum'])

# 9. 打印多因子看板
print("\n" + "="*82)
print("📊 《量价技术面 + 搜索引擎舆情面》合成复合因子策略绩效看板 (黄金窗口)")
print("="*82)
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*82)
print(f"{'被动买入持有 (沪深 300)':<22}{bh_tot:>10.2%}{bh_ann:>12.2%}{bh_sha:>12.2f}{bh_dd:>12.2%}")
print(f"{'💡 复合 Alpha 因子增强':<21}{str_tot:>10.2%}{str_ann:>12.2%}{str_sha:>12.2f}{str_dd:>12.2%}")
print("="*82)

# 导出因子的详细时序矩阵，供机器学习特征工程使用
output_path = os.path.join(reports_dir, "composite_factor_backtest_detail.csv")
test_zone.to_csv(output_path)
print(f"💾 复合因子分值与信号明细已安全导出至：{output_path}\n")
