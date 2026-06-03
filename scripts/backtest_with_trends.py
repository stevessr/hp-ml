import os
import pandas as pd
import numpy as np

# 1. 路径设置
etf_path = "data/topic2_broad_base/510300_daily.csv"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

if not os.path.exists(trends_path):
    print("❌ 错误：未找到谷歌趋势数据，请先确保下载成功。")
    exit()

print("🏃 正在启动《行为金融学舆情因子》复合增强回测引擎...")

# 2. 加载并清洗 ETF 价格数据
etf_df = pd.read_csv(etf_path)
date_etf_col = [col for col in etf_df.columns if 'date' in col.lower()][0]
etf_df[date_etf_col] = pd.to_datetime(etf_df[date_etf_col])
etf_df.set_index(date_etf_col, inplace=True)

# 确保价格列为 1D Series
close_col = 'Close' if 'Close' in etf_df.columns else etf_df.columns[0]
etf_series = etf_df[close_col]
if isinstance(etf_series, pd.DataFrame):
    etf_series = etf_series.iloc[:, 0]

# 3. 加载并清洗舆情数据
trends_df = pd.read_csv(trends_path)
date_trend_col = [col for col in trends_df.columns if 'date' in col.lower()][0]
trends_df[date_trend_col] = pd.to_datetime(trends_df[date_trend_col])
trends_df.set_index(date_trend_col, inplace=True)

# 4. 核心对齐：将周频/不定期舆情数据，升频映射到 A 股 每一个真实交易日上
# 采用量化常用的前向填充 (Forward Fill)，保证回测时当天使用的是最新已知的舆情
trends_daily = trends_df.reindex(etf_df.index, method='ffill')

# 5. 构建异常关注度因子 (AGTI)
# 选用散户常用搜索词 "A 股" 计算其滚动 Z-Score。由于对齐后是日频，窗口设为 20 个交易日（约 1 个月）
trends_daily['Trend_MA'] = trends_daily['A 股'].rolling(window=20).mean()
trends_daily['Trend_STD'] = trends_daily['A 股'].rolling(window=20).std()
trends_daily['Z_Score'] = (trends_daily['A 股'] - trends_daily['Trend_MA']) / trends_daily['Trend_STD']
trends_daily['Z_Score'] = trends_daily['Z_Score'].fillna(0)

# 6. 联立策略信号矩阵
merged = pd.DataFrame(index=etf_df.index)
merged['Price'] = etf_series.astype(float)
merged['Daily_Return'] = merged['Price'].pct_change().fillna(0)
merged['SMA20'] = merged['Price'].rolling(window=20).mean()
merged['Sentiment_Z'] = trends_daily['Z_Score']

# 信号 A：基础技术面信号（价格 > 20 日均线）
merged['Base_Signal'] = (merged['Price'] > merged['SMA20']).astype(int)

# 信号 B：舆情安全阀（当散户搜索热度异常暴增 Z_Score > 1.2 时，代表市场极度过热，强制空仓避险）
merged['Enhanced_Signal'] = merged['Base_Signal']
merged.loc[merged['Sentiment_Z'] > 1.2, 'Enhanced_Signal'] = 0

# 🌟 关键：平移一天，消除未来函数（Look-ahead bias），确保明天交易使用的是今天的收盘信号
merged['Base_Signal_Delayed'] = merged['Base_Signal'].shift(1).fillna(0)
merged['Enhanced_Signal_Delayed'] = merged['Enhanced_Signal'].shift(1).fillna(0)

# 7. 计算各策略累计净值
merged['B&H_Cum'] = (1 + merged['Daily_Return']).cumprod() - 1
merged['SMA20_Cum'] = (1 + merged['Base_Signal_Delayed'] * merged['Daily_Return']).cumprod() - 1
merged['Sentiment_Enhanced_Cum'] = (1 + merged['Enhanced_Signal_Delayed'] * merged['Daily_Return']).cumprod() - 1

# 截取全共存时间区间（通常从 2024-11-08 之后开始）
test_zone = merged.dropna()

# 8. 指标计算函数
def get_metrics(returns, cum_series):
    total_ret = cum_series.iloc[-1]
    ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    cum_prices = (1 + returns).cumprod()
    max_dd = ((cum_prices - cum_prices.cummax()) / cum_prices.cummax()).min()
    return total_ret, ann_ret, sharpe, max_dd

# 评估三个核心阵营
bh_tot, bh_ann, bh_sha, bh_dd = get_metrics(test_zone['Daily_Return'], test_zone['B&H_Cum'])
sma_tot, sma_ann, sma_sha, sma_dd = get_metrics(test_zone['Base_Signal_Delayed'] * test_zone['Daily_Return'], test_zone['SMA20_Cum'])
enh_tot, enh_ann, enh_sha, enh_dd = get_metrics(test_zone['Enhanced_Signal_Delayed'] * test_zone['Daily_Return'], test_zone['Sentiment_Enhanced_Cum'])

# 打印表格
print("\n" + "="*82)
print("📈 融合《Google 搜索舆情异常因子》后的沪深 300 ETF(510300) 增强策略绩效对比")
print("="*82)
print(f"{'组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*82)
print(f"{'纯基准买入持有 (B&H)':<22}{bh_tot:>10.2%}{bh_ann:>12.2%}{bh_sha:>12.2f}{bh_dd:>12.2%}")
print(f"{'纯 SMA20 技术指标增强':<22}{sma_tot:>10.2%}{sma_ann:>12.2%}{sma_sha:>12.2f}{sma_dd:>12.2%}")
print(f"{'💡 LLM/谷歌舆情净化增强':<21}{enh_tot:>10.2%}{enh_ann:>12.2%}{enh_sha:>12.2f}{enh_dd:>12.2%}")
print("="*82)

# 保存细节供后续分析
test_zone.to_csv(os.path.join(reports_dir, "sentiment_enhanced_backtest_detail.csv"))
print("💾 每日信号明细与累计收益率数据已导出至 reports/sentiment_enhanced_backtest_detail.csv")