import os
import pandas as pd
import numpy as np

# 路径设置
etf_path = "data/topic2_broad_base/510300_daily.csv"
trends_path = "data/alternative_data/google_trends_sentiment.csv"

if not os.path.exists(trends_path):
    print("❌ 错误：未找到谷歌趋势数据，请先运行 download_google_trends.py")
    exit()

# 1. 加载并清洗 ETF 数据
etf_df = pd.read_csv(etf_path)
etf_df['Date'] = pd.to_datetime(etf_df['Date'])
etf_df.set_index('Date', inplace=True)

# 2. 加载并清洗舆情数据
trends_df = pd.read_csv(trends_path)
trends_df['date'] = pd.to_datetime(trends_df['date'])
trends_df.set_index('date', inplace=True)

# 将周频/不定期舆情数据升频对齐到 A 股日线数据，采用前向填充 (Forward Fill)
trends_daily = trends_df.reindex(etf_df.index, method='ffill')

# 3. 构建异常关注度因子 (基于 "A 股" 搜索量计算 rolling Z-Score)
# 选用过去 4 个舆情周期作为基准
trends_daily['Trend_MA'] = trends_daily['A 股'].rolling(window=4).mean()
trends_daily['Trend_STD'] = trends_daily['A 股'].rolling(window=4).std()
trends_daily['Z_Score'] = (trends_daily['A 股'] - trends_daily['Trend_MA']) / trends_daily['Trend_STD']
trends_daily['Z_Score'] = trends_daily['Z_Score'].fillna(0)

# 4. 联立策略信号
merged = pd.DataFrame(index=etf_df.index)
merged['Price'] = etf_df['Close'].astype(float)
merged['Log_Ret'] = merged['Price'].pct_change().fillna(0)
merged['SMA20'] = merged['Price'].rolling(window=20).mean()
merged['Sentiment_Z'] = trends_daily['Z_Score']

# 基础信号：价格高于均线
merged['Base_Signal'] = (merged['Price'] > merged['SMA20']).astype(int)

# 舆情过滤器：如果散户异常关注度暴增 (Z > 1.5)，强制清仓防御
merged['Final_Signal'] = merged['Base_Signal']
merged.loc[merged['Sentiment_Z'] > 1.5, 'Final_Signal'] = 0

# 平移一天防止未来函数
merged['Final_Signal'] = merged['Final_Signal'].shift(1).fillna(0)
merged['Base_Signal'] = merged['Base_Signal'].shift(1).fillna(0)

# 5. 计算收益率
merged['B&H_Cum'] = (1 + merged['Log_Ret']).cumprod() - 1
merged['SMA20_Cum'] = (1 + merged['Base_Signal'] * merged['Log_Ret']).cumprod() - 1
merged['Trend_Enhanced_Cum'] = (1 + merged['Final_Signal'] * merged['Log_Ret']).cumprod() - 1

# 截取全共存区间查看绩效
test_zone = merged.loc['2024-11-08':]

print("\n==========================================================================")
print("📊 融合搜索引擎舆情因子后的《指数增强策略》绩效看板")
print("==========================================================================")
print(f"纯被动买入持有最终收益：  {test_zone['B&H_Cum'].iloc[-1]:.2%}")
print(f"纯 SMA20 技术指标增强收益： {test_zone['SMA20_Cum'].iloc[-1]:.2%}")
print(f"💡 引入 Google 舆情净化后的增强收益：{test_zone['Trend_Enhanced_Cum'].iloc[-1]:.2%}")
print("==========================================================================")