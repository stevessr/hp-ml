import os
import pandas as pd
import numpy as np

data_dir = "data/topic2_broad_base"
files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]

if not files:
    print(f"❌ 错误：在 {data_dir} 目录下未找到任何 CSV 文件，请先运行下载脚本。")
    exit()

print("📊 开始读取宽基 ETF 数据并自动适配列名...")

# 使用字典存储各资产的收益率 Series，最后统一对齐，防范 index 不一致
series_dict = {}

for file in files:
    ticker = file.split('_')[0]
    file_path = os.path.join(data_dir, file)
    
    # 文件很小，直接全量加载进内存
    df = pd.read_csv(file_path)
    
    # 1. 动态寻找日期列（防范大小写差异，如 Date 或 date）
    date_col = [col for col in df.columns if 'date' in col.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    
    # 2. 动态寻找收盘价列（优先使用 Adj Close，其次使用 Close）
    available_cols = df.columns.tolist()
    if 'Adj Close' in available_cols:
        target_close = 'Adj Close'
    elif 'Close' in available_cols:
        target_close = 'Close'
    else:
        # 兜底：寻找任何带有 close 字眼的列
        close_candidates = [col for col in available_cols if 'close' in col.lower()]
        target_close = close_candidates[0] if close_candidates else available_cols[0]
        
    # 3. 计算日收益率 (Daily Return)
    # yfinance 导出的数据如果是多级表头，通过 values 降维成 1D 数组
    close_series = df[target_close].values.flatten()
    
    # 重新构建带日期索引的单列 Series
    return_series = pd.Series(close_series, index=df.index).pct_change()
    
    series_dict[ticker] = return_series
    print(f" └─ 已成功解析 {ticker}，使用价格列：[{target_close}]")

# 4. 核心：通过 pandas 字典整合成大表，这一步会自动按 Date 索引完美对齐所有资产
all_ret_df = pd.DataFrame(series_dict)

print(f"\n📅 全量数据时间跨度：{all_ret_df.index.min().strftime('%Y-%m-%d')} 至 {all_ret_df.index.max().strftime('%Y-%m-%d')}")

# 创建报告输出目录
os.makedirs("reports", exist_ok=True)

# -------------------------------------------------------------
# 策略阶段 1：包含最新中证 A500 的近期相关性（截取新基金上市后的重合期）
# -------------------------------------------------------------
print("\n=== 🎯 策略阶段 1: 包含最新中证 A500 的近期相关性矩阵 (全资产对齐) ===")
recent_df = all_ret_df.dropna()  # 剔除所有缺失值，保留全资产共存的最新周期
if not recent_df.empty:
    corr_matrix_recent = recent_df.corr()
    print(corr_matrix_recent.round(3))
    corr_matrix_recent.to_csv("reports/recent_etf_correlation.csv")
    print("💾 近期相关性矩阵已保存至 reports/recent_etf_correlation.csv")
else:
    print("⚠️ 提示：各标的交易日没有完美的交集，可能是因为新基刚上市。")

# -------------------------------------------------------------
# 策略阶段 2：传统经典老牌宽基的长周期历史相关性（回溯近 10 年）
# -------------------------------------------------------------
print("\n=== 🏛️ 策略阶段 2: 传统经典大宽基长周期相关性矩阵 (2015 至今) ===")
# 筛选历史足够长的传统宽基
classic_tickers = [t for t in ['510050', '510300', '512050', '159919'] if t in all_ret_df.columns]
if classic_tickers:
    classic_df = all_ret_df[classic_tickers].loc['2015-01-01':].dropna()
    if not classic_df.empty:
        corr_matrix_classic = classic_df.corr()
        print(corr_matrix_classic.round(3))
        corr_matrix_classic.to_csv("reports/classic_etf_correlation.csv")
        print("💾 长周期相关性矩阵已保存至 reports/classic_etf_correlation.csv")
    else:
        print("⚠️ 传统宽基在设定的长周期区间内无足够数据。")
else:
    print("⚠️ 未能在数据中识别到经典宽基代码。")

print("\n🎉 EDA 流程顺利完成！")