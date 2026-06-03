import os
import pandas as pd

# 路径配置
raw_csv_path = "Chinese Stock Market 1990-2024.csv"
output_dir = "data/topic2_broad_base"  # 为选题 2 创建专属数据子目录
os.makedirs(output_dir, exist_ok=True)

# 🎯 选题 2 指定的 9 只核心宽基 ETF 列表（已自动转换为 Yahoo Finance 格式）
topic2_tickers = [
    "159361.SZ", "159352.SZ", "159351.SZ",  # 中证 A500 系列等深市新宽基
    "512050.SS",                             # 中证 500 系列等
    "563360.SS",                             # 华泰柏瑞中证 A500 等上市沪市宽基
    "510310.SS", "510300.SS", "159919.SZ",  # 核心大盘沪深 300 系列（易方达、华泰、嘉实）
    "510050.SS"                              # 上证 50 核心蓝筹
]

print(f"🚀 开始为《选题 2》流式扫描 1.5G 原始数据集...")

# 初始化存储字典
extracted_data = {ticker: [] for ticker in topic2_tickers}
use_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

chunk_size = 250000
chunk_count = 0

# 开始免爆内存的流式分块读取
for chunk in pd.read_csv(raw_csv_path, chunksize=chunk_size, usecols=use_cols, low_memory=False):
    chunk_count += 1
    chunk["Ticker"] = chunk["Ticker"].astype(str)
    
    # 精准匹配
    for ticker in topic2_tickers:
        matched_rows = chunk[chunk["Ticker"] == ticker]
        if not matched_rows.empty:
            extracted_data[ticker].append(matched_rows)
            
    if chunk_count % 20 == 0:
        print(f"已流式扫描 {chunk_count * chunk_size // 10000} 万行...")

print("\n💾 正在导出《选题 2》专属数据集...")
saved_count = 0

for ticker, dfs in extracted_data.items():
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.sort_values(by="Date", inplace=True)
        
        # 纯数字文件名，方便模型读取
        clean_name = ticker.split(".")[0]
        output_path = os.path.join(output_dir, f"{clean_name}_daily.csv")
        
        final_df.to_csv(output_path, index=False)
        print(f"📊 成功提取 -> {output_path} (共 {len(final_df)} 行交易历史)")
        saved_count += 1
    else:
        print(f"❌ 原始数据集中未找到：{ticker}")

print(f"\n任务结束：成功提取出 {saved_count} 只 ETF。")

# 🚨 关键避坑检查：如果发现提取出来的文件都是 0 字节，或者提示未找到（特别是新发行的中证 A500 系列）
if saved_count == 0:
    print("\n💡 【Vibe 避坑提示】:")
    print("你下载的这个 Kaggle 1990-2024 数据集极大概率是【纯 A 股个股】数据集，未收录基金（ETF）。")
    print("且中证 A500 系列（如 159351 等）大批是在 2024 年底至 2025 年才发行的新基，该历史包绝对没有！")
    print("请立刻执行下方备用方案，利用 yfinance 在 5 秒内直接海外实时拉取这 9 只宽基。")