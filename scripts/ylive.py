import os
import yfinance as yf
import pandas as pd

# 创建选题 2 专属的数据存放目录
output_dir = "data/topic2_broad_base"
os.makedirs(output_dir, exist_ok=True)

# 🎯 选题 2 指定的 9 只核心宽基 ETF（已完美映射海外雅虎 Ticker 规则）
topic2_tickers = [
    "159361.SZ", "159352.SZ", "159351.SZ",  # 中证 A500 系列（深交所）
    "512050.SS",                             # 中证 500ETF（上交所）
    "563360.SS",                             # 中证 A500ETF（上交所）
    "510310.SS", "510300.SS",                # 沪深 300ETF 系列（上交所）
    "159919.SZ",                             # 嘉实沪深 300ETF（深交所）
    "510050.SS"                              # 上证 50ETF（上交所）
]

print("🌐 开始通过本地代理，直接跨海拉取 9 只选题宽基 ETF 的真实最新历史行情...")
print("💡 雅虎数据将自动包含 2024-2026 最新核心周期，自带完美复权。")

success_count = 0

for ticker in topic2_tickers:
    clean_name = ticker.split(".")[0]
    output_path = os.path.join(output_dir, f"{clean_name}_daily.csv")
    
    print(f"正在拉取 {ticker} ...", end="", flush=True)
    
    try:
        # 使用 yf.download 批量拉取自上市以来的最大历史区间 (period="max")
        # group_by="column" 确保单只股票时结构扁平
        df = yf.download(ticker, period="max", progress=False)
        
        if not df.empty:
            # 清洗雅虎新版本可能带来的 MultiIndex 表头
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df.reset_index(inplace=True)
            
            # 统一列名大小写，使其与你之前的 ML 习惯完全一致
            # 雅虎返回包含：Date, Open, High, Low, Close, Adj Close, Volume
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f" -> 🎉 成功！共 {len(df)} 行交易日历史。")
            success_count += 1
        else:
            print(" -> ❌ 未获取到数据（可能该标的在雅虎未维护）")
            
    except Exception as e:
        print(f" -> 💥 发生异常：{str(e)}")

print(f"\n==========================================")
print(f"🚀 任务完成！成功实时同步 {success_count} / {len(topic2_tickers)} 只宽基 ETF。")
print(f"📂 数据已安全躺在：`{output_dir}/` 目录下。")
print(f"==========================================")