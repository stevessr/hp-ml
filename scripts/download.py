import akshare as ak
import pandas as pd

# 以 沪深 300 ETF (510300) 为例
# adjust="qfq" 表示前复权，确保历史价格已剔除分红派息引发的跳空，适合回测
etf_df = ak.fund_etf_hist_em(
    symbol="510300", 
    period="daily", 
    start_date="20100101", 
    end_date="20260603", 
    adjust="qfq"
)

# 重命名列名使其更加直观
etf_df.columns = ['日期', '开盘价', '收盘价', '最高价', '最低价', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

# 保存为 CSV 文件到本地
etf_df.to_csv("510300_沪深 300ETF_历史多年数据.csv", index=False, encoding='utf-8-sig')
print("下载完成！数据样本：")
print(etf_df.head())