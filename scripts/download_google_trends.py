import os
import pandas as pd
import time
from pytrends.request import TrendReq

# 你的本地代理
PROXY_URL = 'http://127.0.0.1:7890'

print("🌐 正在通过本地代理连接 Google Trends API (已适配 urllib3 兼容层)...")

# 纯净版初始化：只传 pytrends 官方标准参数
pytrends = TrendReq(
    hl='zh-CN', 
    tz=360, 
    proxies=[PROXY_URL]
)

kw_list = ["沪深 300", "A 股", "开户"]

print(f"📈 开始拉取舆情因子历史：{kw_list}")

try:
    time.sleep(2)
    # 限定中国区数据
    pytrends.build_payload(kw_list, cat=0, timeframe='2024-11-01 2026-06-03', geo='CN')
    
    time.sleep(2)
    trends_df = pytrends.interest_over_time()
    
    if not trends_df.empty:
        if 'isPartial' in trends_df.columns:
            trends_df = trends_df.drop(columns=['isPartial'])
        trends_df.reset_index(inplace=True)
        
        output_dir = "data/alternative_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "google_trends_sentiment.csv")
        trends_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"🎉 舆情数据成功降维打击，已保存至 -> {output_path}")
    else:
        print("❌ 未能获取到数据，返回为空。")

except Exception as e:
    print(f"💥 运行异常：{str(e)}")