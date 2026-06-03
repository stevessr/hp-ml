import os
import pandas as pd
import time
from pytrends.request import TrendReq

PROXY_URL = 'http://127.0.0.1:7890'

print("🌐 正在初始化具备抗限流能力的 Google Trends 客户端...")

# 🎯 核心修复：注入 retries（重试次数）和 backoff_factor（退避系数）
# status_forcelist=[429] 意思是只有遇到 429 时才触发强制等待重试
pytrends = TrendReq(
    hl='zh-CN', 
    tz=360, 
    proxies=[PROXY_URL],
    retries=5,              # 最多重试 5 次
    backoff_factor=10,      # 每次重试等待时间指数级拉长 (10s, 20s, 40s...)
)

kw_list = ["沪深 300", "A 股", "开户"] # 缩减关键词数量，降低被风控概率

print(f"📈 开始拉取舆情因子历史：{kw_list}")

try:
    # 适当分步进行，加一点点硬性睡眠防御
    time.sleep(2)
    
    # geo='CN' 限定中国区
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
        print(f"🎉 舆情数据抗压拉取成功 -> {output_path}")
    else:
        print("❌ 未能获取到数据，返回为空。")

except Exception as e:
    print(f"💥 依然触发风控。终极玄学解决方案提示：请在你的代理软件（如 Clash/mihomo）中切换一个冷门的节点（如更换到别国或干净的冷门原生 IP 节点）再运行。")
    print(f"错误详情：{str(e)}")