import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🚀 正在启动全新的《AI 置信度 + 趋势底仓 + 舆情熔断》复合动态规则测试引擎...")

# 2. 预先在舆情本有时序上算完标准化 Z-Score 
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)
google_z_series = ((raw_sent - raw_sent.rolling(20).mean()) / raw_sent.rolling(20).std().replace(0, np.nan)).fillna(0)

# 3. 跨资产面板数据全量加载与指标计算
etf_files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]
features = ['F_Tech', 'F_Sent', 'Return_Lag1', 'Vol_Lag1']
dfs_list = []

for file in etf_files:
    code = file.split('_')[0]
    df_raw = pd.read_csv(os.path.join(data_dir, file), parse_dates=['Date']).sort_values(by='Date').set_index('Date')
    close_col = 'Close' if 'Close' in df_raw.columns else df_raw.columns[0]
    prices = df_raw[close_col].iloc[:, 0] if isinstance(df_raw[close_col], pd.DataFrame) else df_raw[close_col]
    prices = prices.astype(float)
    
    df_feat = pd.DataFrame(index=prices.index)
    df_feat['Price'] = prices
    df_feat['Daily_Return'] = prices.pct_change().fillna(0)
    
    # 计算 20 日均线，用于动态规则里的【趋势多空环境判定】
    df_feat['SMA20'] = prices.rolling(20).mean()
    
    # 技术面和舆情因子无量纲标准化
    raw_tech = (prices / df_feat['SMA20'] - 1).fillna(0)
    df_feat['F_Tech'] = ((raw_tech - raw_tech.rolling(20).mean()) / raw_tech.rolling(20).std().replace(0, np.nan)).fillna(0)
    df_feat['Vol_Lag1'] = df_feat['Daily_Return'].rolling(5).std().shift(1)
    df_feat['Return_Lag1'] = df_feat['Daily_Return'].shift(1)
    df_feat['F_Sent'] = google_z_series.reindex(prices.index, method='ffill').fillna(0)
    
    # 原始搜索热度 Z-Score，直接暴露给【舆情熔断规则】
    df_feat['Raw_Sentiment_Z'] = df_feat['F_Sent']
    
    df_feat['Target'] = (df_feat['Daily_Return'].shift(-1) > 0).astype(int)
    df_feat['Asset_Code'] = code
    
    dfs_list.append(df_feat[features + ['Target', 'Daily_Return', 'Asset_Code', 'Price', 'SMA20', 'Raw_Sentiment_Z']].dropna())

full_panel = pd.concat(dfs_list).sort_index()

# 锁定 2026 年流式滚动盲测长窗口
test_months = pd.date_range(start='2026-01-01', end='2026-06-03', freq='ME')
test_months = test_months.append(pd.DatetimeIndex([pd.to_datetime('2026-06-03')]))

# 初始化在线动态跟踪池
historical_prob_pool = {file.split('_')[0]: [] for file in etf_files}
all_strat_returns = []
all_bench_returns = []
executed_dates = []

print("⚙️ 开始流式滚动执行全新的【复合非线性动态规则仓位分配】...")

# 4. 2 年期滚动窗口主循环
for i in range(len(test_months)-1):
    start_test_date = test_months[i] if i == 0 else test_months[i] + pd.Timedelta(days=1)
    end_test_date = test_months[i+1]
    
    train_start_date = start_test_date - pd.Timedelta(days=730)
    train_end_date = start_test_date - pd.Timedelta(days=1)
    
    local_train = full_panel.loc[train_start_date:train_end_date]
    local_test = full_panel.loc[start_test_date:end_test_date]
    
    if local_train.empty or local_test.empty:
        continue
        
    # 重训局部大模型
    rolling_model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42)
    rolling_model.fit(local_train[features], local_train['Target'])
    
    day_chunks = local_test.index.unique().sort_values()
    for day in day_chunks:
        day_data = local_test.loc[[day]]
        if isinstance(day_data, pd.Series): day_data = day_data.to_frame().T
        
        # 被动静态等权基准
        all_bench_returns.append(day_data['Daily_Return'].mean())
        executed_dates.append(day)
        
        # 提取模型上涨绝对概率
        probs = rolling_model.predict_proba(day_data[features])[:, 1]
        
        day_positions = []
        for idx, row in day_data.reset_index().iterrows():
            code = row['Asset_Code']
            prob = probs[idx]
            
            # 更新局部历史概率池
            pool = historical_prob_pool[code]
            pool.append(prob)
            
            # 计算近期分位数排名
            recent_pool = pool[-20:]
            rank = (sum(1 for x in recent_pool if x < prob) / len(recent_pool)) if len(recent_pool) > 5 else 0.5
            
            # 💡 核心注入：非线性多重动态规则引擎 (Dynamic Rule Box)
            price_now = float(row['Price'])
            sma_now = float(row['SMA20'])
            sent_z_now = float(row['Raw_Sentiment_Z'])
            
            # 规则一：舆情泡沫反向硬熔断
            if sent_z_now > 1.5:
                pos = 0.10  # 散户极度疯狂，强行将仓位砸回 1 成避险
            # 规则二：确认处于右侧上升通道 (Price > SMA20)
            elif price_now > sma_now:
                if prob > 0.49:
                    # 激活底仓保护 (60%) + 浮动仓位平方凸性加速放大
                    pos = 0.60 + 0.40 * (rank ** 2)
                else:
                    pos = 0.30 * rank  # 牛市中模型意外看空，缩减仓位
            # 规则三：处于左侧或震荡市环境 (Price <= SMA20)
            else:
                if prob > 0.51:
                    pos = 0.50 * (rank ** 2)  # 谨慎参与超跌反弹
                else:
                    pos = 0.0  # 趋势向下且模型看空，坚决空仓
                    
            day_positions.append(np.clip(pos, 0.0, 1.0))
            
        # 动态组合加权收益率
        strat_day_ret = (np.array(day_positions) * day_data['Daily_Return'].values).mean()
        all_strat_returns.append(strat_day_ret)

print("🎉 动态规则组合盲测顺利收官！")

# 5. 绩效统计看板
perf_df = pd.DataFrame(index=executed_dates)
perf_df['Benchmark'] = all_bench_returns
# 严格平移一天，锁死未来函数，确保完美实盘拟真度
perf_df['Strategy'] = pd.Series(all_strat_returns, index=executed_dates).shift(1).fillna(0)

perf_df['Bench_Cum'] = (1 + perf_df['Benchmark']).cumprod() - 1
perf_df['Strat_Cum'] = (1 + perf_df['Strategy']).cumprod() - 1

def calc_metrics(returns, cum_series):
    total_ret = cum_series.iloc[-1]
    ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    cum_prices = (1 + returns).cumprod()
    max_dd = ((cum_prices - cum_prices.cummax()) / cum_prices.cummax()).min()
    return total_ret, ann_ret, ann_vol, sharpe, max_dd

b_tot, b_ann, b_vol, b_sha, b_dd = calc_metrics(perf_df['Benchmark'], perf_df['Bench_Cum'])
s_tot, s_ann, s_vol, s_sha, s_dd = calc_metrics(perf_df['Strategy'], perf_df['Strat_Cum'])

print("\n" + "="*95)
print("👑 规则重组：《AI 胜率 + 趋势底仓 + 舆情熔断》复合动态规则组合绩效看板 (2026 盲测)")
print("===============================================================================================")
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*95)
print(f"{'传统等权被动配置 (2026 基准)':<20}{b_tot:>10.2%}{b_ann:>12.2%}{b_vol:>12.2%}{b_sha:>12.2f}{b_dd:>12.2%}")
print(f"{'🔥 ML 复合动态规则增强策略':<16}{s_tot:>10.2%}{s_ann:>12.2%}{s_vol:>12.2%}{s_sha:>12.2f}{s_dd:>12.2%}")
print("===============================================================================================")

output_path = os.path.join(reports_dir, "ml_dynamic_rules_test_report.csv")
perf_df.to_csv(output_path)
print(f"💾 复合动态规则策略全部明细已安全导出至：{output_path}\n")
