import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🏛️ 正在启动《2 年期流式滚动面板 + 自适应时序分位数仓位控制》终极增强引擎...")

# 2. 预先清洗舆情时序 Z-Score
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)
google_z_series = ((raw_sent - raw_sent.rolling(20).mean()) / raw_sent.rolling(20).std().replace(0, np.nan)).fillna(0)

# 3. 跨资产面板数据拼装
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
    
    ma20 = prices.rolling(20).mean()
    raw_tech = (prices / ma20 - 1).fillna(0)
    df_feat['F_Tech'] = ((raw_tech - raw_tech.rolling(20).mean()) / raw_tech.rolling(20).std().replace(0, np.nan)).fillna(0)
    df_feat['Vol_Lag1'] = df_feat['Daily_Return'].rolling(5).std().shift(1)
    df_feat['Return_Lag1'] = df_feat['Daily_Return'].shift(1)
    df_feat['F_Sent'] = google_z_series.reindex(prices.index, method='ffill').fillna(0)
    
    df_feat['Target'] = (df_feat['Daily_Return'].shift(-1) > 0).astype(int)
    df_feat['Asset_Code'] = code
    
    dfs_list.append(df_feat[features + ['Target', 'Daily_Return', 'Asset_Code']])

full_panel = pd.concat(dfs_list).sort_index().dropna()

# 锁定 2026 年测试大窗口
test_months = pd.date_range(start='2026-01-01', end='2026-06-03', freq='ME')
test_months = test_months.append(pd.DatetimeIndex([pd.to_datetime('2026-06-03')]))

# 建立全局资产概率跟踪池，用于在线计算分位数
historical_prob_pool = {file.split('_')[0]: [] for file in etf_files}

all_rolling_strat_returns = []
all_rolling_bench_returns = []
executed_dates = []

print("⚙️ 开始执行 2026 战术多轨流式分位数自适应资产配置...")

# 4. 滚动流主循环
for i in range(len(test_months)-1):
    start_test_date = test_months[i] if i == 0 else test_months[i] + pd.Timedelta(days=1)
    end_test_date = test_months[i+1]
    
    train_start_date = start_test_date - pd.Timedelta(days=730)
    train_end_date = start_test_date - pd.Timedelta(days=1)
    
    local_train = full_panel.loc[train_start_date:train_end_date]
    local_test = full_panel.loc[start_test_date:end_test_date]
    
    if local_train.empty or local_test.empty:
        continue
        
    rolling_model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42)
    rolling_model.fit(local_train[features], local_train['Target'])
    
    day_chunks = local_test.index.unique().sort_values()
    for day in day_chunks:
        day_data = local_test.loc[[day]]
        if isinstance(day_data, pd.Series): day_data = day_data.to_frame().T
        
        all_rolling_bench_returns.append(day_data['Daily_Return'].mean())
        executed_dates.append(day)
        
        # 模型输出绝对概率
        probs = rolling_model.predict_proba(day_data[features])[:, 1]
        codes = day_data['Asset_Code'].values
        
        day_positions = []
        for code, prob in zip(codes, probs):
            # 将当前概率喂入该资产的历史池子中
            pool = historical_prob_pool[code]
            pool.append(prob)
            
            # 🎯 核心优化：基于过去 20 个观测值的滚动分位数计算排名
            recent_pool = pool[-20:]
            if len(recent_pool) > 5:
                # 计算当前值在过去窗口中的百分位排名 (0.0 ~ 1.0)
                rank = (sum(1 for x in recent_pool if x < prob) / len(recent_pool))
            else:
                rank = 0.5  # 样本不足时中性持仓
            
            # 连续分位数仓位函数：排名越靠前，说明置信度处于近期高点，仓位直接呈指数级/线性拉满
            pos = np.clip((rank - 0.2) / (0.8 - 0.2), 0.0, 1.0)
            day_positions.append(pos)
            
        # 计算策略今日的组合加权收益率
        strat_day_ret = (np.array(day_positions) * day_data['Daily_Return'].values).mean()
        all_rolling_strat_returns.append(strat_day_ret)

print("🎉 2026 年 终极流式分位数回测顺利收官！")

# 5. 绩效精算大看板
perf_df = pd.DataFrame(index=executed_dates)
perf_df['Benchmark'] = all_rolling_bench_returns
perf_df['Strategy'] = pd.Series(all_rolling_strat_returns, index=executed_dates).shift(1).fillna(0)

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
print("👑 破茧成蝶：《2 年期流式滚动面板集成组合策略》终极分位数自适应看板 (2026 完美收官)")
print("===============================================================================================")
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*95)
print(f"{'传统等权被动配置 (2026 基准)':<20}{b_tot:>10.2%}{b_ann:>12.2%}{b_vol:>12.2%}{b_sha:>12.2f}{b_dd:>12.2%}")
print(f"{'🔥 ML 2 年流式滚动分位数配置组合':<15}{s_tot:>10.2%}{s_ann:>12.2%}{s_vol:>12.2%}{s_sha:>12.2f}{s_dd:>12.2%}")
print("===============================================================================================")

output_path = os.path.join(reports_dir, "ml_rolling_ensemble_final_report.csv")
perf_df.to_csv(output_path)
print(f"💾 滚动流全量明细已安全重新覆盖至：{output_path}\n")
