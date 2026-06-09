import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler  # 引入工业级动态归一化组件

# ==================================================================================
# 🎛️ 核心超参数控制面板 (滑动窗口与归一化全局控制变量)
# ==================================================================================
TRAIN_LOOKBACK_DAYS = 730   # 1. 机器学习滚动训练的历史回溯窗口天数 (2 年)
PROB_RANK_WINDOW = 20       # 2. AI 胜率概率池的滚动分位数排名回溯窗口 (月频自适应)
TECH_MA_WINDOW = 20         # 3. 本地技术面 SMA 均线的滤波滤波窗口
SENT_MA_WINDOW = 20         # 4. 全局搜索引擎情绪因子 Z-Score 的计算滚动窗口
SENT_BREAK_THRESHOLD = 1.5  # 5. 行为金融学极端泡沫反向熔断阈值

# 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🏛️ 正在启动《全参数控制 + 动态流式归一化》多时段切换交叉回测引擎...")

# 2. 预先在舆情本有时序上利用控制参数算完标准化 Z-Score 
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)

sent_ma = raw_sent.rolling(SENT_MA_WINDOW).mean()
sent_std = raw_sent.rolling(SENT_MA_WINDOW).std().replace(0, np.nan)
google_z_series = ((raw_sent - sent_ma) / sent_std).fillna(0)

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
    df_feat['SMA'] = prices.rolling(TECH_MA_WINDOW).mean()
    
    # 基础特征工程
    raw_tech = (prices / df_feat['SMA'] - 1).fillna(0)
    df_feat['F_Tech'] = ((raw_tech - raw_tech.rolling(TECH_MA_WINDOW).mean()) / raw_tech.rolling(TECH_MA_WINDOW).std().replace(0, np.nan)).fillna(0)
    df_feat['Vol_Lag1'] = df_feat['Daily_Return'].rolling(5).std().shift(1)
    df_feat['Return_Lag1'] = df_feat['Daily_Return'].shift(1)
    df_feat['F_Sent'] = google_z_series.reindex(prices.index, method='ffill').fillna(0)
    df_feat['Raw_Sentiment_Z'] = df_feat['F_Sent']
    
    df_feat['Target'] = (df_feat['Daily_Return'].shift(-1) > 0).astype(int)
    df_feat['Asset_Code'] = code
    
    dfs_list.append(df_feat[features + ['Target', 'Daily_Return', 'Asset_Code', 'Price', 'SMA', 'Raw_Sentiment_Z']].dropna())

full_panel = pd.concat(dfs_list).sort_index()

# 锁定大跨度盲测长周期时间轴：从 2024-11-01 起滚动测试
test_months = pd.date_range(start='2024-11-01', end='2026-06-03', freq='ME')
test_months = test_months.append(pd.DatetimeIndex([pd.to_datetime('2026-06-03')]))

historical_prob_pool = {file.split('_')[0]: [] for file in etf_files}
all_strat_returns = []
all_bench_returns = []
executed_dates = []

print("⚙️ 正在执行大跨度 Walk-Forward 流式复合动态规则测算...")

# 4. 流式滚动主循环
for i in range(len(test_months)-1):
    start_test_date = test_months[i] if i == 0 else test_months[i] + pd.Timedelta(days=1)
    end_test_date = test_months[i+1]
    
    train_start_date = start_test_date - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS)
    train_end_date = start_test_date - pd.Timedelta(days=1)
    
    local_train = full_panel.loc[train_start_date:train_end_date]
    local_test = full_panel.loc[start_test_date:end_test_date]
    
    if local_train.empty or local_test.empty:
        continue
        
    # 🎯 核心升级：执行无未来函数的局部数据归一化管线
    scaler = StandardScaler()
    # 仅在当前的滚动训练集上进行 fit_transform 提取无量纲分布
    X_train_scaled = scaler.fit_transform(local_train[features])
    y_train = local_train['Target']
    
    # 重训当期局部树模型
    rolling_model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42)
    rolling_model.fit(X_train_scaled, y_train)
    
    day_chunks = local_test.index.unique().sort_values()
    for day in day_chunks:
        day_data = local_test.loc[[day]]
        if isinstance(day_data, pd.Series): day_data = day_data.to_frame().T
        
        # 被动静态等权基准
        all_bench_returns.append(day_data['Daily_Return'].mean())
        executed_dates.append(day)
        
        # 🎯 核心升级：利用训练集算出的 Scaler，流式 transform 今日特征行，绝无前瞻偏差
        day_features_scaled = scaler.transform(day_data[features])
        probs = rolling_model.predict_proba(day_features_scaled)[:, 1]
        
        day_positions = []
        for idx, row in day_data.reset_index().iterrows():
            code = row['Asset_Code']
            prob = probs[idx]
            
            pool = historical_prob_pool[code]
            pool.append(prob)
            
            recent_pool = pool[-PROB_RANK_WINDOW:]
            rank = (sum(1 for x in recent_pool if x < prob) / len(recent_pool)) if len(recent_pool) > 5 else 0.5
            
            price_now = float(row['Price'])
            sma_now = float(row['SMA'])
            sent_z_now = float(row['Raw_Sentiment_Z'])
            
            # 非线性多重动态规则决策箱
            if sent_z_now > SENT_BREAK_THRESHOLD:
                pos = 0.10  # 规则一：舆情泡沫反向硬熔断
            elif price_now > sma_now:
                if prob > 0.49:
                    pos = 0.60 + 0.40 * (rank ** 2)  # 规则二：趋势多头底仓 (60%) + 置信度平方加速
                else:
                    pos = 0.30 * rank
            else:
                if prob > 0.51:
                    pos = 0.50 * (rank ** 2)  # 规则三：谨慎小仓位参与超跌反弹
                else:
                    pos = 0.0  # 规则四：绝对看空空仓防御
                    
            day_positions.append(np.clip(pos, 0.0, 1.0))
            
        strat_day_ret = (np.array(day_positions) * day_data['Daily_Return'].values).mean()
        all_strat_returns.append(strat_day_ret)

# 5. 组装基础资产净值大表
perf_df = pd.DataFrame(index=executed_dates)
perf_df['Benchmark'] = all_bench_returns
perf_df['Strategy'] = pd.Series(all_strat_returns, index=executed_dates).shift(1).fillna(0)

# 6. 分时间段（Regimes）绩效解剖
regimes = {
    "全周期跨度 (2024.11 - 2026.06)": (perf_df.index.min(), perf_df.index.max()),
    "阶段一：蓄势筑底期 (2024.11 - 2025.05)": (pd.to_datetime('2024-11-08'), pd.to_datetime('2025-05-31')),
    "阶段二：高波分化期 (2025-06-01 - 2025-12-31)": (pd.to_datetime('2025-06-01'), pd.to_datetime('2025-12-31')),
    "阶段三：单边牛市期 (2026-01-01 - 2026-06-03)": (pd.to_datetime('2026-01-01'), perf_df.index.max())
}

print("\n" + "="*98)
print("👑 归一化升级看板：《9 只中证宽基复合动态规则策略》分阶段自适应检验 (稳健性全面净化)")
print("==================================================================================================")
print(f"{'测试历史区间/市场机制':<32}{'模式':<10}{'总收益率':<10}{'年化收益':<10}{'年化波动':<10}{'夏普比率':<10}{'最大回撤':<10}")
print("-"*98)

for name, (start, end) in regimes.items():
    sub_df = perf_df.loc[start:end]
    if sub_df.empty or len(sub_df) < 5: continue
    
    def get_stats(returns):
        cum_series = (1 + returns).cumprod() - 1
        total_ret = cum_series.iloc[-1]
        ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
        cum_prices = (1 + returns).cumprod()
        max_dd = ((cum_prices - cum_prices.cummax()) / cum_prices.cummax()).min()
        return total_ret, ann_ret, ann_vol, sharpe, max_dd

    b_stats = get_stats(sub_df['Benchmark'])
    s_stats = get_stats(sub_df['Strategy'])
    
    print(f"{name:<30}{'基准':<8}{b_stats[0]:>8.2%}{b_stats[1]:>10.2%}{b_stats[2]:>10.2%}{b_stats[3]:>10.2f}{b_stats[4]:>10.2%}")
    print(f"{'':<30}{'🔥 策略':<7}{s_stats[0]:>8.2%}{s_stats[1]:>10.2%}{s_stats[2]:>10.2%}{s_stats[3]:>10.2f}{s_stats[4]:>10.2%}")
    print("-"*98)

print("==================================================================================================")

# 7. 安全保存全周期明细
output_path = os.path.join(reports_dir, "ml_regime_comparison_report.csv")
perf_df.to_csv(output_path)
print(f"💾 经参数化控制与动态归一化洗礼的全生命周期细节已成功覆盖至：{output_path}\n")
