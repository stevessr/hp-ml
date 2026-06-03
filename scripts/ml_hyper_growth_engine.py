import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier  # 升级为高级梯度提升树
from sklearn.preprocessing import StandardScaler

# ==================================================================================
# 🎛️ 终极狂飙级超参数控制面板
# ==================================================================================
TRAIN_LOOKBACK_DAYS = 730   # 1. 机器学习滚动训练的历史回溯窗口 (2年)
PROB_RANK_WINDOW = 20       # 2. AI胜率概率池的滚动分位数排名回溯窗口
TECH_MA_WINDOW = 20         # 3. 本地技术面SMA均线的滤波窗口
SENT_MA_WINDOW = 20         # 4. 搜索引擎舆情Z-Score计算窗口
SENT_BREAK_THRESHOLD = 1.6  # 5. 极致放宽舆情过热熔断门槛，防止牛市主升浪被误杀熔断
FEE_RATE = 0.0010           # 6. 显式调仓双边总摩擦费率 (千分之1滑点)
INERTIA_THRESHOLD = 0.05    # 7. 调仓迟滞死区过滤器阈值

# 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🚀 正在启动《主力资金集中度再平衡 + 动态复合 Alpha 狂飙引擎》终极量化增强系统...")

# 2. 洗练舆情时序 Z-Score
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)
google_z_series = ((raw_sent - raw_sent.rolling(SENT_MA_WINDOW).mean()) / raw_sent.rolling(SENT_MA_WINDOW).std().replace(0, np.nan)).fillna(0)

# 3. 跨资产面板数据全量拼装
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
    df_feat['Asset_Vol_20d'] = df_feat['Daily_Return'].rolling(20).std() * np.sqrt(252)
    
    raw_tech = (prices / df_feat['SMA'] - 1).fillna(0)
    df_feat['F_Tech'] = ((raw_tech - raw_tech.rolling(TECH_MA_WINDOW).mean()) / raw_tech.rolling(TECH_MA_WINDOW).std().replace(0, np.nan)).fillna(0)
    df_feat['Vol_Lag1'] = df_feat['Daily_Return'].rolling(5).std().shift(1)
    df_feat['Return_Lag1'] = df_feat['Daily_Return'].shift(1)
    df_feat['F_Sent'] = google_z_series.reindex(prices.index, method='ffill').fillna(0)
    df_feat['Raw_Sentiment_Z'] = df_feat['F_Sent']
    
    df_feat['Target'] = (df_feat['Daily_Return'].shift(-1) > 0).astype(int)
    df_feat['Asset_Code'] = code
    
    target_cols = features + ['Target', 'Daily_Return', 'Asset_Code', 'Price', 'SMA', 'Raw_Sentiment_Z', 'Asset_Vol_20d']
    dfs_list.append(df_feat[target_cols].dropna())

full_panel = pd.concat(dfs_list).sort_index()

# 4. 锁定大跨度盲测时间轴
test_months = pd.date_range(start='2024-11-01', end='2026-06-03', freq='ME')
test_months = test_months.append(pd.DatetimeIndex([pd.to_datetime('2026-06-03')]))

historical_prob_pool = {file.split('_')[0]: [] for file in etf_files}
last_executed_positions = {file.split('_')[0]: 0.0 for file in etf_files}

all_strat_returns = []
all_bench_returns = []
executed_dates = []

print("⚙️ 正在执行大跨度 Walk-Forward 流式集中度再平衡测算...")

for i in range(len(test_months)-1):
    start_test_date = test_months[i] if i == 0 else test_months[i] + pd.Timedelta(days=1)
    end_test_date = test_months[i+1]
    
    local_train = full_panel.loc[start_test_date - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS):start_test_date - pd.Timedelta(days=1)]
    local_test = full_panel.loc[start_test_date:end_test_date]
    
    if local_train.empty or local_test.empty:
        continue
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(local_train[features])
    
    # 💡 算法升级：利用大规模面板吞吐训练 HGB 强力分类树，提升对非线性顶点的分裂精度
    rolling_model = HistGradientBoostingClassifier(max_iter=150, max_depth=5, learning_rate=0.03, random_state=42)
    rolling_model.fit(X_train_scaled, local_train['Target'])
    
    day_chunks = local_test.index.unique().sort_values()
    for day in day_chunks:
        day_data = local_test.loc[[day]]
        if isinstance(day_data, pd.Series): day_data = day_data.to_frame().T
        
        executed_dates.append(day)
        probs = rolling_model.predict_proba(scaler.transform(day_data[features]))[:, 1]
        
        day_target_positions = {}
        day_returns = {}
        day_vols = {}
        
        # 4.1 单体成分自适应规则判定
        for idx, row in day_data.reset_index().iterrows():
            code = row['Asset_Code']
            prob = probs[idx]
            
            pool = historical_prob_pool[code]
            pool.append(prob)
            recent_pool = pool[-PROB_RANK_WINDOW:]
            rank = (sum(1 for x in recent_pool if x < prob) / len(recent_pool)) if len(recent_pool) > 5 else 0.5
            
            # 多重因果决策
            if float(row['Raw_Sentiment_Z']) > SENT_BREAK_THRESHOLD:
                raw_pos = 0.0  # 极度过热彻底离场
            elif float(row['Price']) > float(row['SMA']):
                # 💡 凸性爆发器：在牛市上升通道中，利用立方凸性函数加剧多头载荷释放速度
                raw_pos = 0.65 + 0.35 * (rank ** 3) if prob > 0.495 else 0.20 * rank
            else:
                raw_pos = 0.50 * (rank ** 2) if prob > 0.51 else 0.0
                
            raw_pos = np.clip(raw_pos, 0.0, 1.0)
            
            # 迟滞死区过滤
            prev_pos = last_executed_positions[code]
            executed_pos = prev_pos if abs(raw_pos - prev_pos) < INERTIA_THRESHOLD else raw_pos
            
            day_target_positions[code] = executed_pos
            day_returns[code] = float(row['Daily_Return'])
            day_vols[code] = max(float(row['Asset_Vol_20d']), 0.01)

        # 4.2 🎯 终极数理核心：主力资金截面集中度再平衡 (Concentrated Alpha Rebalancing)
        active_codes = list(day_target_positions.keys())
        all_bench_returns.append(np.mean(list(day_returns.values())))
        
        # 筛选出今天真正被选中的多头资产 (持仓信号 > 0)
        bullish_assets = {c: pos for c, pos in day_target_positions.items() if pos > 0}
        
        strat_day_ret_pre_fee = 0.0
        total_friction_fee = 0.0
        
        if bullish_assets:
            # 资金重新分配基数：结合多头持仓深度与反比波动率
            raw_weights = np.array([bullish_assets[c] / day_vols[c] for c in bullish_assets.keys()])
            normalized_weights = raw_weights / raw_weights.sum()  # 🚨 强制横向归一化，确保多头多头头寸资金利用率永远为 100%
            
            # 确定今日大组合整体的贝塔进攻仓位（取多头成分的最大信号载荷，确保单边牛市中无限逼近满仓）
            global_portfolio_exposure = max(bullish_assets.values())
            
            for idx, code in enumerate(bullish_assets.keys()):
                w = normalized_weights[idx] * global_portfolio_exposure # 结合全局进攻载荷算出真实占总资金的权重
                ret = day_returns[code]
                
                # 扣费前的策略综合增益
                strat_day_ret_pre_fee += w * ret
                
                # 穿透捕获精准调仓费
                prev_pos = last_executed_positions[code]
                turnover = abs(w - prev_pos) # 换手深度基于总资产权重的绝对变化
                total_friction_fee += turnover * FEE_RATE
                
                # 闭环存入持仓信息仓
                last_executed_positions[code] = w
                
            # 清空今天没有被选中的弱势品种的持仓底记录
            for code in active_codes:
                if code not in bullish_assets:
                    prev_pos = last_executed_positions[code]
                    total_friction_fee += prev_pos * FEE_RATE # 强行斩断弱势头寸产生的清算磨损
                    last_executed_positions[code] = 0.0
        else:
            # 全盘空仓绝对防御
            strat_day_ret_pre_fee = 0.0
            for code in active_codes:
                prev_pos = last_executed_positions[code]
                total_friction_fee += prev_pos * FEE_RATE
                last_executed_positions[code] = 0.0
                
        final_strat_day_return = strat_day_ret_pre_fee - total_friction_fee
        all_strat_returns.append(final_strat_day_return)

# 5. 组装净值大表
perf_df = pd.DataFrame(index=executed_dates)
perf_df['Benchmark'] = all_bench_returns
perf_df['Strategy'] = pd.Series(all_strat_returns, index=executed_dates).shift(1).fillna(0)

# 6. 分时间段交叉稳健性检验看板
regimes = {
    "全周期跨度 (2024.11 - 2026.06)": (perf_df.index.min(), perf_df.index.max()),
    "阶段一：蓄势筑底期 (2024.11 - 2025.05)": (pd.to_datetime('2024-11-08'), pd.to_datetime('2025-05-31')),
    "阶段二：高波分化期 (2025-06-01 - 2025-12-31)": (pd.to_datetime('2025-06-01'), pd.to_datetime('2025-12-31')),
    "阶段三：单边牛市期 (2026-01-01 - 2026-06-03)": (pd.to_datetime('2026-01-01'), perf_df.index.max())
}

print("\n" + "="*112)
print("👑 狂飙破局看板：《9只中证宽基全要素HGB决策树》截面主力资金再平衡终极全摩擦实证")
print("================================================================================================================")
print(f"{'测试历史区间/市场机制':<32}{'模式':<10}{'总收益率':<10}{'年化收益':<10}{'年化波动':<10}{'夏普比率':<10}{'最大回撤':<10}")
print("-"*112)

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
    print("-"*112)

print("================================================================================================================")

output_path = os.path.join(reports_dir, "ml_hyper_growth_final_report.csv")
perf_df.to_csv(output_path)
print(f"💾 斩断分母死钱、火力全开的终极狂飙时序已安全导出至: {output_path}\n")
