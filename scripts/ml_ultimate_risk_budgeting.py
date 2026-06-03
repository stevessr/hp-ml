import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ==================================================================================
# 🎛️ 工业级终极超参数与摩擦控制面板
# ==================================================================================
TRAIN_LOOKBACK_DAYS = 730   # 1. 机器学习滚动训练的历史回溯窗口 (2年)
PROB_RANK_WINDOW = 20       # 2. AI胜率概率池的滚动分位数排名回溯窗口
TECH_MA_WINDOW = 20         # 3. 本地技术面SMA均线的滤波窗口
SENT_MA_WINDOW = 20         # 4. 搜索引擎舆情Z-Score计算窗口
SENT_BREAK_THRESHOLD = 1.5  # 5. 行为金融学极端过热熔断阈值

# 🚀 压轴工业参数
FEE_RATE = 0.0010           # 6. 显式调仓双边总摩擦费率 (千分之1，真实覆盖佣金+过户费+滑点)
INERTIA_THRESHOLD = 0.05    # 7. 调仓迟滞死区过滤器阈值 (低于5%的微调仓直接强制冻结，杜绝频繁摩擦)

# 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🏛️ 正在启动《显式扣费摩擦 + 动态风险预算控制》终极量化资产配置引擎...")

# 2. 预先清洗全局舆情 Z-Score
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)

sent_ma = raw_sent.rolling(SENT_MA_WINDOW).mean()
sent_std = raw_sent.rolling(SENT_MA_WINDOW).std().replace(0, np.nan)
google_z_series = ((raw_sent - sent_ma) / sent_std).fillna(0)

# 3. 跨资产面板数据全量加载与资产特有指标精算
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
    
    # 动态滚动计算当前成分资产的【滚动20天历史年化波动率】，用于后续风险平价反比加权
    df_feat['Asset_Vol_20d'] = df_feat['Daily_Return'].rolling(20).std() * np.sqrt(252)
    
    raw_tech = (prices / df_feat['SMA'] - 1).fillna(0)
    df_feat['F_Tech'] = ((raw_tech - raw_tech.rolling(TECH_MA_WINDOW).mean()) / raw_tech.rolling(TECH_MA_WINDOW).std().replace(0, np.nan)).fillna(0)
    df_feat['Vol_Lag1'] = df_feat['Daily_Return'].rolling(5).std().shift(1)
    df_feat['Return_Lag1'] = df_feat['Daily_Return'].shift(1)
    df_feat['F_Sent'] = google_z_series.reindex(prices.index, method='ffill').fillna(0)
    df_feat['Raw_Sentiment_Z'] = df_feat['F_Sent']
    
    df_feat['Target'] = (df_feat['Daily_Return'].shift(-1) > 0).astype(int)
    df_feat['Asset_Code'] = code
    
    # 提取多资产所需的独立因子矩阵
    target_cols = features + ['Target', 'Daily_Return', 'Asset_Code', 'Price', 'SMA', 'Raw_Sentiment_Z', 'Asset_Vol_20d']
    dfs_list.append(df_feat[target_cols].dropna())

full_panel = pd.concat(dfs_list).sort_index()

# 4. 滚动划分时间轴进行 Walk-Forward 测试
test_months = pd.date_range(start='2024-11-01', end='2026-06-03', freq='ME')
test_months = test_months.append(pd.DatetimeIndex([pd.to_datetime('2026-06-03')]))

historical_prob_pool = {file.split('_')[0]: [] for file in etf_files}

# 终极追溯：记录上一交易日【各资产真实的已执行持仓仓位】，用于精确捕捉调仓换手率扣费
last_executed_positions = {file.split('_')[0]: 0.0 for file in etf_files}

all_strat_returns = []
all_bench_returns = []
executed_dates = []

print("⚙️ 开始流式执行【风险预算加权 + 换手穿透扣费 + 死区过滤】终极测算模型...")

for i in range(len(test_months)-1):
    start_test_date = test_months[i] if i == 0 else test_months[i] + pd.Timedelta(days=1)
    end_test_date = test_months[i+1]
    
    local_train = full_panel.loc[start_test_date - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS):start_test_date - pd.Timedelta(days=1)]
    local_test = full_panel.loc[start_test_date:end_test_date]
    
    if local_train.empty or local_test.empty:
        continue
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(local_train[features])
    rolling_model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42)
    rolling_model.fit(X_train_scaled, local_train['Target'])
    
    day_chunks = local_test.index.unique().sort_values()
    for day in day_chunks:
        day_data = local_test.loc[[day]]
        if isinstance(day_data, pd.Series): day_data = day_data.to_frame().T
        
        executed_dates.append(day)
        probs = rolling_model.predict_proba(scaler.transform(day_data[features]))[:, 1]
        
        day_raw_positions = {}
        day_vols = {}
        day_returns = {}
        
        # 4.1 单体资产级动态规则与迟滞死区过滤
        for idx, row in day_data.reset_index().iterrows():
            code = row['Asset_Code']
            prob = probs[idx]
            
            pool = historical_prob_pool[code]
            pool.append(prob)
            recent_pool = pool[-PROB_RANK_WINDOW:]
            rank = (sum(1 for x in recent_pool if x < prob) / len(recent_pool)) if len(recent_pool) > 5 else 0.5
            
            # 多重复合规则复合规则控制箱
            if float(row['Raw_Sentiment_Z']) > SENT_BREAK_THRESHOLD:
                target_pos = 0.10
            elif float(row['Price']) > float(row['SMA']):
                target_pos = 0.60 + 0.40 * (rank ** 2) if prob > 0.49 else 0.30 * rank
            else:
                target_pos = 0.50 * (rank ** 2) if prob > 0.51 else 0.0
                
            target_pos = np.clip(target_pos, 0.0, 1.0)
            
            # 🎯 核心机制升级：调仓迟滞死区过滤器
            prev_pos = last_executed_positions[code]
            if abs(target_pos - prev_pos) < INERTIA_THRESHOLD:
                # 调仓幅度低于死区门槛，模型强制冻结动作，强制沿用昨日持仓，极大扼杀摩擦损耗
                executed_pos = prev_pos
            else:
                executed_pos = target_pos
                
            day_raw_positions[code] = executed_pos
            day_vols[code] = float(row['Asset_Vol_20d'])
            day_returns[code] = float(row['Daily_Return'])

        # 4.2 组合组合级动态自适应风险预算加权 (Inverse Volatility Portfolio Rebalancing)
        active_codes = list(day_raw_positions.keys())
        vols_array = np.array([day_vols[c] for c in active_codes])
        # 替换零或极端微小的波动率防止除零
        vols_array = np.where(vols_array <= 0, 0.01, vols_array)
        
        # 数学底盘：计算波动率倒数作为分配基础分配权重
        inv_vols = 1.0 / vols_array
        portfolio_weights = inv_vols / inv_vols.sum()  # 风险预算比例归一化
        
        # 计算传统等权配置基准回报
        all_bench_returns.append(np.mean(list(day_returns.values())))
        
        # 4.3 精准穿透换手率，强行扣除交易摩擦成本
        strat_day_ret_pre_fee = 0.0
        total_friction_fee = 0.0
        
        for idx, code in enumerate(active_codes):
            w = portfolio_weights[idx]  # 获得该ETF今天的资产配置分配权重
            pos = day_raw_positions[code]
            ret = day_returns[code]
            
            # 计算成分股层面的今日真实投资增益贡献
            strat_day_ret_pre_fee += w * pos * ret
            
            # 精准抓取换手率绝对变化
            prev_pos = last_executed_positions[code]
            turnover = abs(pos - prev_pos)
            
            # 扣除摩擦费用 = 调仓换手深度 * 资产配置权重 * 实盘总费率
            total_friction_fee += turnover * w * FEE_RATE
            
            # 闭环更新中央持仓数据仓库，供下一个交易日比对
            last_executed_positions[code] = pos
            
        # 策略今日绝对净收益 = 策略税前理论收益 - 调仓换手穿透摩擦扣费
        final_strat_day_return = strat_day_ret_pre_fee - total_friction_fee
        all_strat_returns.append(final_strat_day_return)

# 5. 拼装回测绩效大底表
perf_df = pd.DataFrame(index=executed_dates)
perf_df['Benchmark'] = all_bench_returns
perf_df['Strategy'] = pd.Series(all_strat_returns, index=executed_dates).shift(1).fillna(0)

# 6. 分时间段（Regimes）终极全摩擦稳健性检验
regimes = {
    "全周期跨度 (2024.11 - 2026.06)": (perf_df.index.min(), perf_df.index.max()),
    "阶段一：蓄势筑底期 (2024.11 - 2025.05)": (pd.to_datetime('2024-11-08'), pd.to_datetime('2025-05-31')),
    "阶段二：高波分化期 (2025-06-01 - 2025-12-31)": (pd.to_datetime('2025-06-01'), pd.to_datetime('2025-12-31')),
    "阶段三：单边牛市期 (2026-01-01 - 2026-06-03)": (pd.to_datetime('2026-01-01'), perf_df.index.max())
}

print("\n" + "="*112)
print("👑 终极通关看板：《9只中证宽基复合动态规则策略》扣除全额交易摩擦 + 风险预算反比波动率组合稳健性检验")
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

output_path = os.path.join(reports_dir, "ml_ultimate_risk_budgeting_report.csv")
perf_df.to_csv(output_path)
print(f"💾 扣除真实交易摩擦、经风险平价洗礼的全生命周期核心数据已安全导出至: {output_path}\n")
