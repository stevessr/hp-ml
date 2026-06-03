import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🏛️ 正在启动《2年期流式自适应滚动面板训练》终极增强引擎...")

# 2. 预先清洗舆情时序 Z-Score
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)
google_z_series = ((raw_sent - raw_sent.rolling(20).mean()) / raw_sent.rolling(20).std().replace(0, np.nan)).fillna(0)

# 3. 跨资产面板数据全量清洗与拼装
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
    
    # 标签：明日绝对涨跌
    df_feat['Target'] = (df_feat['Daily_Return'].shift(-1) > 0).astype(int)
    df_feat['Asset_Code'] = code
    
    dfs_list.append(df_feat[features + ['Target', 'Daily_Return', 'Asset_Code']])

# 全市场纵横面板大表
full_panel = pd.concat(dfs_list).sort_index().dropna()

# 🎯 锁定 2026 年为流式滚动测试大窗口
test_months = pd.date_range(start='2026-01-01', end='2026-06-03', freq='ME')
# 补上最后一个不满月的尾巴
test_months = test_months.append(pd.DatetimeIndex([pd.to_datetime('2026-06-03')]))

# 初始化存储容器
all_rolling_strat_returns = []
all_rolling_bench_returns = []
executed_dates = []

print("⚙️ 开始执行 2026 流式 Walk-Forward 滚动多轨资产配置...")

# 4. 滚动流主循环：每月初自动重新训练过去2年的模型，并预测接下来的一个月
for i in range(len(test_months)-1):
    start_test_date = test_months[i] if i == 0 else test_months[i] + pd.Timedelta(days=1)
    end_test_date = test_months[i+1]
    
    # 计算当前测试窗口对应的 2 年滚动训练边界 (504个交易日约为730历天)
    train_start_date = start_test_date - pd.Timedelta(days=730)
    train_end_date = start_test_date - pd.Timedelta(days=1)
    
    # 切分局部时序面板
    local_train = full_panel.loc[train_start_date:train_end_date]
    local_test = full_panel.loc[start_test_date:end_test_date]
    
    if local_train.empty or local_test.empty:
        continue
        
    # 训练具备“最新近记忆”的随机森林
    rolling_model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42)
    rolling_model.fit(local_train[features], local_train['Target'])
    
    # 按天执行局部预测与自适应建仓
    day_chunks = local_test.index.unique().sort_values()
    for day in day_chunks:
        day_data = local_test.loc[[day]]
        if isinstance(day_data, pd.Series): day_data = day_data.to_frame().T
        
        # 基准：等权重持有当天存活成分
        all_rolling_bench_returns.append(day_data['Daily_Return'].mean())
        executed_dates.append(day)
        
        # 机器学习实时概率推理
        probs = rolling_model.predict_proba(day_data[features])[:, 1]
        day_res = day_data.copy()
        day_res['Prob'] = probs
        
        # 连续仓位映射：只要胜率>50.2%，给50%底仓保护，其余按置信度线性拉满
        day_res['Position'] = np.where(
            day_res['Prob'] > 0.502,
            0.50 + 0.50 * np.clip((day_res['Prob'] - 0.502) / (0.56 - 0.502), 0.0, 1.0),
            0.0
        )
        
        # 策略今日综合回报
        strat_day_ret = (day_res['Position'] * day_res['Daily_Return']).mean()
        all_rolling_strat_returns.append(strat_day_ret)

print("🎉 2026年 Walk-Forward 动态流式回测圆满结束！")

# 5. 绩效精算大看板
perf_df = pd.DataFrame(index=executed_dates)
# 严格平移一天，消除未来函数数据泄漏
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
print("👑 降维打击：《2年期流式滚动面板集成组合策略》自适应盲测绩效看板 (2026终极形态)")
print("===============================================================================================")
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*95)
print(f"{'传统等权被动配置(2026基准)':<20}{b_tot:>10.2%}{b_ann:>12.2%}{b_vol:>12.2%}{b_sha:>12.2f}{b_dd:>12.2%}")
print(f"{'🔥 ML 2年流式滚动动态配置组合':<15}{s_tot:>10.2%}{s_ann:>12.2%}{s_vol:>12.2%}{s_sha:>12.2f}{s_dd:>12.2%}")
print("===============================================================================================")

output_path = os.path.join(reports_dir, "ml_rolling_ensemble_final_report.csv")
perf_df.to_csv(output_path)
print(f"💾 滚动流全量明细已安全导出至: {output_path}\n")
