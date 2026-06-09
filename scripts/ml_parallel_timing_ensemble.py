import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🏛️ 正在启动《9 只中证宽基平行时序自适应仓位集成组合》终极量化增强引擎...")

# 2. 预先在舆情本有时序上算完标准化 Z-Score (保护时序大杀器)
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)
google_z_series = ((raw_sent - raw_sent.rolling(20).mean()) / raw_sent.rolling(20).std().replace(0, np.nan)).fillna(0)

# 3. 纵向加载所有 9 只 ETF 的历史行情
etf_files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]
features = ['F_Tech', 'F_Sent', 'Return_Lag1', 'Vol_Lag1']

# 统一划分 2026 年为纯盲测区间
CUTOFF_DATE = pd.to_datetime('2025-12-31')

asset_executed_returns = {}
market_raw_returns = {}

print("⚙️ 正在启动多轨平行机器学习模型流...")
for file in etf_files:
    code = file.split('_')[0]
    df_raw = pd.read_csv(os.path.join(data_dir, file), parse_dates=['Date']).sort_values(by='Date').set_index('Date')
    close_col = 'Close' if 'Close' in df_raw.columns else df_raw.columns[0]
    
    prices = df_raw[close_col].iloc[:, 0] if isinstance(df_raw[close_col], pd.DataFrame) else df_raw[close_col]
    prices = prices.astype(float)
    
    # 抽取单体技术特征
    df_feat = pd.DataFrame(index=prices.index)
    df_feat['Price'] = prices
    df_feat['Daily_Return'] = prices.pct_change().fillna(0)
    
    ma20 = prices.rolling(20).mean()
    raw_tech = (prices / ma20 - 1).fillna(0)
    df_feat['F_Tech'] = ((raw_tech - raw_tech.rolling(20).mean()) / raw_tech.rolling(20).std().replace(0, np.nan)).fillna(0)
    df_feat['Vol_Lag1'] = df_feat['Daily_Return'].rolling(5).std().shift(1)
    df_feat['Return_Lag1'] = df_feat['Daily_Return'].shift(1)
    df_feat['F_Sent'] = google_z_series.reindex(prices.index, method='ffill').fillna(0)
    
    # 标签：明天的绝对涨跌
    df_feat['Target'] = (df_feat['Daily_Return'].shift(-1) > 0).astype(int)
    df_cleaned = df_feat[features + ['Target', 'Daily_Return']].dropna()
    
    df_train = df_cleaned[df_cleaned.index <= CUTOFF_DATE]
    df_test = df_cleaned[df_cleaned.index > CUTOFF_DATE]
    
    if len(df_train) < 50 or df_test.empty: continue
    
    # 4. 独立并行训练随机森林
    model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42)
    model.fit(df_train[features], df_train['Target'])
    
    pred_probs = model.predict_proba(df_test[features])[:, 1]
    
    # 🎯 核心调优：Beta 底仓保护 + Alpha 置信度弹性加速器
    # 只要模型预测明天上涨概率 > 50%，立刻派发 50% 基础底仓，其余 50% 仓位根据置信度在 [50%, 56%] 区间线性拉满
    target_position = np.where(
        pred_probs > 0.50,
        0.50 + 0.50 * np.clip((pred_probs - 0.50) / (0.56 - 0.50), 0.0, 1.0),
        0.0
    )
    
    df_res = pd.DataFrame(index=df_test.index)
    df_res['Daily_Return'] = df_test['Daily_Return']
    df_res['Position'] = target_position
    
    # 严格平移，消除未来函数
    df_res['Executed_Position'] = df_res['Position'].shift(1).fillna(0)
    
    asset_executed_returns[code] = df_res['Executed_Position'] * df_res['Daily_Return']
    market_raw_returns[code] = df_test['Daily_Return']

# 5. 横向联立集成大资金池
test_dates = pd.date_range(start='2026-01-01', end='2026-06-03')
strat_panel = pd.DataFrame(asset_executed_returns).reindex(test_dates).dropna(how='all')
bench_panel = pd.DataFrame(market_raw_returns).reindex(strat_panel.index)

portfolio_strat_returns = strat_panel.mean(axis=1)
portfolio_bench_returns = bench_panel.mean(axis=1)

# 6. 最终累计收益计算
perf_df = pd.DataFrame(index=strat_panel.index)
perf_df['Benchmark'] = portfolio_bench_returns
perf_df['Strategy'] = portfolio_strat_returns

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
print("👑 破茧成蝶：《9 只中证宽基 ETF 平行时序自适应仓位集成组合》加速增强版看板 (2026 盲测)")
print("===============================================================================================")
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*95)
print(f"{'传统等权被动资产配置 (基准)':<20}{b_tot:>10.2%}{b_ann:>12.2%}{b_vol:>12.2%}{b_sha:>12.2f}{b_dd:>12.2%}")
print(f"{'🔥 ML 独立多轨平行择时集成组合':<15}{s_tot:>10.2%}{s_ann:>12.2%}{s_vol:>12.2%}{s_sha:>12.2f}{s_dd:>12.2%}")
print("===============================================================================================")

output_path = os.path.join(reports_dir, "ml_parallel_ensemble_report.csv")
perf_df.to_csv(output_path)
print(f"💾 终极多轨择时集成组合回测明细已成功重新导出至：{output_path}\n")
