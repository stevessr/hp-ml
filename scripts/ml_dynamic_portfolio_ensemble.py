import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🏛️ 正在启动《9 只中证宽基多资产时序集成与动态配置》终极量化增强引擎...")

# 2. 加载并清洗全球舆情特征
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)

# 在舆情本有时序上算完标准化 Z-Score，防范跨索引除法陷阱
google_z_series = ((raw_sent - raw_sent.rolling(20).mean()) / raw_sent.rolling(20).std().replace(0, np.nan)).fillna(0)

# 3. 纵向加载所有 9 只 ETF 的历史行情
etf_files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]
etf_tables = {}

for file in etf_files:
    code = file.split('_')[0]
    df = pd.read_csv(os.path.join(data_dir, file), parse_dates=['Date']).set_index('Date')
    close_col = 'Close' if 'Close' in df.columns else df.columns[0]
    
    # 兼容 MultiIndex
    if isinstance(df[close_col], pd.DataFrame):
        etf_tables[code] = df[[close_col]].iloc[:, 0].to_frame(name='Close')
    else:
        etf_tables[code] = df[[close_col]].rename(columns={close_col: 'Close'})

print(f" └─ 成功收拢 9 只 ETF 原始时序矩阵。开始独立特征训练与盲测...")

# 4. 核心逻辑：为每一只 ETF 独立训练大模型并预测看多概率
all_asset_probs = {}
all_asset_returns = {}

features = ['F_Tech', 'F_Sent', 'Return_Lag1', 'Vol_Lag1']
test_dates = None

for code, df_etf in etf_tables.items():
    # 抽取单体特征
    df_etf['Daily_Return'] = df_etf['Close'].pct_change().fillna(0)
    ma20 = df_etf['Close'].rolling(20).mean()
    raw_tech = (df_etf['Close'] / ma20 - 1).fillna(0)
    df_etf['F_Tech'] = ((raw_tech - raw_tech.rolling(20).mean()) / raw_tech.rolling(20).std().replace(0, np.nan)).fillna(0)
    df_etf['Vol_Lag1'] = df_etf['Daily_Return'].rolling(5).std().shift(1)
    df_etf['Return_Lag1'] = df_etf['Daily_Return'].shift(1)
    
    # 无缝映射时序舆情因子
    df_etf['F_Sent'] = google_z_series.reindex(df_etf.index, method='ffill').fillna(0)
    
    # 独立预测标签：明天自己是否上涨
    df_etf['Target'] = (df_etf['Daily_Return'].shift(-1) > 0).astype(int)
    df_model_data = df_etf[features + ['Target', 'Daily_Return']].dropna().loc['2024-01-01':]
    
    if len(df_model_data) < 50: continue
    
    # 时序严格切分
    split_idx = int(len(df_model_data) * 0.65)
    train_df = df_model_data.iloc[:split_idx]
    test_df = df_model_data.iloc[split_idx:]
    
    # 为当前资产量身定制决策树
    model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=4, random_state=42)
    model.fit(train_df[features], train_df['Target'])
    
    # 产出盲测期的看多概率
    probs = model.predict_proba(test_df[features])[:, 1]
    
    # 存入中央调度仓库
    all_asset_probs[code] = pd.Series(probs, index=test_df.index)
    all_asset_returns[code] = test_df['Daily_Return']
    
    if test_dates is None:
        test_dates = test_df.index
    else:
        test_dates = test_dates.intersection(test_df.index)

# 5. 横向联立对齐盲测窗口的概率大矩阵与收益率大矩阵
test_dates = test_dates.sort_values()
prob_matrix = pd.DataFrame({c: all_asset_probs[c].reindex(test_dates) for c in all_asset_probs.keys()}).fillna(0)
ret_matrix = pd.DataFrame({c: all_asset_returns[c].reindex(test_dates) for c in all_asset_returns.keys()}).fillna(0)

# 6. 模拟多资产投资组合资产配置 (Portfolio Weight Asset Allocation)
portfolio_returns = []
benchmark_returns = []

for date in test_dates:
    day_probs = prob_matrix.loc[date]
    day_rets = ret_matrix.loc[date]
    
    # 传统基准：全市场等权重被动大盘资产配置 (Buy & Hold All)
    benchmark_returns.append(day_rets.mean())
    
    # 💎 动态仓位资产配置组合策略
    # 筛选出大模型认为明天胜率绝对大于安全边际 (52.5%) 的优质资产
    confident_assets = day_probs[day_probs > 0.525]
    
    if not confident_assets.empty:
        # 资金分配：根据大模型给出的信心概率进行 Softmax 或是归一化分配仓位
        weights = confident_assets / confident_assets.sum()
        
        # 策略今日总收益 = 各成分股收益的加权平方和
        strat_day_ret = (weights * day_rets[confident_assets.index]).sum()
    else:
        # 💡 全网防御：当没有任何一个宽基 ETF 能够说服大模型时，整体策略强制 0 仓位空仓，完美避险
        strat_day_ret = 0.0
        
    portfolio_returns.append(strat_day_ret)

# 7. 最终战果精算看板
perf_df = pd.DataFrame(index=test_dates)
perf_df['Benchmark'] = benchmark_returns
perf_df['Strategy'] = portfolio_returns

perf_df['Bench_Cum'] = (1 + perf_df['Benchmark']).cumprod() - 1
perf_df['Strat_Cum'] = (1 + perf_df['Strategy']).cumprod() - 1

def get_metrics(returns, cum_series):
    total_ret = cum_series.iloc[-1]
    ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    cum_prices = (1 + returns).cumprod()
    max_dd = ((cum_prices - cum_prices.cummax()) / cum_prices.cummax()).min()
    return total_ret, ann_ret, ann_vol, sharpe, max_dd

b_tot, b_ann, b_vol, b_sha, b_dd = get_metrics(perf_df['Benchmark'], perf_df['Bench_Cum'])
s_tot, s_ann, s_vol, s_sha, s_dd = get_metrics(perf_df['Strategy'], perf_df['Strat_Cum'])

print("\n" + "="*95)
print("👑 降维打击：《9 只中证宽基 ETF 时序集成与动态仓位投资组合》盲测绩效看板")
print("===============================================================================================")
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*95)
print(f"{'传统等权被动资产配置 (基准)':<20}{b_tot:>10.2%}{b_ann:>12.2%}{b_vol:>12.2%}{b_sha:>12.2f}{b_dd:>12.2%}")
print(f"{'💎 ML 独立时序集成动态配置组合':<16}{s_tot:>10.2%}{s_ann:>12.2%}{s_vol:>12.2%}{s_sha:>12.2f}{s_dd:>12.2%}")
print("===============================================================================================")

perf_df.to_csv(os.path.join(reports_dir, "ml_portfolio_ensemble_final_detail.csv"))
print(f"💾 终极多资产配置细节已成功导出至 reports/ml_portfolio_ensemble_final_detail.csv\n")
