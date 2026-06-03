import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🏛️ 正在启动《全要素大跨度通用截面 Alpha 排序模型》终极资产配置引擎...")

# 2. 舆情时序标准化
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)
google_z_series = ((raw_sent - raw_sent.rolling(20).mean()) / raw_sent.rolling(20).std().replace(0, np.nan)).fillna(0)

# 3. 跨资产非对称流式特征抽取
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
    
    # 核心转换：存储明天的真实收益率（先不打绝对标签）
    df_feat['Next_Return'] = df_feat['Daily_Return'].shift(-1)
    df_feat['Asset_Code'] = code
    
    dfs_list.append(df_feat[features + ['Next_Return', 'Daily_Return', 'Asset_Code']])

# 汇聚成全局横纵向大面板
full_panel = pd.concat(dfs_list).sort_index().dropna(subset=['Next_Return'])

# 🎯 核心优化点：计算每日全市场的平均回报率，构建【截面相对超额标签】
full_panel['Market_Mean_Next'] = full_panel.groupby(full_panel.index)['Next_Return'].transform('mean')
full_panel['Target'] = (full_panel['Next_Return'] > full_panel['Market_Mean_Next']).astype(int)
full_panel = full_panel.dropna()

# 4. 划分 2025 年底之前为训练集，2026年为纯净盲测集
CUTOFF_DATE = pd.to_datetime('2025-12-31')
panel_train = full_panel[full_panel.index <= CUTOFF_DATE]
panel_test = full_panel[full_panel.index > CUTOFF_DATE]

print(f"📊 截面多因子大面板构建完毕！")
print(f" ├─ 🏛️ 历史训练集规模: {len(panel_train)} 行特征样本")
print(f" └─ 🎯 2026纯净盲测空间: {len(panel_test)} 行资产样本")

# 5. 训练通用横向截面 Alpha 分类器
print("🤖 正在利用全历史截面样本训练通用 Alpha 决策树...")
universal_ranker = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
universal_ranker.fit(panel_train[features], panel_train['Target'])

# 6. 在 2026 盲测长周期内，模拟每日【Top-2 战术强弱轮动配置】
test_dates = panel_test.index.unique().sort_values()
portfolio_returns = []
benchmark_returns = []

print("🏃 正在启动 2026 战术阿尔法横向排序轮动流...")
for date in test_dates:
    day_data = panel_test.loc[[date]]
    if isinstance(day_data, pd.Series): day_data = day_data.to_frame().T
    if day_data.empty or len(day_data) < 2: continue
    
    # 基准：当天存活的所有宽基 ETF 等权重持有回报
    benchmark_returns.append(day_data['Daily_Return'].mean())
    
    # 机器学习横向截面打分
    day_features = day_data[features]
    probs = universal_ranker.predict_proba(day_features)[:, 1]
    
    day_results = day_data.copy()
    day_results['Alpha_Score'] = probs
    
    # 🏎️ 终极配置核心：无视绝对概率偏误，每日强行锁定跑赢胜率最高的 Top 2 成分股满仓
    top_2_assets = day_results.sort_values(by='Alpha_Score', ascending=False).head(2)
    
    # 等权重（各50%）计算策略今日收益
    strat_day_ret = top_2_assets['Daily_Return'].mean()
    portfolio_returns.append(strat_day_ret)

# 7. 终极绩效指标结算
perf_df = pd.DataFrame(index=test_dates[:len(portfolio_returns)])
perf_df['Benchmark'] = benchmark_returns
perf_df['Strategy'] = portfolio_returns

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
print("👑 绝地反击：《全要素通用面板 Alpha 排序轮动组合》终极绩效看板")
print("===============================================================================================")
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*95)
print(f"{'传统等权被动资配(2026基准)':<20}{b_tot:>10.2%}{b_ann:>12.2%}{b_vol:>12.2%}{b_sha:>12.2f}{b_dd:>12.2%}")
print(f"{'🔥 ML 截面阿尔法 Top-2 动态增强':<16}{s_tot:>10.2%}{s_ann:>12.2%}{s_vol:>12.2%}{s_sha:>12.2f}{s_dd:>12.2%}")
print("===============================================================================================")

importances = universal_ranker.feature_importances_
print("🧠 通用截面排序大模型多多因子贡献度分析:")
for feat, imp in zip(features, importances):
    print(f" ├─ 特征名称: {feat:<12} | 截面树分裂贡献度: {imp:.4f}")

perf_df.to_csv(os.path.join(reports_dir, "ml_ultimate_rank_report.csv"))
print(f"\n💾 截面排序轮动终极回测明细已导出至 reports/ml_ultimate_rank_report.csv\n")
