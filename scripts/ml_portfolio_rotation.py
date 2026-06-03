import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🏛️ 正在启动《9 只中证宽基多标的 AI 战术风格轮动》多因子增强引擎...")

# 2. 加载舆情特征并进行容错清洗
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)
print(f" └─ 舆情数据列名自适应成功，当前情绪特征列：[{target_sent_col}]")

# 🎯 核心修复：直接在舆情时序上闭环算完 Z-Score，彻底规避跨索引除法陷阱
sent_ma = raw_sent.rolling(20).mean()
sent_std = raw_sent.rolling(20).std().replace(0, np.nan) # 防止除以 0
google_z_series = ((raw_sent - sent_ma) / sent_std).fillna(0)

# 3. 纵向清洗并联立所有 9 只 ETF 的量价数据
etf_files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]
etf_data = {}

for file in etf_files:
    code = file.split('_')[0]
    df = pd.read_csv(os.path.join(data_dir, file), parse_dates=['Date']).set_index('Date')
    close_col = 'Close' if 'Close' in df.columns else df.columns[0]
    
    # 兼容 MultiIndex 表头
    if isinstance(df[close_col], pd.DataFrame):
        etf_data[code] = df[close_col].iloc[:, 0].astype(float)
    else:
        etf_data[code] = df[close_col].astype(float)

# 合并所有 ETF 价格对齐时间轴（拉取 2024 年之后的窗口）
all_prices = pd.DataFrame(etf_data).loc['2024-01-01':]

# 4. 流式构建全资产面板特征库 (Panel Feature Matrix)
print("⚙️ 正在跨资产批量抽取技术特征与横向标签...")
dfs_list = []

for code in all_prices.columns:
    prices = all_prices[code].dropna()
    if len(prices) < 30:  # 过滤掉历史过短的极端样本
        continue
    
    df_feat = pd.DataFrame(index=prices.index)
    df_feat['Price'] = prices
    df_feat['Daily_Return'] = prices.pct_change().fillna(0)
    
    # 特征 A：本土量价技术指标
    ma20 = prices.rolling(20).mean()
    raw_tech = (prices / ma20 - 1).fillna(0)
    tech_std = raw_tech.rolling(20).std().replace(0, np.nan)
    df_feat['F_Tech'] = ((raw_tech - raw_tech.rolling(20).mean()) / tech_std).fillna(0)
    
    df_feat['Vol_Lag1'] = df_feat['Daily_Return'].rolling(5).std().shift(1)
    df_feat['Return_Lag1'] = df_feat['Daily_Return'].shift(1)
    
    # 🎯 核心修复：直接通过前向填充将算好的 Z-Score 映射到 A 股交易日
    df_feat['F_Sent'] = google_z_series.reindex(prices.index, method='ffill').fillna(0)
    
    # 基础标识项与未平移的明天收益率（用于计算横向 Alpha 标签）
    df_feat['Asset_Code'] = code
    df_feat['Next_Return'] = df_feat['Daily_Return'].shift(-1)
    
    # 先剔除单只 ETF 内部由于 rolling 产生的局部 NaN
    local_features = ['Price', 'Daily_Return', 'F_Tech', 'Vol_Lag1', 'Return_Lag1', 'F_Sent', 'Asset_Code', 'Next_Return']
    dfs_list.append(df_feat[local_features].dropna())

# 汇聚成大面板数据
panel_df = pd.concat(dfs_list).sort_index()

# 5. 横向截面动态打标签：计算每一天全市场（9 只 ETF）的平均明日收益
panel_df['Market_Avg_Next_Return'] = panel_df.groupby(panel_df.index)['Next_Return'].transform('mean')
# 标签：明天能否跑赢大盘平均水平
panel_df['Target'] = (panel_df['Next_Return'] > panel_df['Market_Avg_Next_Return']).astype(int)

# 再次全面清除无用尾部数据
panel_df = panel_df.dropna()

features = ['F_Tech', 'F_Sent', 'Return_Lag1', 'Vol_Lag1']

# 6. 划分时序训练集与完全隔离的盲测集
unique_dates = panel_df.index.unique().sort_values()
split_date = unique_dates[int(len(unique_dates) * 0.65)]

train_panel = panel_df.loc[:split_date]
test_panel = panel_df.loc[split_date:]

print(f"📊 面板数据对齐完毕！训练集：{len(train_panel)} 行 | 盲测集：{len(test_panel)} 行")

# 7. 训练抗噪能力极强的随机森林横向分类器
print(f"🤖 正在训练跨资产截面阿尔法模型...")
model = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=4, random_state=42)
model.fit(train_panel[features], train_panel['Target'])

# 8. 在盲测集上执行每日横向战术轮动与风格对冲
print("🏃 正在模拟每日多标的非线性轮动与战术资产配置...")
test_dates = test_panel.index.unique().sort_values()

portfolio_returns = []
benchmark_returns = []

for date in test_dates:
    day_data = test_panel.loc[[date]]
    if day_data.empty: continue
    
    # 提取今日特征，预测明天跑赢大盘的胜率
    day_features = day_data[features]
    probs = model.predict_proba(day_features)[:, 1]
    
    day_results = day_data.copy()
    day_results['Alpha_Prob'] = probs
    
    # 🏎️ 轮动规则：精选看多胜率最高的前 2 只宽基 ETF 进行轮动
    top_assets = day_results.sort_values(by='Alpha_Prob', ascending=False).head(2)
    
    # 基准：等权重买入当天市场上所有的宽基 ETF（代表被动大盘资产配置）
    benchmark_day_ret = day_data['Daily_Return'].mean()
    benchmark_returns.append(benchmark_day_ret)
    
    # 策略收益计算
    strat_day_ret = 0
    allocated_weight = 0.5  # 资金平分两只
    
    for _, row in top_assets.iterrows():
        # 行为金融学安全阀：胜率必须大于 50.2% 才有统计学优势，否则该头寸强制空仓避险
        if row['Alpha_Prob'] > 0.502:
            strat_day_ret += allocated_weight * row['Daily_Return']
        else:
            strat_day_ret += 0
            
    portfolio_returns.append(strat_day_ret)

# 9. 导出最终绩效大表
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
print("👑 终极通关：《中证全量 9 只宽基 ETF 截面战术轮动策略》盲测绩效看板")
print("===============================================================================================")
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*95)
print(f"{'传统被动等权配置 (基准)':<22}{b_tot:>10.2%}{b_ann:>12.2%}{b_vol:>12.2%}{b_sha:>12.2f}{b_dd:>12.2%}")
print(f"{'🔥 ML 决策树截面风格轮动增强':<18}{s_tot:>10.2%}{s_ann:>12.2%}{s_vol:>12.2%}{s_sha:>12.2f}{s_dd:>12.2%}")
print("===============================================================================================")

importances = model.feature_importances_
print("🧠 跨资产通用机器学习决策权重（主导因子说明）:")
for feat, imp in zip(features, importances):
    print(f" ├─ 核心特征：{feat:<12} | 截面分裂贡献度：{imp:.4f}")

perf_df.to_csv(os.path.join(reports_dir, "ml_portfolio_rotation_detail.csv"))
print(f"\n💾 9 只宽基多标的轮动时序与净值曲线数据已成功安全导出至 reports/ 目录。\n")