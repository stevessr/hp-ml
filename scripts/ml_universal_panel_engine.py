import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 路径设置
data_dir = "data/topic2_broad_base"
trends_path = "data/alternative_data/google_trends_sentiment.csv"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

print("🏛️ 正在启动《全要素大跨度通用面板 Alpha 模型》终极量化资产配置引擎...")

# 2. 加载并预先在整体时序上清洗舆情 Z-Score
trends_df = pd.read_csv(trends_path, parse_dates=['date']).set_index('date')
sent_candidates = [col for col in trends_df.columns if 'A' in col or '股' in col or '开户' in col]
target_sent_col = sent_candidates[0] if sent_candidates else trends_df.columns[0]
raw_sent = trends_df[target_sent_col].fillna(0)
google_z_series = ((raw_sent - raw_sent.rolling(20).mean()) / raw_sent.rolling(20).std().replace(0, np.nan)).fillna(0)

# 3. 动态加载 9 只 ETF 建立通用全量大面板
etf_files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]
features = ['F_Tech', 'F_Sent', 'Return_Lag1', 'Vol_Lag1']

train_frames = []
test_frames = []

# 🎯 设定全市场统一的盲测历史分水岭 (留出2026整年作为高纯度盲测)
CUTOFF_DATE = pd.to_datetime('2025-12-31')

print("⚙️ 正在执行流式跨资产非对称全生命周期特征抽取...")
for file in etf_files:
    code = file.split('_')[0]
    df_raw = pd.read_csv(os.path.join(data_dir, file), parse_dates=['Date']).sort_values(by='Date').set_index('Date')
    close_col = 'Close' if 'Close' in df_raw.columns else df_raw.columns[0]
    
    # 兼容 MultiIndex 表头
    prices = df_raw[close_col].iloc[:, 0] if isinstance(df_raw[close_col], pd.DataFrame) else df_raw[close_col]
    prices = prices.astype(float)
    
    # 基础特征工程
    df_feat = pd.DataFrame(index=prices.index)
    df_feat['Price'] = prices
    df_feat['Daily_Return'] = prices.pct_change().fillna(0)
    
    ma20 = prices.rolling(20).mean()
    raw_tech = (prices / ma20 - 1).fillna(0)
    df_feat['F_Tech'] = ((raw_tech - raw_tech.rolling(20).mean()) / raw_tech.rolling(20).std().replace(0, np.nan)).fillna(0)
    df_feat['Vol_Lag1'] = df_feat['Daily_Return'].rolling(5).std().shift(1)
    df_feat['Return_Lag1'] = df_feat['Daily_Return'].shift(1)
    df_feat['F_Sent'] = google_z_series.reindex(prices.index, method='ffill').fillna(0)
    
    # 标签：明日是否绝对上涨
    df_feat['Target'] = (df_feat['Daily_Return'].shift(-1) > 0).astype(int)
    df_feat['Asset_Code'] = code
    
    df_cleaned = df_feat[features + ['Target', 'Daily_Return', 'Asset_Code']].dropna()
    
    # 按照统一的日历时间切分训练集与测试集
    df_train = df_cleaned[df_cleaned.index <= CUTOFF_DATE]
    df_test = df_cleaned[df_cleaned.index > CUTOFF_DATE]
    
    if not df_train.empty:
        train_frames.append(df_train)
    if not df_test.empty:
        test_frames.append(df_test)

# 4. 汇聚横纵向大面板矩阵
panel_train = pd.concat(train_frames).sort_index()
panel_test = pd.concat(test_frames).sort_index()

print(f"📊 大数据面板构建完毕！")
print(f" ├─ 🏛️ 历史训练吞吐量 (包含多轮完整牛熊): {len(panel_train)} 行特征样本")
print(f" └─ 🎯 独立高纯度盲测空间 (2026全貌行情): {len(panel_test)} 行资产样本")

# 5. 训练通用阿尔法决策树模型
print("🤖 正在利用10年跨度全样本训练通用阿尔法模型...")
universal_model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
universal_model.fit(panel_train[features], panel_train['Target'])

# 6. 在 2026 盲测长周期内，模拟每日动态资产配置组合
test_dates = panel_test.index.unique().sort_values()
portfolio_returns = []
benchmark_returns = []

print("🏃 正在启动 2026 战术组合动态资金配置流...")
for date in test_dates:
    # 核心自适应：捞出今天在市场上正在交易的所有资产（管你上市了10年还是10天）
    day_data = panel_test.loc[[date]]
    if isinstance(day_data, pd.Series): day_data = day_data.to_frame().T
    
    # 传统被动基准：今日市场上所有存活宽基 ETF 的等权被动持有回报
    benchmark_returns.append(day_data['Daily_Return'].mean())
    
    # 机器学习动态推理
    day_features = day_data[features]
    probs = universal_model.predict_proba(day_features)[:, 1]
    
    day_results = day_data.copy()
    day_results['Alpha_Prob'] = probs
    
    # 🌟 动态资配法则：筛选出看多置信度绝对过硬（>51.5%）的成分进行资金池重组
    active_assets = day_results[day_results['Alpha_Prob'] > 0.515]
    
    if not active_assets.empty:
        # 信心权重分配 (置信度越高，分到的资产仓位权重越大)
        raw_weights = active_assets['Alpha_Prob'] - 0.50
        weights = raw_weights / raw_weights.sum()
        
        # 策略今日收益
        strat_day_ret = (weights * active_assets['Daily_Return']).sum()
    else:
        # 舆情与技术共振冰点：全盘看空，触发跨资产全天候 0 仓位离场，绝对防御
        strat_day_ret = 0.0
        
    portfolio_returns.append(strat_day_ret)

# 7. 终极绩效指标结算
perf_df = pd.DataFrame(index=test_dates)
perf_df['Benchmark'] = benchmark_returns
perf_df['Strategy'] = portfolio_returns

perf_df['Bench_Cum'] = (1 + perf_df['Benchmark']).cumprod() - 1
perf_df['Strat_Cum'] = (1 + perf_df['Strategy']).cumprod() - 1

def calc_metrics(returns, cum_series):
    total_ret = cum_series.iloc[-1]
    # 2026 盲测跨越了近半年的真实真实时间
    ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    cum_prices = (1 + returns).cumprod()
    max_dd = ((cum_prices - cum_prices.cummax()) / cum_prices.cummax()).min()
    return total_ret, ann_ret, ann_vol, sharpe, max_dd

b_tot, b_ann, b_vol, b_sha, b_dd = calc_metrics(perf_df['Benchmark'], perf_df['Bench_Cum'])
s_tot, s_ann, s_vol, s_sha, s_dd = calc_metrics(perf_df['Strategy'], perf_df['Strat_Cum'])

print("\n" + "="*95)
print("👑 降维打击：《9只中证宽基 ETF 通用面板时序集成与动态配置组合》终极绩效看板")
print("===============================================================================================")
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*95)
print(f"{'传统等权被动资配(2026基准)':<20}{b_tot:>10.2%}{b_ann:>12.2%}{b_vol:>12.2%}{b_sha:>12.2f}{b_dd:>12.2%}")
print(f"{'💎 ML 全要素面板自适应动态资产组合':<15}{s_tot:>10.2%}{s_ann:>12.2%}{s_vol:>12.2%}{s_sha:>12.2f}{s_dd:>12.2%}")
print("===============================================================================================")

importances = universal_model.feature_importances_
print("🧠 10年跨度通用阿尔法大模型特征权重:")
for feat, imp in zip(features, importances):
    print(f" ├─ 特征名称: {feat:<12} | 全局树分裂权重: {imp:.4f}")

perf_df.to_csv(os.path.join(reports_dir, "ml_universal_panel_final_report.csv"))
print(f"\n💾 2026长窗口高纯度盲测细节已成功导出至 reports/ml_universal_panel_final_report.csv\n")
