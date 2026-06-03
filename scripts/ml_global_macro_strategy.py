import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# 1. 路径设置
input_path = "reports/composite_factor_backtest_detail.csv"
reports_dir = "reports"

print("🌐 正在启动《全球宏观外溢 + 梯度提升树》终极量化增强引擎...")

# 2. 跨海拉取新数据源：美股标普 500 历史历史数据 (Ticker: ^GSPC)
print("🇺🇸 正在读取/同步全球风险核心指标：标普 500 指数...")
try:
    sp500 = yf.download("^GSPC", start="2024-01-01", end="2026-06-03", progress=False)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    sp500_ret = sp500['Close'].pct_change().fillna(0)
    print(" └─ 标普 500 数据同步成功。")
except Exception as e:
    print(f" ⚠️ 标普 500 拉取失败，启用兜底零向量。错误：{e}")
    sp500_ret = pd.Series(0, index=pd.date_range("2024-01-01", "2026-06-03"))

# 3. 加载本地基础因子库
df = pd.read_csv(input_path, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# 4. 国际数据交叉对齐与特征工程
# 将美股昨晚的收益率映射到 A 股今天的时序行上 (前向填充，完美规避未来函数)
df['US_Stock_Lag1'] = sp500_ret.reindex(df.index, method='ffill').fillna(0)

# 衍生其他本土高级特征
df['Return_Lag1'] = df['Daily_Return'].shift(1)
df['Return_Lag2'] = df['Daily_Return'].shift(2)
df['Vol_Lag1'] = df['Daily_Return'].rolling(5).std().shift(1)

# 目标标签：明日 A 股是否上涨
df['Target'] = (df['Daily_Return'].shift(-1) > 0).astype(int)
df_ml = df[['F_Tech', 'F_Sent', 'US_Stock_Lag1', 'Return_Lag1', 'Return_Lag2', 'Vol_Lag1', 'Target', 'Daily_Return']].dropna()

# 特征阵列（加入了全新的国际溢价因子 US_Stock_Lag1）
features = ['F_Tech', 'F_Sent', 'US_Stock_Lag1', 'Return_Lag1', 'Return_Lag2', 'Vol_Lag1']
X = df_ml[features]
y = df_ml['Target']

# 时序划分 (65% 训练，35% 盲测验证)
split_idx = int(len(df_ml) * 0.65)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_returns = df_ml['Daily_Return'].iloc[split_idx:]

# 5. 升级算法：使用高级直方图梯度提升树 (HistGradientBoosting)
# 该算法对金融时序中的非线性离群点有极强的鲁棒性
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    'max_iter': [50, 100],
    'max_depth': [3, 4],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(
    HistGradientBoostingClassifier(random_state=42), 
    param_grid=param_grid, 
    cv=tscv, 
    scoring='accuracy', 
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 6. 预测盲测集概率并映射为连续智能仓位
pred_probs = best_model.predict_proba(X_test)[:, 1]
# 概率区间平滑映射到 [0, 1] 实际持仓仓位
target_position = np.clip((pred_probs - 0.48) / (0.58 - 0.48), 0.0, 1.0)

results_df = pd.DataFrame(index=X_test.index)
results_df['Actual_Return'] = test_returns
results_df['Target_Position'] = target_position
results_df['Executed_Position'] = results_df['Target_Position'].shift(1).fillna(0)

# 7. 累计收益精算
results_df['B&H_Cum'] = (1 + results_df['Actual_Return']).cumprod() - 1
results_df['Global_Macro_Strategy_Cum'] = (1 + results_df['Executed_Position'] * results_df['Actual_Return']).cumprod() - 1

def calc_metrics(returns, cum_series):
    total_ret = cum_series.iloc[-1]
    ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    cum_prices = (1 + returns).cumprod()
    max_dd = ((cum_prices - cum_prices.cummax()) / cum_prices.cummax()).min()
    return total_ret, ann_ret, ann_vol, sharpe, max_dd

bh_tot, bh_ann, bh_vol, bh_sha, bh_dd = calc_metrics(results_df['Actual_Return'], results_df['B&H_Cum'])
ml_tot, ml_ann, ml_vol, ml_sha, ml_dd = calc_metrics(results_df['Executed_Position'] * results_df['Actual_Return'], results_df['Global_Macro_Strategy_Cum'])

print("\n" + "="*95)
print("🌍 跨国多维特征：全球宏观联动 + 梯度提升树 (HGB) 智能仓位增强绩效看板 (盲测集)")
print("===============================================================================================")
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*95)
print(f"{'被动买入持有 (沪深 300)':<22}{bh_tot:>10.2%}{bh_ann:>12.2%}{bh_vol:>12.2%}{bh_sha:>12.2f}{bh_dd:>12.2%}")
print(f"{'👑 全球宏观混合增强':<21}{ml_tot:>10.2%}{ml_ann:>12.2%}{ml_vol:>12.2%}{ml_sha:>12.2f}{ml_dd:>12.2%}")
print("===============================================================================================")

results_df.to_csv(os.path.join(reports_dir, "ml_global_macro_position_detail.csv"))
print("💾 全球宏观动态策略细节已成功导出至 reports/ml_global_macro_position_detail.csv\n")
