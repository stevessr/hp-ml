import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# 1. 数据加载
input_path = "reports/composite_factor_backtest_detail.csv"
reports_dir = "reports"

df = pd.read_csv(input_path, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# 2. 特征工程与标签
df['Return_Lag1'] = df['Daily_Return'].shift(1)
df['Return_Lag2'] = df['Daily_Return'].shift(2)
df['Vol_Lag1'] = df['Daily_Return'].rolling(5).std().shift(1)

df['Target'] = (df['Daily_Return'].shift(-1) > 0).astype(int)
df_ml = df[['F_Tech', 'F_Sent', 'Return_Lag1', 'Return_Lag2', 'Vol_Lag1', 'Target', 'Daily_Return']].dropna()

features = ['F_Tech', 'F_Sent', 'Return_Lag1', 'Return_Lag2', 'Vol_Lag1']
X = df_ml[features]
y = df_ml['Target']

# 时序划分 (65% 训练, 35% 盲测)
split_idx = int(len(df_ml) * 0.65)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_returns = df_ml['Daily_Return'].iloc[split_idx:]

# 3. 网格搜索调优决策树
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {'n_estimators': [100, 150], 'max_depth': [4, 5], 'min_samples_leaf': [2, 4]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 4. 盲测集概率预测
pred_probs = best_model.predict_proba(X_test)[:, 1]

# 🎯 核心优化：动态仓位控制函数 (Continuous Position Sizing)
# 设定概率安全边际：低于 50% 概率绝对不开仓；达到 60% 概率则认为极其自信，直接满仓。
# 50% 到 60% 之间进行线性仓位放大
target_position = np.clip((pred_probs - 0.50) / (0.60 - 0.50), 0.0, 1.0)

results_df = pd.DataFrame(index=X_test.index)
results_df['Actual_Return'] = test_returns
results_df['ML_Prob'] = pred_probs
results_df['Target_Position'] = target_position

# 严格平移一天生效，剔除未来函数
results_df['Executed_Position'] = results_df['Target_Position'].shift(1).fillna(0)

# 5. 收益计算 (策略收益 = 仓位权重 * 市场明日真实收益)
results_df['B&H_Cum'] = (1 + results_df['Actual_Return']).cumprod() - 1
results_df['Dynamic_Weight_Strategy_Cum'] = (1 + results_df['Executed_Position'] * results_df['Actual_Return']).cumprod() - 1

# 6. 绩效精算
def calc_metrics(returns, cum_series):
    total_ret = cum_series.iloc[-1]
    ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    cum_prices = (1 + returns).cumprod()
    max_dd = ((cum_prices - cum_prices.cummax()) / cum_prices.cummax()).min()
    return total_ret, ann_ret, ann_vol, sharpe, max_dd

bh_tot, bh_ann, bh_vol, bh_sha, bh_dd = calc_metrics(results_df['Actual_Return'], results_df['B&H_Cum'])
ml_tot, ml_ann, ml_vol, ml_sha, ml_dd = calc_metrics(results_df['Executed_Position'] * results_df['Actual_Return'], results_df['Dynamic_Weight_Strategy_Cum'])

print("\n" + "="*85)
print("👑 终极进化：机器学习 + 动态智能仓位管理策略绩效看板 (盲测集)")
print("=====================================================================================")
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'年化波动':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*85)
print(f"{'被动买入持有 (沪深300)':<22}{bh_tot:>10.2%}{bh_ann:>12.2%}{bh_vol:>12.2%}{bh_sha:>12.2f}{bh_dd:>12.2%}")
print(f"{'💎 ML 智能动态仓位增强':<20}{ml_tot:>10.2%}{ml_ann:>12.2%}{ml_vol:>12.2%}{ml_sha:>12.2f}{ml_dd:>12.2%}")
print("=====================================================================================")

results_df.to_csv(os.path.join(reports_dir, "ml_advanced_position_detail.csv"))
print("💾 动态仓位及净值明细已成功导出至 reports/ml_advanced_position_detail.csv\n")
