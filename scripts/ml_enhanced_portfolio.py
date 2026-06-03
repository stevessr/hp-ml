import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# 1. 路径配置与数据加载
input_path = "reports/composite_factor_backtest_detail.csv"
reports_dir = "reports"

if not os.path.exists(input_path):
    print(f"❌ 错误：未找到基础因子数据 {input_path}，请先运行上一步的回测脚本。")
    exit()

print("🤖 正在初始化《机器学习 + 自动调优》复合增强引擎...")
df = pd.read_csv(input_path, parse_dates=['Date'])
df.set_index('Date', inplace=True)

# 2. 特征工程 (Feature Engineering)
# 基础特征：量价因子、舆情因子
# 衍生衍生特征：加入收益率的滞后项（Lagged Returns），让大模型具有时间序列的“记忆力”
df['Return_Lag1'] = df['Daily_Return'].shift(1)
df['Return_Lag2'] = df['Daily_Return'].shift(2)
df['Vol_Lag1'] = df['Daily_Return'].rolling(5).std().shift(1) # 过去 5 天的波动率

# 🎯 标签构建 (Labeling)
# 预测目标：明天的真实涨跌幅是否大于 0（1 代表涨，0 代表跌/平）
# 必须使用 .shift(-1) 将明天的结果移到今天作为标签
df['Target'] = (df['Daily_Return'].shift(-1) > 0).astype(int)

# 剔除因为 rolling 和 shift 产生的 NaN 行
df_ml = df[['F_Tech', 'F_Sent', 'Return_Lag1', 'Return_Lag2', 'Vol_Lag1', 'Target', 'Daily_Return']].dropna()

# 划分特征矩阵 X 和标签向量 y
features = ['F_Tech', 'F_Sent', 'Return_Lag1', 'Return_Lag2', 'Vol_Lag1']
X = df_ml[features]
y = df_ml['Target']

# 3. 严格的时序划分 (防止未来数据泄漏)
# 前 65% 的日子作为历史训练集，后 35% 的日子作为完全隔离的盲测验证集
split_idx = int(len(df_ml) * 0.65)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_returns = df_ml['Daily_Return'].iloc[split_idx:]

print(f"📊 数据集切分完毕：训练集 {len(X_train)} 天 | 测试集 {len(X_test)} 天")

# 4. 自动调优：利用网格搜索与时序交叉验证 (GridSearchCV + TimeSeriesSplit)
print("⚙️ 正在启动网格搜索，全自动优化随机森林超参数...")
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],               # 严格限制树深度防止在嘈杂的股市中过拟合
    'min_samples_leaf': [2, 4]
}

# 采用专门应对时间序列的 TimeSeriesSplit 交叉验证器
tscv = TimeSeriesSplit(n_splits=3)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"🎉 自动调优完成！最优模型参数：{grid_search.best_params_}")

# 5. 在盲测集上执行流式预测与回测
# 模型预测出明天上涨的概率
pred_probs = best_model.predict_proba(X_test)[:, 1]

# 🎯 动态策略信号：只有当大模型预测明天上涨的概率大于 55% 时才超配做多，否则空仓防御
ml_signal = (pred_probs > 0.55).astype(int)

# 构建测试集绩效表
results_df = pd.DataFrame(index=X_test.index)
results_df['Actual_Return'] = test_returns
results_df['ML_Signal'] = ml_signal

# 关键：信号延迟一天生效，规避未来函数
results_df['ML_Signal_Delayed'] = results_df['ML_Signal'].shift(1).fillna(0)

# 计算累计净值
results_df['B&H_Cum'] = (1 + results_df['Actual_Return']).cumprod() - 1
results_df['ML_Strategy_Cum'] = (1 + results_df['ML_Signal_Delayed'] * results_df['Actual_Return']).cumprod() - 1

# 6. 精算机器学习增强绩效
def calc_metrics(returns, cum_series):
    total_ret = cum_series.iloc[-1]
    ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    cum_prices = (1 + returns).cumprod()
    max_dd = ((cum_prices - cum_prices.cummax()) / cum_prices.cummax()).min()
    return total_ret, ann_ret, sharpe, max_dd

bh_tot, bh_ann, bh_sha, bh_dd = calc_metrics(results_df['Actual_Return'], results_df['B&H_Cum'])
ml_tot, ml_ann, ml_sha, ml_dd = calc_metrics(results_df['ML_Signal_Delayed'] * results_df['Actual_Return'], results_df['ML_Strategy_Cum'])

# 7. 打印多因子看板
print("\n" + "="*82)
print("🤖 机器学习 (Random Forest) 全自动调优增强策略盲测绩效看板")
print("="*82)
print(f"{'策略组合模式':<25}{'总收益率':<12}{'年化收益':<12}{'夏普比率':<12}{'最大回撤':<12}")
print("-"*82)
print(f"{'隔离盲测基准 (沪深 300)':<22}{bh_tot:>10.2%}{bh_ann:>12.2%}{bh_sha:>12.2f}{bh_dd:>12.2%}")
print(f"{'🔥 ML 决策树智能动态增强':<20}{ml_tot:>10.2%}{ml_ann:>12.2%}{ml_sha:>12.2f}{ml_dd:>12.2%}")
print("="*82)

# 特征重要性输出（告诉答辩老师大模型到底看了啥）
importances = best_model.feature_importances_
print("\n🧠 机器学习模型特征权重（主导因子）分析：")
for feat, imp in zip(features, importances):
    print(f" ├─ 因子：{feat:<12} | 大模型重视度权重：{imp:.4f}")

# 导出细节
results_df.to_csv(os.path.join(reports_dir, "ml_enhanced_backtest_detail.csv"))
print(f"\n💾 机器学习动态信号已成功导出至 reports/ml_enhanced_backtest_detail.csv\n")
