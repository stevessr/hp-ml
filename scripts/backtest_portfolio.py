import os
import pandas as pd
import numpy as np

# 路径配置
data_dir = "data/topic2_broad_base"
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

# 选取经典与新一代的代表性宽基标的（防范矩阵过度共线性）
selected_assets = {
    '510050': '上证 50',
    '510300': '沪深 300',
    '512050': '中证 500',
    '159351': '中证 A500'
}

print("🏃 正在启动《选题 2》量化回测引擎...")
print("💡 增强策略方案：20 日均线 (SMA20) 趋势跟踪策略 vs 纯基准买入持有 (B&H)")

# 加载数据
price_dict = {}
for code in selected_assets.keys():
    file_path = os.path.join(data_dir, f"{code}_daily.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        date_col = [col for col in df.columns if 'date' in col.lower()][0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        
        # 兼容价格列
        close_col = 'Close' if 'Close' in df.columns else df.columns[0]
        
        # 降维处理，防止 MultiIndex 干扰
        series = df[close_col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
            
        price_dict[code] = series
        print(f" └─ 成功加载 {selected_assets[code]} ({code})，历史区间：{df.index.min().strftime('%Y-%m-%d')} -> {df.index.max().strftime('%Y-%m-%d')}")

# 联立对齐最新共存周期 (重点测试中证 A500 上市以来的表现)
all_prices = pd.DataFrame(price_dict).dropna().astype(float)
print(f"\n🎯 策略对齐回测区间 (全资产共存): {all_prices.index.min().strftime('%Y-%m-%d')} 至 {all_prices.index.max().strftime('%Y-%m-%d')}")

# 初始化回测报告大表
summary_results = []

for code, name in selected_assets.items():
    if code not in all_prices.columns:
        continue
    prices = all_prices[code]
    
    # 1. 纯买入持有基准 (Buy & Hold)
    bh_returns = prices.pct_change().fillna(0)
    bh_cum = (1 + bh_returns).cumprod() - 1
    
    # 2. 均线增强策略 (Trend Following Enhancement)
    # 当价格高于 20 日均线时满仓，低于时空仓
    sma20 = prices.rolling(window=20).mean()
    
    # 信号生成：1 为持仓，0 为空仓。将信号平移一天防范未来函数 (Look-ahead bias)
    signal = (prices > sma20).astype(int).shift(1).fillna(0)
    
    # 策略收益率 = 信号 * 资产明日收益率
    strat_returns = signal * bh_returns
    strat_cum = (1 + strat_returns).cumprod() - 1
    
    # 3. 指标计算
    def calc_metrics(returns, cum_perf):
        total_ret = cum_perf.iloc[-1]
        # 年化收益率 (假设一年 252 个交易日)
        ann_ret = (1 + total_ret) ** (252 / len(returns)) - 1
        # 年化波动率
        ann_vol = returns.std() * np.sqrt(252)
        # 夏普比率 (假设无风险利率 2.0%)
        sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
        
        # 最大回撤
        cum_prices = (1 + returns).cumprod()
        running_max = cum_prices.cummax()
        drawdown = (cum_prices - running_max) / running_max
        max_dd = drawdown.min()
        
        return total_ret, ann_ret, ann_vol, sharpe, max_dd

    bh_tot, bh_ann, bh_vol, bh_sha, bh_dd = calc_metrics(bh_returns, bh_cum)
    st_tot, st_ann, st_vol, st_sha, st_dd = calc_metrics(strat_returns, strat_cum)
    
    summary_results.append({
        '资产': name, '代码': code, '组合模式': '纯基准买入持有',
        '总收益率': f"{bh_tot:.2%}", '年化收益': f"{bh_ann:.2%}", 
        '年化波动': f"{bh_vol:.2%}", '夏普比率': f"{bh_sha:.2f}", '最大回撤': f"{bh_dd:.2%}"
    })
    summary_results.append({
        '资产': name, '代码': code, '组合模式': 'SMA20 趋势增强',
        '总收益率': f"{st_tot:.2%}", '年化收益': f"{st_ann:.2%}", 
        '年化波动': f"{st_vol:.2%}", '夏普比率': f"{st_sha:.2f}", '最大回撤': f"{st_dd:.2%}"
    })

# 展示并保存回测报表
perf_df = pd.DataFrame(summary_results)
print("\n==========================================================================")
print("📈 回测绩效对比表 (SMA20 均线增强 vs 纯被动持有)")
print("==========================================================================")
print(perf_df.to_string(index=False))
print("==========================================================================")

output_report_path = os.path.join(reports_dir, "etf_enhancement_backtest.csv")
perf_df.to_csv(output_report_path, index=False, encoding='utf-8-sig')
print(f"🎉 回测数据已成功清洗并导出至：{output_report_path}")
