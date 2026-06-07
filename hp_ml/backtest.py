"""回测框架：多模型回测与性能评估"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """回测指标"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int


def calculate_drawdown(equity_curve: pd.Series) -> tuple[float, pd.Series]:
    """
    计算最大回撤和回撤序列

    Args:
        equity_curve: 权益曲线

    Returns:
        (最大回撤, 回撤序列)
    """
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    return float(max_dd), drawdown


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    计算夏普比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）

    Returns:
        夏普比率
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()
    trading_days = 252

    sharpe = (mean_return * trading_days - risk_free_rate) / (std_return * np.sqrt(trading_days))
    return float(sharpe)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    计算索提诺比率（仅考虑下行波动）

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）

    Returns:
        索提诺比率
    """
    if len(returns) == 0:
        return 0.0

    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    mean_return = returns.mean()
    downside_std = downside_returns.std()
    trading_days = 252

    sortino = (mean_return * trading_days - risk_free_rate) / (downside_std * np.sqrt(trading_days))
    return float(sortino)


def backtest_strategy(
    predictions_df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    top_k: int = 3,
    min_pred_threshold: float = 0.0,
    transaction_cost: float = 0.001,
) -> tuple[BacktestMetrics, pd.DataFrame]:
    """
    回测交易策略

    Args:
        predictions_df: 预测结果数据框，必须包含 date, code, pred_col, target_col
        pred_col: 预测列名
        target_col: 实际收益列名
        top_k: 每次选择排名前 K 的 ETF
        min_pred_threshold: 预测值最小阈值
        transaction_cost: 交易成本（双边）

    Returns:
        (回测指标, 每日收益详情)
    """
    if predictions_df.empty:
        raise ValueError("预测数据为空")

    required_cols = ["date", "code", pred_col, target_col]
    missing = [c for c in required_cols if c not in predictions_df.columns]
    if missing:
        raise ValueError(f"缺少必需列: {missing}")

    daily_returns = []
    trade_details = []

    for date, group in predictions_df.groupby("date"):
        group = group.dropna(subset=[pred_col, target_col])

        if group.empty:
            continue

        # 筛选预测值高于阈值的标的
        candidates = group[group[pred_col] >= min_pred_threshold]

        if len(candidates) == 0:
            daily_returns.append({"date": date, "return": 0.0, "position_count": 0})
            continue

        # 选择预测排名前 K 的标的
        selected = candidates.nlargest(min(top_k, len(candidates)), pred_col)

        # 等权配置
        weights = 1.0 / len(selected)
        actual_returns = selected[target_col].values
        position_return = np.sum(actual_returns * weights) - transaction_cost

        daily_returns.append({
            "date": date,
            "return": position_return,
            "position_count": len(selected)
        })

        for _, row in selected.iterrows():
            trade_details.append({
                "date": date,
                "code": row["code"],
                "prediction": row[pred_col],
                "actual_return": row[target_col],
                "weight": weights
            })

    if not daily_returns:
        raise ValueError("没有有效的回测交易日")

    returns_df = pd.DataFrame(daily_returns)
    returns_series = pd.Series(returns_df["return"].values)

    # 计算权益曲线
    equity_curve = (1 + returns_series).cumprod()
    total_return = equity_curve.iloc[-1] - 1

    # 计算年化收益
    n_days = len(returns_series)
    years = n_days / 252.0
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # 计算最大回撤
    max_dd, _ = calculate_drawdown(equity_curve)

    # 计算夏普比率和索提诺比率
    sharpe = calculate_sharpe_ratio(returns_series)
    sortino = calculate_sortino_ratio(returns_series)

    # 计算交易统计
    winning_trades = (returns_series > 0).sum()
    losing_trades = (returns_series < 0).sum()
    total_trades = len(returns_series)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    avg_win = returns_series[returns_series > 0].mean() if winning_trades > 0 else 0.0
    avg_loss = returns_series[returns_series < 0].mean() if losing_trades > 0 else 0.0

    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0.0

    # 计算卡玛比率
    calmar = abs(annualized_return / max_dd) if max_dd != 0 else 0.0

    metrics = BacktestMetrics(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(max_dd),
        win_rate=float(win_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(profit_factor),
        calmar_ratio=float(calmar),
        sortino_ratio=float(sortino),
        total_trades=int(total_trades),
        winning_trades=int(winning_trades),
        losing_trades=int(losing_trades),
    )

    returns_df["equity"] = equity_curve.values
    returns_df["cumulative_return"] = equity_curve.values - 1

    return metrics, returns_df


def compare_models_backtest(
    model_predictions: dict[str, pd.DataFrame],
    target_col: str,
    **backtest_kwargs: Any
) -> pd.DataFrame:
    """
    对比多个模型的回测结果

    Args:
        model_predictions: {模型名称: 预测数据框} 字典
        target_col: 实际收益列名
        **backtest_kwargs: 传递给 backtest_strategy 的参数

    Returns:
        对比结果数据框
    """
    results = []

    for model_name, pred_df in model_predictions.items():
        pred_col = "prediction" if "prediction" in pred_df.columns else "pred_fwd_ret_5"

        try:
            metrics, _ = backtest_strategy(pred_df, pred_col, target_col, **backtest_kwargs)

            results.append({
                "model": model_name,
                "total_return": metrics.total_return,
                "annualized_return": metrics.annualized_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "calmar_ratio": metrics.calmar_ratio,
                "win_rate": metrics.win_rate,
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "profit_factor": metrics.profit_factor,
                "total_trades": metrics.total_trades,
            })
        except Exception as e:
            print(f"模型 {model_name} 回测失败: {e}")
            continue

    if not results:
        raise ValueError("所有模型回测均失败")

    return pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)
