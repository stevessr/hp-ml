"""单元测试：回测框架"""
import unittest

import numpy as np
import pandas as pd

from hp_ml.backtest import backtest_strategy, calculate_sharpe_ratio, calculate_drawdown


class TestBacktest(unittest.TestCase):

    def setUp(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="D")
        codes = ["510300", "510500", "512100"]

        data = []
        for date in dates:
            for code in codes:
                data.append({
                    "date": date,
                    "code": code,
                    "prediction": np.random.uniform(-0.02, 0.03),
                    "fwd_ret_5": np.random.uniform(-0.03, 0.03),
                })

        self.predictions_df = pd.DataFrame(data)

    def test_backtest_strategy(self):
        """测试回测策略"""
        metrics, returns_df = backtest_strategy(
            self.predictions_df,
            pred_col="prediction",
            target_col="fwd_ret_5",
            top_k=2,
            min_pred_threshold=-0.01,
            transaction_cost=0.001
        )

        # 验证指标
        self.assertIsNotNone(metrics.total_return)
        self.assertIsNotNone(metrics.sharpe_ratio)
        self.assertIsNotNone(metrics.max_drawdown)
        self.assertGreaterEqual(metrics.win_rate, 0)
        self.assertLessEqual(metrics.win_rate, 1)

        # 验证返回数据框
        self.assertIn("date", returns_df.columns)
        self.assertIn("return", returns_df.columns)
        self.assertIn("equity", returns_df.columns)

    def test_calculate_sharpe_ratio(self):
        """测试夏普比率计算"""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.03)

        self.assertIsInstance(sharpe, float)

    def test_calculate_drawdown(self):
        """测试回撤计算"""
        equity = pd.Series([1.0, 1.1, 1.05, 1.15, 1.08, 1.20])
        max_dd, dd_series = calculate_drawdown(equity)

        self.assertLess(max_dd, 0)
        self.assertEqual(len(dd_series), len(equity))


if __name__ == "__main__":
    unittest.main()
