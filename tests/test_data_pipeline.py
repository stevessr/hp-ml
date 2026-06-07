"""单元测试：数据划分管道"""
import unittest

import numpy as np
import pandas as pd

from hp_ml.data_pipeline import time_series_split, expanding_window_cv, prepare_lstm_sequences


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        codes = ["510300", "510500", "512100"]

        data = []
        for code in codes:
            for date in dates:
                data.append({
                    "date": date,
                    "code": code,
                    "close": np.random.uniform(3, 5),
                    "ret_5": np.random.uniform(-0.05, 0.05),
                    "vol_20": np.random.uniform(0.01, 0.03),
                    "fwd_ret_5": np.random.uniform(-0.05, 0.05),
                    "is_trainable": True,
                })

        self.panel = pd.DataFrame(data)

    def test_time_series_split(self):
        """测试时间序列划分"""
        split = time_series_split(
            self.panel,
            target_col="fwd_ret_5",
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )

        self.assertIsInstance(split.train, pd.DataFrame)
        self.assertIsInstance(split.val, pd.DataFrame)
        self.assertIsInstance(split.test, pd.DataFrame)

        self.assertGreater(len(split.train), 0)
        self.assertGreater(len(split.val), 0)
        self.assertGreater(len(split.test), 0)

        # 验证时间顺序
        self.assertLess(split.train_dates[1], split.val_dates[0])
        self.assertLess(split.val_dates[1], split.test_dates[0])

    def test_expanding_window_cv(self):
        """测试扩展窗口交叉验证"""
        splits = expanding_window_cv(
            self.panel,
            target_col="fwd_ret_5",
            n_splits=3,
            min_train_size=100,
            test_size=30
        )

        self.assertGreater(len(splits), 0)
        self.assertLessEqual(len(splits), 3)

        for train_df, test_df in splits:
            self.assertIsInstance(train_df, pd.DataFrame)
            self.assertIsInstance(test_df, pd.DataFrame)
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(test_df), 0)

    def test_prepare_lstm_sequences(self):
        """测试 LSTM 序列准备"""
        feature_cols = ["ret_5", "vol_20"]

        X_seq, y_seq = prepare_lstm_sequences(
            self.panel,
            feature_cols,
            target_col="fwd_ret_5",
            seq_length=10
        )

        self.assertEqual(len(X_seq.shape), 3)
        self.assertEqual(X_seq.shape[1], 10)
        self.assertEqual(X_seq.shape[2], len(feature_cols))
        self.assertEqual(len(y_seq), len(X_seq))


if __name__ == "__main__":
    unittest.main()
