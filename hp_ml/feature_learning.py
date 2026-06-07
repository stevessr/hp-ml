"""技术指标特征自动学习和选择"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression

warnings.filterwarnings("ignore")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


class TechnicalIndicatorGenerator:
    """技术指标自动生成器"""

    def __init__(self, include_groups: list[str] | None = None):
        """
        Args:
            include_groups: 包含的指标组
                ['momentum', 'trend', 'volatility', 'volume', 'others']
        """
        if not TA_AVAILABLE:
            raise ImportError("ta 库未安装，请运行: pip install ta")

        self.include_groups = include_groups or ["momentum", "trend", "volatility", "volume"]
        self.generated_features_: list[str] = []

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        自动生成技术指标

        Args:
            df: 包含 open, high, low, close, volume 的数据框

        Returns:
            添加了技术指标的数据框
        """
        result = df.copy()

        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in result.columns:
                raise ValueError(f"缺少必需列: {col}")

        # 确保数值类型
        for col in required_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        # 动量指标
        if "momentum" in self.include_groups:
            result = self._add_momentum_indicators(result)

        # 趋势指标
        if "trend" in self.include_groups:
            result = self._add_trend_indicators(result)

        # 波动率指标
        if "volatility" in self.include_groups:
            result = self._add_volatility_indicators(result)

        # 成交量指标
        if "volume" in self.include_groups:
            result = self._add_volume_indicators(result)

        # 其他指标
        if "others" in self.include_groups:
            result = self._add_other_indicators(result)

        # 记录生成的特征
        self.generated_features_ = [col for col in result.columns if col not in df.columns]

        return result

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加动量指标"""
        # RSI
        df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["rsi_7"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # ROC
        df["roc_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()
        df["roc_20"] = ta.momentum.ROCIndicator(df["close"], window=20).roc()

        # Williams %R
        df["williams_r"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"]).williams_r()

        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趋势指标"""
        # SMA
        df["sma_5"] = ta.trend.SMAIndicator(df["close"], window=5).sma_indicator()
        df["sma_10"] = ta.trend.SMAIndicator(df["close"], window=10).sma_indicator()
        df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
        df["sma_60"] = ta.trend.SMAIndicator(df["close"], window=60).sma_indicator()

        # EMA
        df["ema_5"] = ta.trend.EMAIndicator(df["close"], window=5).ema_indicator()
        df["ema_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()

        # ADX
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        # CCI
        df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"]).cci()

        # Aroon
        aroon = ta.trend.AroonIndicator(df["close"])
        df["aroon_up"] = aroon.aroon_up()
        df["aroon_down"] = aroon.aroon_down()

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率指标"""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()

        # ATR
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"])
        df["kc_high"] = kc.keltner_channel_hband()
        df["kc_low"] = kc.keltner_channel_lband()
        df["kc_mid"] = kc.keltner_channel_mband()

        # Donchian Channel
        dc = ta.volatility.DonchianChannel(df["high"], df["low"], df["close"])
        df["dc_high"] = dc.donchian_channel_hband()
        df["dc_low"] = dc.donchian_channel_lband()

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量指标"""
        # OBV
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

        # CMF
        df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            df["high"], df["low"], df["close"], df["volume"]
        ).chaikin_money_flow()

        # MFI
        df["mfi"] = ta.volume.MFIIndicator(
            df["high"], df["low"], df["close"], df["volume"]
        ).money_flow_index()

        # Force Index
        df["force_index"] = ta.volume.ForceIndexIndicator(df["close"], df["volume"]).force_index()

        # VWAP
        df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            df["high"], df["low"], df["close"], df["volume"]
        ).volume_weighted_average_price()

        return df

    def _add_other_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加其他指标"""
        # Daily Return
        df["daily_return"] = ta.others.DailyReturnIndicator(df["close"]).daily_return()

        # Cumulative Return
        df["cumulative_return"] = ta.others.CumulativeReturnIndicator(df["close"]).cumulative_return()

        return df


class FeatureSelector:
    """特征自动选择器"""

    def __init__(
        self,
        method: str = "importance",
        n_features: int | None = None,
        threshold: float = 0.01,
    ):
        """
        Args:
            method: 选择方法
                'importance' - 基于特征重要性
                'rfe' - 递归特征消除
                'univariate' - 单变量统计检验
                'mutual_info' - 互信息
            n_features: 保留特征数量
            threshold: 重要性阈值（importance 方法）
        """
        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.selected_features_: list[str] = []
        self.feature_scores_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeatureSelector:
        """拟合特征选择器"""
        feature_names = X.columns.tolist()

        if self.method == "importance":
            self._select_by_importance(X, y, feature_names)

        elif self.method == "rfe":
            self._select_by_rfe(X, y, feature_names)

        elif self.method == "univariate":
            self._select_by_univariate(X, y, feature_names)

        elif self.method == "mutual_info":
            self._select_by_mutual_info(X, y, feature_names)

        else:
            raise ValueError(f"不支持的方法: {self.method}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """转换特征"""
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """拟合并转换"""
        self.fit(X, y)
        return self.transform(X)

    def _select_by_importance(self, X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> None:
        """基于随机森林特征重要性"""
        # 填充缺失值
        X_filled = X.fillna(X.median())

        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_filled, y)

        importances = rf.feature_importances_
        self.feature_scores_ = dict(zip(feature_names, importances))

        # 排序
        sorted_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)

        if self.n_features is not None:
            self.selected_features_ = [f for f, _ in sorted_features[: self.n_features]]
        else:
            self.selected_features_ = [f for f, score in sorted_features if score >= self.threshold]

    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> None:
        """递归特征消除"""
        X_filled = X.fillna(X.median())

        n_features_to_select = self.n_features or max(1, len(feature_names) // 2)

        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=5)
        rfe.fit(X_filled, y)

        self.selected_features_ = [f for f, selected in zip(feature_names, rfe.support_) if selected]
        self.feature_scores_ = dict(zip(feature_names, rfe.ranking_))

    def _select_by_univariate(self, X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> None:
        """单变量F检验"""
        X_filled = X.fillna(X.median())

        k = self.n_features or "all"
        selector = SelectKBest(f_regression, k=k)
        selector.fit(X_filled, y)

        self.selected_features_ = [f for f, selected in zip(feature_names, selector.get_support()) if selected]
        self.feature_scores_ = dict(zip(feature_names, selector.scores_))

    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> None:
        """互信息"""
        X_filled = X.fillna(X.median())

        mi_scores = mutual_info_regression(X_filled, y, random_state=42)
        self.feature_scores_ = dict(zip(feature_names, mi_scores))

        sorted_features = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)

        if self.n_features is not None:
            self.selected_features_ = [f for f, _ in sorted_features[: self.n_features]]
        else:
            threshold = np.percentile(mi_scores, 75)
            self.selected_features_ = [f for f, score in sorted_features if score >= threshold]

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性排名"""
        return pd.DataFrame(
            list(self.feature_scores_.items()), columns=["feature", "score"]
        ).sort_values("score", ascending=False)
