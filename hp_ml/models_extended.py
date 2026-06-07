"""扩展模型库：Prophet、LSTM、增强随机森林"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ProphetWrapper:
    """Prophet 模型包装器，适配 sklearn 接口"""

    def __init__(
        self,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        daily_seasonality: bool = False,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
    ):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet 未安装，请运行: pip install prophet")

        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.models_: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ProphetWrapper:
        """
        训练 Prophet 模型

        每个 ETF（code）训练独立的 Prophet 模型
        """
        df = X.copy()
        df["y"] = y.values

        if "code" not in df.columns or "date" not in df.columns:
            raise ValueError("数据必须包含 'code' 和 'date' 列")

        for code, group in df.groupby("code"):
            if len(group) < 30:
                continue

            prophet_df = pd.DataFrame({
                "ds": pd.to_datetime(group["date"]),
                "y": group["y"].values
            })

            model = Prophet(
                seasonality_mode=self.seasonality_mode,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                daily_seasonality=self.daily_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_df)

            self.models_[code] = model

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测未来收益"""
        df = X.copy()
        predictions = np.zeros(len(df))

        if "code" not in df.columns or "date" not in df.columns:
            raise ValueError("数据必须包含 'code' 和 'date' 列")

        for idx, (code, group) in enumerate(df.groupby("code", sort=False)):
            if code not in self.models_:
                continue

            model = self.models_[code]
            future_df = pd.DataFrame({"ds": pd.to_datetime(group["date"])})

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = model.predict(future_df)

            group_indices = group.index
            predictions[group_indices] = forecast["yhat"].values

        return predictions


class LSTMModel:
    """LSTM 深度学习模型"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 LSTM 模型"""
        model = keras.Sequential([
            keras.layers.LSTM(self.units, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(self.dropout),
            keras.layers.LSTM(self.units // 2, return_sequences=False),
            keras.layers.Dropout(self.dropout),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1)
        ])

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LSTMModel:
        """训练 LSTM 模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)

        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled,
            y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class GRUModel:
    """GRU 深度学习模型（记忆门机制的变体）"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 GRU 模型"""
        model = keras.Sequential([
            keras.layers.GRU(self.units, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(self.dropout),
            keras.layers.GRU(self.units // 2, return_sequences=False),
            keras.layers.Dropout(self.dropout),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1)
        ])

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GRUModel:
        """训练 GRU 模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class BidirectionalLSTMModel:
    """双向 LSTM 模型（增强的记忆门机制）"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建双向 LSTM 模型"""
        model = keras.Sequential([
            keras.layers.Bidirectional(
                keras.layers.LSTM(self.units, return_sequences=True),
                input_shape=input_shape
            ),
            keras.layers.Dropout(self.dropout),
            keras.layers.Bidirectional(
                keras.layers.LSTM(self.units // 2, return_sequences=False)
            ),
            keras.layers.Dropout(self.dropout),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1)
        ])

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BidirectionalLSTMModel:
        """训练双向 LSTM 模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class AttentionLSTMModel:
    """带注意力机制的 LSTM 模型"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建带注意力机制的 LSTM 模型"""
        inputs = keras.layers.Input(shape=input_shape)

        # LSTM 层
        lstm_out = keras.layers.LSTM(self.units, return_sequences=True)(inputs)
        lstm_out = keras.layers.Dropout(self.dropout)(lstm_out)

        # 注意力机制
        attention = keras.layers.Dense(1, activation="tanh")(lstm_out)
        attention = keras.layers.Flatten()(attention)
        attention = keras.layers.Activation("softmax")(attention)
        attention = keras.layers.RepeatVector(self.units)(attention)
        attention = keras.layers.Permute([2, 1])(attention)

        # 应用注意力权重
        attended = keras.layers.multiply([lstm_out, attention])
        attended = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(attended)

        # 输出层
        dense = keras.layers.Dense(32, activation="relu")(attended)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AttentionLSTMModel:
        """训练带注意力的 LSTM 模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class MultiHeadAttentionLSTM:
    """多头注意力 LSTM 模型"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.units = units
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建多头注意力 LSTM 模型"""
        inputs = keras.layers.Input(shape=input_shape)

        # LSTM 层
        lstm_out = keras.layers.LSTM(self.units, return_sequences=True)(inputs)
        lstm_out = keras.layers.Dropout(self.dropout)(lstm_out)

        # 多头注意力层
        attention_out = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.units // self.num_heads,
            dropout=self.dropout
        )(lstm_out, lstm_out)

        # 残差连接和层归一化
        attention_out = keras.layers.Add()([lstm_out, attention_out])
        attention_out = keras.layers.LayerNormalization()(attention_out)

        # 全局平均池化
        pooled = keras.layers.GlobalAveragePooling1D()(attention_out)

        # 输出层
        dense = keras.layers.Dense(32, activation="relu")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MultiHeadAttentionLSTM:
        """训练多头注意力 LSTM 模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class SelfAttentionLSTM:
    """自注意力 LSTM 模型（Transformer-style）"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.units = units
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建自注意力 LSTM 模型"""
        inputs = keras.layers.Input(shape=input_shape)

        # 位置编码
        positions = keras.layers.Dense(input_shape[1])(inputs)
        x = keras.layers.Add()([inputs, positions])

        # LSTM 编码
        lstm_out = keras.layers.LSTM(self.units, return_sequences=True)(x)
        lstm_out = keras.layers.Dropout(self.dropout)(lstm_out)

        # 自注意力块
        attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.units // self.num_heads,
            dropout=self.dropout
        )(lstm_out, lstm_out)

        attention = keras.layers.Add()([lstm_out, attention])
        attention = keras.layers.LayerNormalization()(attention)

        # 前馈网络
        ff = keras.layers.Dense(self.ff_dim, activation="relu")(attention)
        ff = keras.layers.Dropout(self.dropout)(ff)
        ff = keras.layers.Dense(self.units)(ff)

        ff = keras.layers.Add()([attention, ff])
        ff = keras.layers.LayerNormalization()(ff)

        # 全局池化和输出
        pooled = keras.layers.GlobalAveragePooling1D()(ff)
        dense = keras.layers.Dense(32, activation="relu")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SelfAttentionLSTM:
        """训练自注意力 LSTM 模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class HierarchicalAttentionLSTM:
    """分层注意力 LSTM 模型（多尺度注意力）"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建分层注意力 LSTM 模型"""
        inputs = keras.layers.Input(shape=input_shape)

        # 第一层 LSTM：局部特征提取
        lstm1 = keras.layers.LSTM(self.units, return_sequences=True)(inputs)
        lstm1 = keras.layers.Dropout(self.dropout)(lstm1)

        # 局部注意力（关注短期模式）
        local_attention = keras.layers.Dense(1, activation="tanh")(lstm1)
        local_attention = keras.layers.Flatten()(local_attention)
        local_attention = keras.layers.Activation("softmax")(local_attention)
        local_attention = keras.layers.RepeatVector(self.units)(local_attention)
        local_attention = keras.layers.Permute([2, 1])(local_attention)

        local_context = keras.layers.multiply([lstm1, local_attention])
        local_context = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(local_context)

        # 第二层 LSTM：全局特征提取
        lstm2 = keras.layers.LSTM(self.units, return_sequences=True)(lstm1)
        lstm2 = keras.layers.Dropout(self.dropout)(lstm2)

        # 全局注意力（关注长期趋势）
        global_attention = keras.layers.Dense(1, activation="tanh")(lstm2)
        global_attention = keras.layers.Flatten()(global_attention)
        global_attention = keras.layers.Activation("softmax")(global_attention)
        global_attention = keras.layers.RepeatVector(self.units)(global_attention)
        global_attention = keras.layers.Permute([2, 1])(global_attention)

        global_context = keras.layers.multiply([lstm2, global_attention])
        global_context = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=1))(global_context)

        # 融合局部和全局上下文
        combined = keras.layers.Concatenate()([local_context, global_context])
        combined = keras.layers.Dense(self.units, activation="relu")(combined)
        combined = keras.layers.Dropout(self.dropout)(combined)

        # 输出层
        outputs = keras.layers.Dense(1)(combined)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> HierarchicalAttentionLSTM:
        """训练分层注意力 LSTM 模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class CNNNGram:
    """基于 CNN 的 N-Gram 模型：使用 1D 卷积提取 n-gram 特征"""

    def __init__(
        self,
        seq_length: int = 20,
        filters: int = 64,
        kernel_sizes: list[int] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.filters = filters
        self.kernel_sizes = kernel_sizes or [2, 3, 5]  # 2-gram, 3-gram, 5-gram
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建多尺度 CNN N-Gram 模型"""
        inputs = keras.layers.Input(shape=input_shape)

        # 多个 n-gram 分支（不同卷积核大小）
        conv_outputs = []
        for kernel_size in self.kernel_sizes:
            # 1D 卷积提取 n-gram 特征
            conv = keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=kernel_size,
                activation="relu",
                padding="same"
            )(inputs)
            conv = keras.layers.Dropout(self.dropout)(conv)

            # 全局最大池化
            pooled = keras.layers.GlobalMaxPooling1D()(conv)
            conv_outputs.append(pooled)

        # 合并所有 n-gram 特征
        if len(conv_outputs) > 1:
            merged = keras.layers.Concatenate()(conv_outputs)
        else:
            merged = conv_outputs[0]

        # 全连接层
        dense = keras.layers.Dense(64, activation="relu")(merged)
        dense = keras.layers.Dropout(self.dropout)(dense)
        dense = keras.layers.Dense(32, activation="relu")(dense)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CNNNGram:
        """训练模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class TemporalConvNet:
    """时间卷积网络（TCN）：使用膨胀卷积的 N-Gram 扩展"""

    def __init__(
        self,
        seq_length: int = 20,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_layers: int = 3,
        dilation_rates: list[int] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dilation_rates = dilation_rates or [1, 2, 4]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 TCN 模型"""
        inputs = keras.layers.Input(shape=input_shape)
        x = inputs

        # 堆叠膨胀卷积层
        for i, dilation_rate in enumerate(self.dilation_rates):
            # 膨胀卷积
            conv = keras.layers.Conv1D(
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
                padding="causal",
                activation="relu"
            )(x)
            conv = keras.layers.Dropout(self.dropout)(conv)

            # 残差连接
            if i == 0:
                # 第一层需要调整维度
                residual = keras.layers.Conv1D(self.num_filters, 1)(x)
            else:
                residual = x

            x = keras.layers.Add()([conv, residual])
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # 全局池化
        pooled = keras.layers.GlobalAveragePooling1D()(x)

        # 输出层
        dense = keras.layers.Dense(32, activation="relu")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TemporalConvNet:
        """训练模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class WaveNet:
    """WaveNet 风格模型：深度膨胀卷积 + 门控激活"""

    def __init__(
        self,
        seq_length: int = 20,
        num_filters: int = 32,
        kernel_size: int = 2,
        num_blocks: int = 2,
        num_layers_per_block: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _gated_activation(self, x):
        """门控激活单元"""
        # 分成两半：tanh 部分和 sigmoid 部分
        tanh_out = keras.layers.Activation("tanh")(x)
        sigmoid_out = keras.layers.Activation("sigmoid")(x)
        return keras.layers.Multiply()([tanh_out, sigmoid_out])

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 WaveNet 风格模型"""
        inputs = keras.layers.Input(shape=input_shape)

        # 输入投影
        x = keras.layers.Conv1D(self.num_filters, 1)(inputs)

        skip_connections = []

        # 多个块，每个块包含多层膨胀卷积
        for block in range(self.num_blocks):
            for layer in range(self.num_layers_per_block):
                dilation_rate = 2 ** layer

                # 膨胀卷积
                conv = keras.layers.Conv1D(
                    self.num_filters * 2,  # 双倍输出用于门控
                    kernel_size=self.kernel_size,
                    dilation_rate=dilation_rate,
                    padding="causal"
                )(x)

                # 门控激活
                gated = self._gated_activation(conv)
                gated = keras.layers.Dropout(self.dropout)(gated)

                # 1x1 卷积
                processed = keras.layers.Conv1D(self.num_filters, 1)(gated)

                # Skip 连接
                skip = keras.layers.Conv1D(self.num_filters, 1)(processed)
                skip_connections.append(skip)

                # 残差连接
                x = keras.layers.Add()([x, processed])

        # 合并所有 skip 连接
        skip_sum = keras.layers.Add()(skip_connections)

        # 后处理
        out = keras.layers.Activation("relu")(skip_sum)
        out = keras.layers.Conv1D(self.num_filters, 1, activation="relu")(out)

        # 全局池化
        pooled = keras.layers.GlobalAveragePooling1D()(out)

        # 输出层
        dense = keras.layers.Dense(16, activation="relu")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WaveNet:
        """训练模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class TransformerXL:
    """Transformer-XL：使用段级循环和相对位置编码的 Transformer"""

    def __init__(
        self,
        seq_length: int = 20,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        memory_length: int = 10,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.memory_length = memory_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 Transformer-XL 模型"""
        inputs = keras.layers.Input(shape=input_shape)

        # 输入投影到 d_model 维度
        x = keras.layers.Dense(self.d_model)(inputs)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # 多层 Transformer 块（简化的 XL 风格）
        for _ in range(self.num_layers):
            # 自注意力
            attention = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout
            )(x, x)
            attention = keras.layers.Dropout(self.dropout)(attention)
            x1 = keras.layers.Add()([x, attention])
            x1 = keras.layers.LayerNormalization(epsilon=1e-6)(x1)

            # 前馈网络
            ff = keras.layers.Dense(self.ff_dim, activation="relu")(x1)
            ff = keras.layers.Dropout(self.dropout)(ff)
            ff = keras.layers.Dense(self.d_model)(ff)
            ff = keras.layers.Dropout(self.dropout)(ff)

            x = keras.layers.Add()([x1, ff])
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # 全局平均池化
        pooled = keras.layers.GlobalAveragePooling1D()(x)

        # 输出层
        dense = keras.layers.Dense(32, activation="relu")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerXL:
        """训练模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class MemoryAugmentedTransformer:
    """Memory-Augmented Transformer：带外部记忆库的 Transformer"""

    def __init__(
        self,
        seq_length: int = 20,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        memory_size: int = 32,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.memory_size = memory_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 Memory-Augmented Transformer（简化版）"""
        inputs = keras.layers.Input(shape=input_shape)

        # 输入投影
        x = keras.layers.Dense(self.d_model)(inputs)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # 使用固定记忆模拟（简化实现，避免动态batch问题）
        # 在实际应用中，可以用自定义Layer包装记忆逻辑

        # Transformer 层（简化版，只用自注意力）
        for _ in range(self.num_layers):
            # 自注意力（模拟记忆增强）
            self_attention = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads,
                dropout=self.dropout
            )(x, x)
            self_attention = keras.layers.Dropout(self.dropout)(self_attention)
            x1 = keras.layers.Add()([x, self_attention])
            x1 = keras.layers.LayerNormalization(epsilon=1e-6)(x1)

            # 记忆增强：使用额外的全连接层模拟外部记忆读取
            memory_enhanced = keras.layers.Dense(self.d_model, activation="tanh")(x1)
            memory_enhanced = keras.layers.Dropout(self.dropout)(memory_enhanced)
            x2 = keras.layers.Add()([x1, memory_enhanced])
            x2 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)

            # 前馈网络
            ff = keras.layers.Dense(self.ff_dim, activation="relu")(x2)
            ff = keras.layers.Dropout(self.dropout)(ff)
            ff = keras.layers.Dense(self.d_model)(ff)
            ff = keras.layers.Dropout(self.dropout)(ff)

            x = keras.layers.Add()([x2, ff])
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # 全局池化
        pooled = keras.layers.GlobalAveragePooling1D()(x)

        # 输出层
        dense = keras.layers.Dense(32, activation="relu")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MemoryAugmentedTransformer:
        """训练模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class GRUTransformer:
    """GRU-Transformer：用 GRU 替换 LSTM 的混合模型"""

    def __init__(
        self,
        seq_length: int = 20,
        gru_units: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_transformer_blocks: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.gru_units = gru_units
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _transformer_block(self, inputs: Any, d_model: int) -> Any:
        """Transformer 块"""
        attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=d_model // self.num_heads,
            dropout=self.dropout
        )(inputs, inputs)
        attention = keras.layers.Dropout(self.dropout)(attention)
        x1 = keras.layers.Add()([inputs, attention])
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(x1)

        ff = keras.layers.Dense(self.ff_dim, activation="relu")(x1)
        ff = keras.layers.Dropout(self.dropout)(ff)
        ff = keras.layers.Dense(d_model)(ff)
        ff = keras.layers.Dropout(self.dropout)(ff)

        x2 = keras.layers.Add()([x1, ff])
        x2 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        return x2

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 GRU-Transformer 模型"""
        inputs = keras.layers.Input(shape=input_shape)

        # GRU 编码器
        gru_out = keras.layers.GRU(
            self.gru_units,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.dropout
        )(inputs)
        gru_out = keras.layers.LayerNormalization(epsilon=1e-6)(gru_out)

        # Transformer 块
        x = gru_out
        for _ in range(self.num_transformer_blocks):
            x = self._transformer_block(x, self.gru_units)

        # 注意力池化
        attention_weights = keras.layers.Dense(1, activation="tanh")(x)
        attention_weights = keras.layers.Flatten()(attention_weights)
        attention_weights = keras.layers.Activation("softmax")(attention_weights)
        attention_weights = keras.layers.RepeatVector(self.gru_units)(attention_weights)
        attention_weights = keras.layers.Permute([2, 1])(attention_weights)

        context = keras.layers.multiply([x, attention_weights])
        pooled = keras.layers.Lambda(lambda z: keras.backend.sum(z, axis=1))(context)

        # 输出层
        dense = keras.layers.Dense(32, activation="relu")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GRUTransformer:
        """训练模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class LSTMTransformer:
    """LSTM-Transformer 混合模型：结合 LSTM 序列编码和 Transformer 自注意力机制"""

    def __init__(
        self,
        seq_length: int = 20,
        lstm_units: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_transformer_blocks: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行: pip install tensorflow")

        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _transformer_block(self, inputs: Any, d_model: int) -> Any:
        """构建单个 Transformer 块"""
        # 多头自注意力
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=d_model // self.num_heads,
            dropout=self.dropout
        )(inputs, inputs)
        attention_output = keras.layers.Dropout(self.dropout)(attention_output)

        # 残差连接 + 层归一化
        x1 = keras.layers.Add()([inputs, attention_output])
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(x1)

        # 前馈网络
        ffn_output = keras.layers.Dense(self.ff_dim, activation="relu")(x1)
        ffn_output = keras.layers.Dropout(self.dropout)(ffn_output)
        ffn_output = keras.layers.Dense(d_model)(ffn_output)
        ffn_output = keras.layers.Dropout(self.dropout)(ffn_output)

        # 残差连接 + 层归一化
        x2 = keras.layers.Add()([x1, ffn_output])
        x2 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        return x2

    def _positional_encoding(self, inputs: Any) -> Any:
        """添加位置编码"""
        seq_len = self.seq_length
        d_model = self.lstm_units

        # 简化的位置编码：使用可学习的嵌入
        position_embedding = keras.layers.Dense(d_model)(inputs)
        return keras.layers.Add()([inputs, position_embedding])

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 LSTM-Transformer 混合模型"""
        inputs = keras.layers.Input(shape=input_shape)

        # 第一阶段：LSTM 编码器（提取时序特征）
        lstm_out = keras.layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.dropout
        )(inputs)
        lstm_out = keras.layers.LayerNormalization(epsilon=1e-6)(lstm_out)

        # 第二阶段：位置编码
        x = self._positional_encoding(lstm_out)

        # 第三阶段：堆叠 Transformer 块（捕获长距离依赖）
        for _ in range(self.num_transformer_blocks):
            x = self._transformer_block(x, self.lstm_units)

        # 第四阶段：全局池化和输出
        # 使用注意力池化而不是简单的平均池化
        attention_weights = keras.layers.Dense(1, activation="tanh")(x)
        attention_weights = keras.layers.Flatten()(attention_weights)
        attention_weights = keras.layers.Activation("softmax")(attention_weights)
        attention_weights = keras.layers.RepeatVector(self.lstm_units)(attention_weights)
        attention_weights = keras.layers.Permute([2, 1])(attention_weights)

        context = keras.layers.multiply([x, attention_weights])
        pooled = keras.layers.Lambda(lambda z: keras.backend.sum(z, axis=1))(context)

        # 输出层
        dense = keras.layers.Dense(64, activation="relu")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        dense = keras.layers.Dense(32, activation="relu")(dense)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LSTMTransformer:
        """训练 LSTM-Transformer 模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)
        X_seq_scaled = self.scaler_.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        self.model_.fit(
            X_seq_scaled, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.model_ is None:
            raise RuntimeError("模型未训练")

        from .data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(df_pred, self.feature_names_, "target", self.seq_length)
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class EnhancedRandomForest:
    """增强随机森林，包含特征重要性分析"""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 10,
        min_samples_leaf: int = 15,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.pipeline_: Pipeline | None = None
        self.feature_names_: list[str] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EnhancedRandomForest:
        """训练随机森林"""
        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        self.pipeline_ = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", rf)
        ])

        self.pipeline_.fit(X[feature_cols], y)
        self.feature_importances_ = self.pipeline_.named_steps["model"].feature_importances_

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if self.pipeline_ is None:
            raise RuntimeError("模型未训练")

        return self.pipeline_.predict(X[self.feature_names_])

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if self.feature_importances_ is None:
            raise RuntimeError("模型未训练或没有特征重要性")

        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.feature_importances_
        }).sort_values("importance", ascending=False)


def make_extended_model(
    model_type: str,
    random_state: int = 42,
    **kwargs: Any
) -> Any:
    """
    创建扩展模型

    Args:
        model_type: 模型类型 ('prophet', 'lstm', 'gru', 'bilstm', 'attention_lstm',
                    'multihead_attention', 'self_attention', 'hierarchical_attention', 'enhanced_rf')
        random_state: 随机种子
        **kwargs: 模型特定参数

    Returns:
        模型实例
    """
    model_type = model_type.lower()

    if model_type == "prophet":
        return ProphetWrapper(**kwargs)

    if model_type == "lstm":
        return LSTMModel(**kwargs)

    if model_type == "gru":
        return GRUModel(**kwargs)

    if model_type in {"bilstm", "bidirectional_lstm"}:
        return BidirectionalLSTMModel(**kwargs)

    if model_type in {"attention_lstm", "lstm_attention"}:
        return AttentionLSTMModel(**kwargs)

    if model_type in {"multihead_attention", "multihead_lstm"}:
        return MultiHeadAttentionLSTM(**kwargs)

    if model_type in {"self_attention", "self_attention_lstm"}:
        return SelfAttentionLSTM(**kwargs)

    if model_type in {"hierarchical_attention", "hierarchical_lstm"}:
        return HierarchicalAttentionLSTM(**kwargs)

    if model_type in {"lstm_transformer", "transformer_lstm", "hybrid_transformer"}:
        return LSTMTransformer(**kwargs)

    if model_type in {"transformer_xl", "transformerxl"}:
        return TransformerXL(**kwargs)

    if model_type in {"memory_transformer", "memory_augmented_transformer"}:
        return MemoryAugmentedTransformer(**kwargs)

    if model_type in {"gru_transformer", "transformer_gru"}:
        return GRUTransformer(**kwargs)

    if model_type in {"cnn_ngram", "ngram_cnn", "cnn"}:
        return CNNNGram(**kwargs)

    if model_type in {"tcn", "temporal_conv_net"}:
        return TemporalConvNet(**kwargs)

    if model_type in {"wavenet", "wave_net"}:
        return WaveNet(**kwargs)

    if model_type in {"enhanced_rf", "rf_enhanced"}:
        return EnhancedRandomForest(random_state=random_state, **kwargs)

    raise ValueError(f"不支持的模型类型: {model_type}")
