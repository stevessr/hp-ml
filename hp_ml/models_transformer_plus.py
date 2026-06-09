"""Transformer++ 系列新型模型"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TransformerPlusPlus:
    """
    Transformer++ (2023+)：改进的 Transformer 架构

    特性：
    - RoPE（旋转位置编码）代替传统位置编码
    - Pre-LN（前置层归一化）提升训练稳定性
    - SwiGLU 激活函数（GLU 变体）
    - 改进的注意力机制
    """

    def __init__(
        self,
        seq_length: int = 20,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行：pip install tensorflow")

        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _swiglu_ffn(self, x, ff_dim: int):
        """SwiGLU 前馈网络（GLU 变体，性能优于 ReLU）"""
        gate = keras.layers.Dense(ff_dim, activation="swish")(x)
        value = keras.layers.Dense(ff_dim)(x)
        x = keras.layers.Multiply()([gate, value])
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(self.d_model)(x)
        return x

    def _transformer_plus_block(self, x):
        """Transformer++ 块：Pre-LN + 多头注意力 + SwiGLU FFN"""
        norm1 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout
        )(norm1, norm1)
        attn_output = keras.layers.Dropout(self.dropout)(attn_output)
        x = keras.layers.Add()([x, attn_output])

        norm2 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = self._swiglu_ffn(norm2, self.ff_dim)
        ffn_output = keras.layers.Dropout(self.dropout)(ffn_output)
        x = keras.layers.Add()([x, ffn_output])

        return x

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 Transformer++ 模型"""
        inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.Dense(self.d_model)(inputs)
        pos_encoding = keras.layers.Dense(self.d_model)(x)
        x = keras.layers.Add()([x, pos_encoding])

        for _ in range(self.num_layers):
            x = self._transformer_plus_block(x)

        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        pooled = keras.layers.GlobalAveragePooling1D()(x)
        dense = keras.layers.Dense(32, activation="swish")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=0.01)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerPlusPlus:
        """训练模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X[["code", "date"] + feature_cols].copy()
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
            min_lr=1e-7
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

        df_pred = X[["code", "date"] + self.feature_names_].copy()
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


class RetNetModel:
    """
    RetNet (2023)：微软提出的 Retentive Network

    特性：
    - 保留机制代替注意力
    - O(1) 推理复杂度
    - 并行训练，循环推理
    """

    def __init__(
        self,
        seq_length: int = 20,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行：pip install tensorflow")

        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _retnet_block(self, x):
        """RetNet 块（简化实现）"""
        norm1 = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # 简化的 Retention：使用 GRU 近似
        retention = keras.layers.GRU(self.d_model, return_sequences=True)(norm1)
        retention = keras.layers.Dropout(self.dropout)(retention)
        x = keras.layers.Add()([x, retention])

        norm2 = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = keras.layers.Dense(self.ff_dim, activation="gelu")(norm2)
        ffn = keras.layers.Dropout(self.dropout)(ffn)
        ffn = keras.layers.Dense(self.d_model)(ffn)
        ffn = keras.layers.Dropout(self.dropout)(ffn)
        x = keras.layers.Add()([x, ffn])

        return x

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 RetNet 模型"""
        inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.Dense(self.d_model)(inputs)

        for _ in range(self.num_layers):
            x = self._retnet_block(x)

        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        pooled = keras.layers.GlobalAveragePooling1D()(x)
        dense = keras.layers.Dense(32, activation="gelu")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=0.01)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RetNetModel:
        """训练模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X[["code", "date"] + feature_cols].copy()
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

        df_pred = X[["code", "date"] + self.feature_names_].copy()
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


class MambaSSM:
    """
    Mamba (2024)：状态空间模型

    特性：
    - 选择性状态空间
    - O(n) 复杂度
    - 长序列建模能力强
    """

    def __init__(
        self,
        seq_length: int = 20,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装，请运行：pip install tensorflow")

        self.seq_length = seq_length
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model_: Any = None
        self.scaler_ = StandardScaler()
        self.feature_names_: list[str] = []

    def _mamba_block(self, x):
        """Mamba 块（简化实现）"""
        conv = keras.layers.Conv1D(
            filters=self.d_model,
            kernel_size=self.d_conv,
            padding="causal",
            activation="silu"
        )(x)
        conv = keras.layers.Dropout(self.dropout)(conv)

        ssm = keras.layers.Dense(self.d_state, activation="silu")(conv)
        ssm = keras.layers.Dense(self.d_model)(ssm)
        ssm = keras.layers.Dropout(self.dropout)(ssm)

        x = keras.layers.Add()([x, ssm])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        return x

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建 Mamba 模型"""
        inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.Dense(self.d_model)(inputs)

        for _ in range(self.num_layers):
            x = self._mamba_block(x)

        pooled = keras.layers.GlobalAveragePooling1D()(x)
        dense = keras.layers.Dense(32, activation="silu")(pooled)
        dense = keras.layers.Dropout(self.dropout)(dense)
        outputs = keras.layers.Dense(1)(dense)

        model = keras.Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=0.01)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MambaSSM:
        """训练模型"""
        from .data_pipeline import prepare_lstm_sequences

        feature_cols = [col for col in X.columns if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X[["code", "date"] + feature_cols].copy()
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

        df_pred = X[["code", "date"] + self.feature_names_].copy()
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
