#!/usr/bin/env python3
"""Transformer架构 - 突破单模型天花板

Transformer的优势:
1. 全局注意力 - 捕捉所有时间步的关系
2. 位置编码 - 理解时间序列的顺序
3. 多头注意力 - 从多个角度学习
4. 前馈网络 - 强大的非线性变换
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hp_ml.config import PROCESSED_DIR, REPORTS_DIR

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TransformerEncoder(keras.layers.Layer):
    """Transformer编码器层"""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model),
        ])

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        # 多头注意力
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class PositionalEncoding(keras.layers.Layer):
    """位置编码"""

    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(seq_len, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, seq_len, d_model):
        angle_rads = self.get_angles(
            np.arange(seq_len)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        # 偶数索引使用sin
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # 奇数索引使用cos
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


class TransformerModel:
    """基于Transformer的时间序列预测模型"""

    def __init__(
        self,
        seq_length: int = 20,
        d_model: int = 64,
        num_heads: int = 4,
        dff: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 40,
        batch_size: int = 32,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow未安装")

        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建Transformer模型"""
        inputs = keras.Input(shape=input_shape)

        # 输入投影到d_model维度
        x = keras.layers.Dense(self.d_model)(inputs)

        # 位置编码
        x = PositionalEncoding(self.seq_length, self.d_model)(x)
        x = keras.layers.Dropout(self.dropout)(x)

        # Transformer编码器层
        for _ in range(self.num_layers):
            x = TransformerEncoder(
                self.d_model,
                self.num_heads,
                self.dff,
                self.dropout
            )(x, training=True)

        # 全局池化
        x = keras.layers.GlobalAveragePooling1D()(x)

        # 输出层
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(self.dropout / 2)(x)
        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练"""
        from hp_ml.data_pipeline import prepare_lstm_sequences
        from sklearn.preprocessing import StandardScaler

        feature_cols = [col for col in X.columns
                       if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)

        self.scaler_ = StandardScaler()
        X_seq_scaled = self.scaler_.fit_transform(
            X_seq.reshape(-1, X_seq.shape[-1])
        ).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=0.00001,
            verbose=0
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

        from hp_ml.data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(
                df_pred, self.feature_names_, "target", self.seq_length
            )
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(
            X_seq.reshape(-1, X_seq.shape[-1])
        ).reshape(X_seq.shape)

        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


class HybridTransformerLSTM:
    """混合架构：Transformer + LSTM"""

    def __init__(
        self,
        seq_length: int = 20,
        d_model: int = 64,
        num_heads: int = 4,
        lstm_units: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 40,
        batch_size: int = 32,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow未安装")

        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建混合模型"""
        inputs = keras.Input(shape=input_shape)

        # Transformer分支
        trans_x = keras.layers.Dense(self.d_model)(inputs)
        trans_x = PositionalEncoding(self.seq_length, self.d_model)(trans_x)
        trans_x = TransformerEncoder(
            self.d_model, self.num_heads, self.d_model * 2, self.dropout
        )(trans_x, training=True)
        trans_out = keras.layers.GlobalAveragePooling1D()(trans_x)

        # LSTM分支
        lstm_out = keras.layers.LSTM(self.lstm_units, return_sequences=False)(inputs)
        lstm_out = keras.layers.LayerNormalization()(lstm_out)

        # 融合
        fused = keras.layers.Concatenate()([trans_out, lstm_out])
        fused = keras.layers.Dropout(self.dropout)(fused)

        # 输出
        x = keras.layers.Dense(64, activation='relu')(fused)
        x = keras.layers.Dropout(self.dropout)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练"""
        from hp_ml.data_pipeline import prepare_lstm_sequences
        from sklearn.preprocessing import StandardScaler

        feature_cols = [col for col in X.columns
                       if col not in ["code", "date", "name", "family_id", "is_trainable"]]
        self.feature_names_ = feature_cols

        df_train = X.copy()
        df_train["target"] = y.values

        X_seq, y_seq = prepare_lstm_sequences(df_train, feature_cols, "target", self.seq_length)

        self.scaler_ = StandardScaler()
        X_seq_scaled = self.scaler_.fit_transform(
            X_seq.reshape(-1, X_seq.shape[-1])
        ).reshape(X_seq.shape)

        self.model_ = self._build_model((self.seq_length, len(feature_cols)))

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=0.00001,
            verbose=0
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

        from hp_ml.data_pipeline import prepare_lstm_sequences

        df_pred = X.copy()
        df_pred["target"] = 0.0

        try:
            X_seq, _ = prepare_lstm_sequences(
                df_pred, self.feature_names_, "target", self.seq_length
            )
        except ValueError:
            return np.zeros(len(X))

        X_seq_scaled = self.scaler_.transform(
            X_seq.reshape(-1, X_seq.shape[-1])
        ).reshape(X_seq.shape)

        predictions = self.model_.predict(X_seq_scaled, verbose=0).flatten()

        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions

        return full_predictions


def main() -> int:
    """测试Transformer架构"""
    print("\n" + "="*80)
    print("🚀 测试Transformer架构")
    print("="*80 + "\n")

    # 加载数据
    data_path = PROCESSED_DIR / "training_panel_lite.csv"
    df = pd.read_csv(data_path)

    target_col = "fwd_ret_5"
    exclude_cols = {'date', 'code', 'name', 'family_id', 'close', target_col,
                   'is_trainable', 'is_future'}
    feature_cols = [c for c in df.columns if c not in exclude_cols
                   and not c.startswith('fwd_')]

    # 切分数据
    df = df.sort_values("date")
    trainable = df[df.get("is_trainable", True)].copy()
    unique_dates = sorted(trainable["date"].unique())
    split_idx = max(1, len(unique_dates) - 60)
    split_date = unique_dates[split_idx]

    train_df = trainable[trainable["date"] < split_date].copy()
    test_df = trainable[trainable["date"] >= split_date].copy()

    print(f"训练集: {len(train_df)} 行")
    print(f"测试集: {len(test_df)} 行\n")

    results = []

    # 1. 纯Transformer
    print("="*80)
    print("🔧 训练Transformer...")
    print("="*80)

    model1 = TransformerModel(
        seq_length=20,
        d_model=64,
        num_heads=4,
        dff=128,
        num_layers=2,
        dropout=0.3,
        learning_rate=0.001,
        epochs=40,
        batch_size=32
    )

    model1.fit(train_df[feature_cols + ["code", "date"]], train_df[target_col])
    pred1 = model1.predict(test_df[feature_cols + ["code", "date"]])
    actual = test_df[target_col].values

    acc1 = float(np.mean((pred1 > 0) == (actual > 0)))
    ic1 = float(spearmanr(pred1, actual)[0])

    print(f"  准确率: {acc1:.4f}")
    print(f"  IC: {ic1:.4f}\n")

    results.append({
        "model": "Transformer",
        "accuracy": acc1,
        "ic": ic1
    })

    # 2. 混合架构
    print("="*80)
    print("🔧 训练Transformer+LSTM混合架构...")
    print("="*80)

    model2 = HybridTransformerLSTM(
        seq_length=20,
        d_model=64,
        num_heads=4,
        lstm_units=64,
        dropout=0.3,
        learning_rate=0.001,
        epochs=40,
        batch_size=32
    )

    model2.fit(train_df[feature_cols + ["code", "date"]], train_df[target_col])
    pred2 = model2.predict(test_df[feature_cols + ["code", "date"]])

    acc2 = float(np.mean((pred2 > 0) == (actual > 0)))
    ic2 = float(spearmanr(pred2, actual)[0])

    print(f"  准确率: {acc2:.4f}")
    print(f"  IC: {ic2:.4f}\n")

    results.append({
        "model": "Hybrid Transformer+LSTM",
        "accuracy": acc2,
        "ic": ic2
    })

    # 保存结果
    output_dir = REPORTS_DIR / "transformer_architectures"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)

    print("\n" + "="*80)
    print("📊 结果对比")
    print("="*80 + "\n")

    print("基线对比:")
    print(f"  多尺度LSTM: 68.08%, IC: 0.3976\n")

    print("Transformer架构:")
    for idx, row in results_df.iterrows():
        print(f"{idx+1}. {row['model']}")
        print(f"   准确率: {row['accuracy']:.4f}")
        print(f"   IC: {row['ic']:.4f}\n")

    best_acc = results_df['accuracy'].max()
    print(f"💡 最佳准确率: {best_acc:.4f}")

    if best_acc > 0.6808:
        improvement = (best_acc - 0.6808) * 100
        print(f"💡 超越基线: +{improvement:.2f}%")
        print(f"\n✅ Transformer突破成功！")
    else:
        print(f"\n⚠️ 未超越基线 (68.08%)")

    print(f"\n💾 结果已保存: {output_dir}/results.csv\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
