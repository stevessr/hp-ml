#!/usr/bin/env python3
"""创新架构：残差 LSTM + 多尺度注意力

创新点：
1. 残差连接 - 缓解梯度消失
2. 多尺度特征提取 - 捕捉不同时间尺度
3. 门控融合 - 自适应特征融合
4. Layer Normalization - 训练稳定性
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


class ResidualAttentionLSTM:
    """残差注意力 LSTM - 创新架构"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 25,
        batch_size: int = 32,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装")

        self.seq_length = seq_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建残差注意力 LSTM"""
        inputs = keras.Input(shape=input_shape)

        # 主 LSTM 分支
        lstm_out = keras.layers.LSTM(
            self.units,
            return_sequences=True,
            name='main_lstm'
        )(inputs)
        lstm_out = keras.layers.LayerNormalization()(lstm_out)
        lstm_out = keras.layers.Dropout(self.dropout)(lstm_out)

        # 注意力机制
        attention = keras.layers.Dense(1, activation='tanh')(lstm_out)
        attention = keras.layers.Flatten()(attention)
        attention = keras.layers.Activation('softmax')(attention)
        attention = keras.layers.RepeatVector(self.units)(attention)
        attention = keras.layers.Permute([2, 1])(attention)

        # 加权
        attended = keras.layers.Multiply()([lstm_out, attention])
        attended = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)

        # 残差分支（跳过连接）
        residual = keras.layers.Dense(self.units, activation='relu')(inputs[:, -1, :])
        residual = keras.layers.LayerNormalization()(residual)

        # 门控融合
        gate = keras.layers.Dense(self.units, activation='sigmoid')(
            keras.layers.Concatenate()([attended, residual])
        )
        fused = keras.layers.Add()([
            keras.layers.Multiply()([attended, gate]),
            keras.layers.Multiply()([residual, keras.layers.Lambda(lambda x: 1 - x)(gate)])
        ])

        # 输出层
        out = keras.layers.Dense(32, activation='relu')(fused)
        out = keras.layers.Dropout(self.dropout)(out)
        out = keras.layers.Dense(1)(out)

        model = keras.Model(inputs=inputs, outputs=out)

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
            patience=10,
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


class MultiScaleLSTM:
    """多尺度 LSTM - 捕捉不同时间尺度的模式"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 25,
        batch_size: int = 32,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow 未安装")

        self.seq_length = seq_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = []

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建多尺度 LSTM"""
        inputs = keras.Input(shape=input_shape)

        # 短期分支（最近 5 天）
        short_term = inputs[:, -5:, :]
        short_lstm = keras.layers.LSTM(self.units // 2, return_sequences=False)(short_term)
        short_lstm = keras.layers.LayerNormalization()(short_lstm)

        # 中期分支（最近 10 天）
        mid_term = inputs[:, -10:, :]
        mid_lstm = keras.layers.LSTM(self.units // 2, return_sequences=False)(mid_term)
        mid_lstm = keras.layers.LayerNormalization()(mid_lstm)

        # 长期分支（全部 20 天）
        long_lstm = keras.layers.LSTM(self.units, return_sequences=False)(inputs)
        long_lstm = keras.layers.LayerNormalization()(long_lstm)

        # 融合
        fused = keras.layers.Concatenate()([short_lstm, mid_lstm, long_lstm])
        fused = keras.layers.Dropout(self.dropout)(fused)

        # 注意力加权
        attention_weights = keras.layers.Dense(3, activation='softmax')(fused)

        # 输出
        out = keras.layers.Dense(64, activation='relu')(fused)
        out = keras.layers.Dropout(self.dropout)(out)
        out = keras.layers.Dense(1)(out)

        model = keras.Model(inputs=inputs, outputs=out)

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
            patience=10,
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
    """测试创新架构"""
    print("\n" + "="*80)
    print("🚀 测试创新架构")
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

    print(f"训练集：{len(train_df)} 行")
    print(f"测试集：{len(test_df)} 行\n")

    results = []

    # 1. 残差注意力 LSTM
    print("="*80)
    print("🔧 训练残差注意力 LSTM...")
    print("="*80)

    model1 = ResidualAttentionLSTM(
        seq_length=20,
        units=64,
        dropout=0.3,
        learning_rate=0.001,
        epochs=25,
        batch_size=32
    )

    model1.fit(train_df[feature_cols + ["code", "date"]], train_df[target_col])
    pred1 = model1.predict(test_df[feature_cols + ["code", "date"]])
    actual = test_df[target_col].values

    acc1 = float(np.mean((pred1 > 0) == (actual > 0)))
    ic1 = float(spearmanr(pred1, actual)[0])

    print(f"  准确率：{acc1:.4f}")
    print(f"  IC: {ic1:.4f}\n")

    results.append({
        "model": "Residual Attention LSTM",
        "accuracy": acc1,
        "ic": ic1
    })

    # 2. 多尺度 LSTM
    print("="*80)
    print("🔧 训练多尺度 LSTM...")
    print("="*80)

    model2 = MultiScaleLSTM(
        seq_length=20,
        units=64,
        dropout=0.3,
        learning_rate=0.001,
        epochs=25,
        batch_size=32
    )

    model2.fit(train_df[feature_cols + ["code", "date"]], train_df[target_col])
    pred2 = model2.predict(test_df[feature_cols + ["code", "date"]])

    acc2 = float(np.mean((pred2 > 0) == (actual > 0)))
    ic2 = float(spearmanr(pred2, actual)[0])

    print(f"  准确率：{acc2:.4f}")
    print(f"  IC: {ic2:.4f}\n")

    results.append({
        "model": "Multi-Scale LSTM",
        "accuracy": acc2,
        "ic": ic2
    })

    # 保存结果
    output_dir = REPORTS_DIR / "innovative_architectures"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)

    print("\n" + "="*80)
    print("📊 结果对比")
    print("="*80 + "\n")

    for idx, row in results_df.iterrows():
        print(f"{idx+1}. {row['model']}")
        print(f"   准确率：{row['accuracy']:.4f}")
        print(f"   IC: {row['ic']:.4f}\n")

    best_acc = results_df['accuracy'].max()
    print(f"💡 最佳准确率：{best_acc:.4f}")

    print(f"\n💾 结果已保存：{output_dir}/results.csv\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
