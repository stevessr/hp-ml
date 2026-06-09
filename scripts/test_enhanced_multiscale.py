#!/usr/bin/env python3
"""增强多尺度 LSTM - 5 个时间尺度

从 3 尺度扩展到 5 尺度：
- 3 天：超短期（日内波动）
- 5 天：短期（周内趋势）
- 10 天：中期（双周动量）
- 15 天：中长期（月中信息）
- 20 天：长期（完整月度）
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


class EnhancedMultiScaleLSTM:
    """增强多尺度 LSTM - 5 个时间尺度"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 30,
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
        """构建增强多尺度 LSTM"""
        inputs = keras.Input(shape=input_shape)

        # 5 个时间尺度分支
        scales = [3, 5, 10, 15, 20]
        scale_features = []

        for scale in scales:
            # 提取对应长度的序列
            if scale < self.seq_length:
                scale_input = inputs[:, -scale:, :]
            else:
                scale_input = inputs

            # LSTM 处理
            lstm_units = self.units // 4 if scale <= 5 else self.units // 3
            lstm_out = keras.layers.LSTM(
                lstm_units,
                return_sequences=False,
                name=f'lstm_scale_{scale}'
            )(scale_input)

            # Layer Normalization
            lstm_out = keras.layers.LayerNormalization()(lstm_out)
            scale_features.append(lstm_out)

        # 融合所有尺度
        fused = keras.layers.Concatenate()(scale_features)
        fused = keras.layers.Dropout(self.dropout)(fused)

        # 注意力权重（学习各尺度的重要性）
        scale_weights = keras.layers.Dense(
            len(scales),
            activation='softmax',
            name='scale_attention'
        )(fused)

        # 加权融合（可选，这里使用简单拼接）
        # 深度融合层
        fusion = keras.layers.Dense(self.units, activation='relu')(fused)
        fusion = keras.layers.LayerNormalization()(fusion)
        fusion = keras.layers.Dropout(self.dropout)(fusion)

        # 输出层
        out = keras.layers.Dense(32, activation='relu')(fusion)
        out = keras.layers.Dropout(self.dropout / 2)(out)
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

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
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


class MultiScaleAttentionLSTM:
    """多尺度注意力 LSTM - 在每个尺度上添加注意力"""

    def __init__(
        self,
        seq_length: int = 20,
        units: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 30,
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

    def _attention_block(self, lstm_output, name_prefix):
        """注意力块"""
        # 注意力权重
        attention = keras.layers.Dense(1, activation='tanh', name=f'{name_prefix}_attn_dense')(lstm_output)
        attention = keras.layers.Flatten(name=f'{name_prefix}_attn_flatten')(attention)
        attention = keras.layers.Activation('softmax', name=f'{name_prefix}_attn_softmax')(attention)
        attention = keras.layers.RepeatVector(self.units // 2, name=f'{name_prefix}_attn_repeat')(attention)
        attention = keras.layers.Permute([2, 1], name=f'{name_prefix}_attn_permute')(attention)

        # 加权
        attended = keras.layers.Multiply(name=f'{name_prefix}_attn_multiply')([lstm_output, attention])
        attended = keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=1),
            name=f'{name_prefix}_attn_sum'
        )(attended)

        return attended

    def _build_model(self, input_shape: tuple[int, int]) -> keras.Model:
        """构建多尺度注意力 LSTM"""
        inputs = keras.Input(shape=input_shape)

        # 3 个主要时间尺度（带注意力）
        scales = [5, 10, 20]
        scale_features = []

        for i, scale in enumerate(scales):
            # 提取序列
            if scale < self.seq_length:
                scale_input = inputs[:, -scale:, :]
            else:
                scale_input = inputs

            # LSTM with return_sequences=True (为了注意力)
            lstm_out = keras.layers.LSTM(
                self.units // 2,
                return_sequences=True,
                name=f'lstm_scale_{scale}'
            )(scale_input)

            # 注意力机制
            attended = self._attention_block(lstm_out, f'scale_{scale}')

            # Layer Normalization
            attended = keras.layers.LayerNormalization()(attended)
            scale_features.append(attended)

        # 融合
        fused = keras.layers.Concatenate()(scale_features)
        fused = keras.layers.Dropout(self.dropout)(fused)

        # 深度融合
        fusion = keras.layers.Dense(self.units, activation='relu')(fused)
        fusion = keras.layers.LayerNormalization()(fusion)
        fusion = keras.layers.Dropout(self.dropout)(fusion)

        # 输出
        out = keras.layers.Dense(32, activation='relu')(fusion)
        out = keras.layers.Dropout(self.dropout / 2)(out)
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

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
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
    """测试增强架构"""
    print("\n" + "="*80)
    print("🚀 测试增强多尺度架构")
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

    # 1. 增强多尺度 LSTM (5 个尺度)
    print("="*80)
    print("🔧 训练增强多尺度 LSTM (5 尺度)...")
    print("="*80)

    model1 = EnhancedMultiScaleLSTM(
        seq_length=20,
        units=64,
        dropout=0.3,
        learning_rate=0.001,
        epochs=30,
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
        "model": "Enhanced Multi-Scale LSTM (5 scales)",
        "accuracy": acc1,
        "ic": ic1
    })

    # 2. 多尺度注意力 LSTM
    print("="*80)
    print("🔧 训练多尺度注意力 LSTM...")
    print("="*80)

    model2 = MultiScaleAttentionLSTM(
        seq_length=20,
        units=64,
        dropout=0.3,
        learning_rate=0.001,
        epochs=30,
        batch_size=32
    )

    model2.fit(train_df[feature_cols + ["code", "date"]], train_df[target_col])
    pred2 = model2.predict(test_df[feature_cols + ["code", "date"]])

    acc2 = float(np.mean((pred2 > 0) == (actual > 0)))
    ic2 = float(spearmanr(pred2, actual)[0])

    print(f"  准确率：{acc2:.4f}")
    print(f"  IC: {ic2:.4f}\n")

    results.append({
        "model": "Multi-Scale Attention LSTM",
        "accuracy": acc2,
        "ic": ic2
    })

    # 保存结果
    output_dir = REPORTS_DIR / "enhanced_architectures"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)

    print("\n" + "="*80)
    print("📊 结果对比")
    print("="*80 + "\n")

    # 添加基线对比
    print("基线对比：")
    print(f"  多尺度 LSTM (3 尺度): 68.08%, IC: 0.3976\n")

    print("新架构：")
    for idx, row in results_df.iterrows():
        print(f"{idx+1}. {row['model']}")
        print(f"   准确率：{row['accuracy']:.4f}")
        print(f"   IC: {row['ic']:.4f}\n")

    best_acc = results_df['accuracy'].max()
    print(f"💡 最佳准确率：{best_acc:.4f}")

    if best_acc > 0.6808:
        improvement = (best_acc - 0.6808) * 100
        print(f"💡 超越基线：+{improvement:.2f}%")
        print(f"\n✅ 架构增强成功！")
    else:
        print(f"\n⚠️ 未超越基线 (68.08%)")

    print(f"\n💾 结果已保存：{output_dir}/results.csv\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
