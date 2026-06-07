"""统一数据划分管道：训练集、验证集、测试集"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    """数据划分结果"""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    val_dates: tuple[pd.Timestamp, pd.Timestamp]
    test_dates: tuple[pd.Timestamp, pd.Timestamp]


def time_series_split(
    panel: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> DataSplit:
    """
    时间序列数据划分

    Args:
        panel: 包含日期列的面板数据
        target_col: 目标列名称
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        DataSplit: 划分后的数据集
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为1，当前为 {train_ratio + val_ratio + test_ratio}")

    trainable = panel[panel["is_trainable"] & panel[target_col].notna()].copy()
    if trainable.empty:
        raise RuntimeError("没有可训练的数据行")

    dates = pd.Series(pd.to_datetime(trainable["date"].unique())).sort_values().reset_index(drop=True)
    if len(dates) < 30:
        raise RuntimeError(f"可用日期太少，无法划分: {len(dates)} 天")

    n_dates = len(dates)
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))

    train_end_idx = max(10, min(train_end_idx, n_dates - 20))
    val_end_idx = max(train_end_idx + 5, min(val_end_idx, n_dates - 10))

    train_end_date = pd.Timestamp(dates.iloc[train_end_idx])
    val_end_date = pd.Timestamp(dates.iloc[val_end_idx])

    trainable["date_ts"] = pd.to_datetime(trainable["date"])
    train_df = trainable[trainable["date_ts"] < train_end_date].copy()
    val_df = trainable[(trainable["date_ts"] >= train_end_date) & (trainable["date_ts"] < val_end_date)].copy()
    test_df = trainable[trainable["date_ts"] >= val_end_date].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("数据划分后某个集合为空")

    return DataSplit(
        train=train_df.drop(columns=["date_ts"]),
        val=val_df.drop(columns=["date_ts"]),
        test=test_df.drop(columns=["date_ts"]),
        train_dates=(train_df["date_ts"].min(), train_df["date_ts"].max()),
        val_dates=(val_df["date_ts"].min(), val_df["date_ts"].max()),
        test_dates=(test_df["date_ts"].min(), test_df["date_ts"].max()),
    )


def expanding_window_cv(
    panel: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    min_train_size: int = 252,
    test_size: int = 63,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    扩展窗口交叉验证（walk-forward）

    Args:
        panel: 面板数据
        target_col: 目标列
        n_splits: 划分数量
        min_train_size: 最小训练集大小（交易日）
        test_size: 测试集大小（交易日）

    Returns:
        list: 每个元素为 (train_df, test_df) 元组
    """
    trainable = panel[panel["is_trainable"] & panel[target_col].notna()].copy()
    dates = pd.Series(pd.to_datetime(trainable["date"].unique())).sort_values().reset_index(drop=True)

    if len(dates) < min_train_size + test_size * n_splits:
        raise ValueError(f"数据不足以支持 {n_splits} 次交叉验证")

    splits = []
    trainable["date_ts"] = pd.to_datetime(trainable["date"])

    for i in range(n_splits):
        train_end_idx = min_train_size + i * test_size
        test_end_idx = train_end_idx + test_size

        if test_end_idx > len(dates):
            break

        train_end_date = pd.Timestamp(dates.iloc[train_end_idx])
        test_end_date = pd.Timestamp(dates.iloc[test_end_idx])

        train_df = trainable[trainable["date_ts"] < train_end_date].copy()
        test_df = trainable[(trainable["date_ts"] >= train_end_date) & (trainable["date_ts"] < test_end_date)].copy()

        if not train_df.empty and not test_df.empty:
            splits.append((train_df.drop(columns=["date_ts"]), test_df.drop(columns=["date_ts"])))

    return splits


def prepare_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_length: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    为 LSTM 准备序列数据

    Args:
        df: 数据框，需要按 code 和 date 排序
        feature_cols: 特征列
        target_col: 目标列
        seq_length: 序列长度

    Returns:
        (X, y): X 的形状为 (样本数, seq_length, 特征数)，y 的形状为 (样本数,)
    """
    X_list = []
    y_list = []

    for code, group in df.groupby("code"):
        group = group.sort_values("date").reset_index(drop=True)

        if len(group) < seq_length + 1:
            continue

        features = group[feature_cols].fillna(0).values
        targets = group[target_col].fillna(0).values

        for i in range(len(group) - seq_length):
            X_list.append(features[i:i + seq_length])
            y_list.append(targets[i + seq_length])

    if not X_list:
        raise ValueError("没有足够的序列数据")

    return np.array(X_list), np.array(y_list)
