"""Model training and evaluation helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame
    split_date: pd.Timestamp


def make_time_split(panel: pd.DataFrame, target_col: str, test_days: int = 252) -> SplitData:
    trainable = panel[panel["is_trainable"] & panel[target_col].notna()].copy()
    if trainable.empty:
        raise RuntimeError("No trainable rows with target values")
    dates = pd.Series(pd.to_datetime(trainable["date"].unique())).sort_values().reset_index(drop=True)
    if len(dates) < 40:
        raise RuntimeError(f"Too few unique trading dates for a time split: {len(dates)}")
    split_index = max(1, len(dates) - int(test_days))
    split_date = pd.Timestamp(dates.iloc[split_index])
    train = trainable[pd.to_datetime(trainable["date"]) < split_date].copy()
    test = trainable[pd.to_datetime(trainable["date"]) >= split_date].copy()
    if len(train) < 100 or len(test) < 20:
        # Fall back to a 75/25 chronological split for shorter newly-listed ETFs.
        split_index = max(1, int(len(dates) * 0.75))
        split_date = pd.Timestamp(dates.iloc[split_index])
        train = trainable[pd.to_datetime(trainable["date"]) < split_date].copy()
        test = trainable[pd.to_datetime(trainable["date"]) >= split_date].copy()
    if train.empty or test.empty:
        raise RuntimeError("Time split produced empty train or test set")
    return SplitData(train=train, test=test, split_date=split_date)


def make_model(model_type: str = "hgb", random_state: int = 42) -> Pipeline:
    """Create a regression model pipeline."""

    model_type = model_type.lower()
    if model_type in {"hgb", "hist_gradient_boosting", "gbdt"}:
        estimator: Any = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=260,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            early_stopping=True,
            random_state=random_state,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if model_type in {"rf", "random_forest"}:
        estimator = RandomForestRegressor(
            n_estimators=260,
            max_depth=8,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_state,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if model_type in {"ridge", "linear"}:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=2.0, random_state=random_state)),
            ]
        )
    raise ValueError(f"Unsupported model_type={model_type!r}")


def _spearman_by_date(df: pd.DataFrame, pred_col: str, target_col: str) -> float | None:
    values = []
    for _, group in df.groupby("date"):
        if len(group) < 2:
            continue
        corr = group[pred_col].corr(group[target_col], method="spearman")
        if pd.notna(corr):
            values.append(float(corr))
    if not values:
        return None
    return float(np.mean(values))


def _top_bottom_spread(df: pd.DataFrame, pred_col: str, target_col: str) -> dict[str, float | None]:
    top_returns: list[float] = []
    bottom_returns: list[float] = []
    all_returns: list[float] = []
    for _, group in df.groupby("date"):
        group = group.dropna(subset=[pred_col, target_col])
        if group.empty:
            continue
        top = group.nlargest(1, pred_col)[target_col].mean()
        bottom = group.nsmallest(1, pred_col)[target_col].mean()
        top_returns.append(float(top))
        bottom_returns.append(float(bottom))
        all_returns.append(float(group[target_col].mean()))
    if not top_returns:
        return {"top1_mean_fwd_ret": None, "bottom1_mean_fwd_ret": None, "top_bottom_spread": None, "top_vs_all": None}
    return {
        "top1_mean_fwd_ret": float(np.mean(top_returns)),
        "bottom1_mean_fwd_ret": float(np.mean(bottom_returns)),
        "top_bottom_spread": float(np.mean(top_returns) - np.mean(bottom_returns)),
        "top_vs_all": float(np.mean(top_returns) - np.mean(all_returns)),
    }


def evaluate_predictions(df: pd.DataFrame, pred_col: str, target_col: str) -> dict[str, Any]:
    y_true = df[target_col].astype(float)
    y_pred = df[pred_col].astype(float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    metrics: dict[str, Any] = {
        "rows": int(len(df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "mean_target": float(y_true.mean()),
        "mean_prediction": float(y_pred.mean()),
        "spearman_ic_by_date": _spearman_by_date(df, pred_col=pred_col, target_col=target_col),
    }
    metrics.update(_top_bottom_spread(df, pred_col=pred_col, target_col=target_col))
    return metrics


def fit_and_evaluate(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    model_type: str = "hgb",
    test_days: int = 252,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame, SplitData]:
    split = make_time_split(panel, target_col=target_col, test_days=test_days)
    model = make_model(model_type=model_type, random_state=random_state)
    x_train = split.train[feature_cols]
    y_train = split.train[target_col].astype(float)
    model.fit(x_train, y_train)

    test_predictions = split.test.copy()
    test_predictions["prediction"] = model.predict(test_predictions[feature_cols])
    metrics = evaluate_predictions(test_predictions, pred_col="prediction", target_col=target_col)
    metrics.update(
        {
            "model_type": model_type,
            "target_col": target_col,
            "feature_count": len(feature_cols),
            "train_rows": int(len(split.train)),
            "test_rows": int(len(split.test)),
            "train_start": str(pd.to_datetime(split.train["date"]).min().date()),
            "train_end": str(pd.to_datetime(split.train["date"]).max().date()),
            "test_start": str(pd.to_datetime(split.test["date"]).min().date()),
            "test_end": str(pd.to_datetime(split.test["date"]).max().date()),
            "split_date": str(split.split_date.date()),
        }
    )
    return model, metrics, test_predictions, split


def predict_latest(panel: pd.DataFrame, feature_cols: list[str], model: Pipeline, horizon: int) -> pd.DataFrame:
    latest_rows = panel.sort_values("date").groupby("code", as_index=False).tail(1).copy()
    latest_rows[f"pred_fwd_ret_{horizon}"] = model.predict(latest_rows[feature_cols])
    latest_rows["pred_score_rank"] = latest_rows[f"pred_fwd_ret_{horizon}"].rank(ascending=False, method="first")
    keep = [
        "pred_score_rank",
        "date",
        "code",
        "name",
        "family_id",
        "close",
        f"pred_fwd_ret_{horizon}",
        "ret_5",
        "ret_20",
        "vol_20",
        "drawdown_20",
        "amount_log",
        "turnover_rate",
    ]
    return latest_rows[[c for c in keep if c in latest_rows.columns]].sort_values("pred_score_rank")
