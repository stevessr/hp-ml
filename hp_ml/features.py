"""Feature engineering for ETF time-series panels."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLL_WINDOWS = (3, 5, 10, 20, 60)
FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "vol_5",
    "vol_20",
    "vol_60",
    "ma_gap_5_20",
    "ma_gap_20_60",
    "drawdown_20",
    "drawdown_60",
    "amount_log",
    "amount_z20",
    "turnover_rate",
    "amplitude",
    "intraday_range",
    "volume_chg_5",
    "liquidity_shock_20",
    "month_sin",
    "month_cos",
    "days_since_start",
]


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods, fill_method=None).replace([np.inf, -np.inf], np.nan)


def add_time_series_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create one ETF feature frame sorted by date."""

    out = df.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ["open", "close", "high", "low", "volume", "amount", "amplitude", "turnover_rate"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    close = out["close"].replace(0, np.nan)
    daily_ret = _safe_pct_change(close, 1)
    out["ret_1"] = daily_ret
    for window in ROLL_WINDOWS:
        out[f"ret_{window}"] = _safe_pct_change(close, window)

    for window in (5, 20, 60):
        out[f"vol_{window}"] = daily_ret.rolling(window, min_periods=max(3, window // 3)).std()

    ma5 = close.rolling(5, min_periods=3).mean()
    ma20 = close.rolling(20, min_periods=8).mean()
    ma60 = close.rolling(60, min_periods=20).mean()
    out["ma_gap_5_20"] = ma5 / ma20 - 1
    out["ma_gap_20_60"] = ma20 / ma60 - 1

    for window in (20, 60):
        rolling_max = close.rolling(window, min_periods=max(5, window // 3)).max()
        out[f"drawdown_{window}"] = close / rolling_max - 1

    amount = out.get("amount", pd.Series(index=out.index, dtype="float64")).astype(float)
    volume = out.get("volume", pd.Series(index=out.index, dtype="float64")).astype(float)
    out["amount_log"] = np.log1p(amount.clip(lower=0))
    amount_mean20 = amount.rolling(20, min_periods=5).mean()
    amount_std20 = amount.rolling(20, min_periods=5).std()
    out["amount_z20"] = (amount - amount_mean20) / amount_std20.replace(0, np.nan)
    out["volume_chg_5"] = _safe_pct_change(volume.replace(0, np.nan), 5)
    out["liquidity_shock_20"] = amount / amount_mean20.replace(0, np.nan) - 1

    out["turnover_rate"] = pd.to_numeric(out.get("turnover_rate"), errors="coerce")
    out["amplitude"] = pd.to_numeric(out.get("amplitude"), errors="coerce")
    out["intraday_range"] = (out["high"] - out["low"]) / close

    month = out["date"].dt.month.astype(float)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["days_since_start"] = (out["date"] - out["date"].min()).dt.days.astype(float)

    out[f"fwd_ret_{horizon}"] = close.shift(-horizon) / close - 1
    return out.replace([np.inf, -np.inf], np.nan)


def build_feature_panel(histories: dict[str, pd.DataFrame], universe: pd.DataFrame, horizon: int = 5) -> tuple[pd.DataFrame, list[str], str]:
    """Build a model-ready panel and return ``(panel, feature_cols, target_col)``."""

    meta = universe.copy()
    meta["code"] = meta["code"].astype(str).str.zfill(6)
    frames: list[pd.DataFrame] = []
    family_ids = sorted(meta["family_id"].dropna().unique().tolist())

    for code, hist in histories.items():
        if hist.empty:
            continue
        code = str(code).zfill(6)

        # 确保历史数据有 code 列
        hist_copy = hist.copy()
        if 'code' not in hist_copy.columns:
            hist_copy['code'] = code

        family_id = meta.loc[meta["code"] == code, "family_id"]
        name = meta.loc[meta["code"] == code, "name"]
        family_id_value = str(family_id.iloc[0]) if not family_id.empty else "UNKNOWN"
        frame = add_time_series_features(hist_copy, horizon=horizon)
        frame["code"] = code  # 确保 code 列存在
        frame["family_id"] = family_id_value
        frame["name"] = str(name.iloc[0]) if not name.empty else code
        for fam in family_ids:
            frame[f"family_{fam}"] = 1.0 if family_id_value == fam else 0.0
        frames.append(frame)

    if not frames:
        raise RuntimeError("No history frames available to build features")

    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
    family_feature_cols = [f"family_{fam}" for fam in family_ids]
    feature_cols = [c for c in FEATURE_COLUMNS + family_feature_cols if c in panel.columns]
    target_col = f"fwd_ret_{horizon}"
    panel["is_trainable"] = panel[target_col].notna()
    return panel, feature_cols, target_col
