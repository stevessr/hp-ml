"""Deep-dive automatically discovered ETF signals into related bullish stocks.

The ETF training pipeline finds broad-index ETF candidates and predicts the
strongest near-term ETF/family signals.  This module follows those signals down
to the component-stock layer, then enriches the most bullish stock candidates
with live trading amount/turnover, recent K-line momentum, and latest top-holder
ratios from Eastmoney Data Center.

Outputs are research artifacts only; they are not investment advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .charts import write_column_chart, write_horizontal_bar_chart, write_line_chart, write_pie_chart
from .config import RAW_DIR, REPORTS_DIR
from .data_sources import (
    EASTMONEY_KLINE_URL,
    HISTORY_COLUMNS,
    TENCENT_KLINE_URL,
    _get_json,
    human_amount,
    latest_cache_path,
    make_session,
    secid_for_code,
    today_yyyymmdd,
)

EASTMONEY_STOCK_LIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_STOCK_ULIST_URL = "https://push2.eastmoney.com/api/qt/ulist.np/get"
EASTMONEY_DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
TENCENT_REALTIME_URL = "https://qt.gtimg.cn/q="

# Shanghai main/KCB and Shenzhen main/ChiNext.  The index-component data used by
# this project is currently concentrated here.  Beijing symbols can be added if
# future CSI component endpoints include them.
A_SHARE_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"

STOCK_SPOT_FIELDS = {
    "f12": "stock_code",
    "f14": "stock_name_quote",
    "f2": "latest_price",
    "f3": "quote_pct_chg",
    "f4": "quote_price_change",
    "f5": "quote_volume",
    "f6": "quote_amount",
    "f7": "quote_amplitude",
    "f8": "quote_turnover_rate",
    "f9": "pe_ttm",
    "f10": "volume_ratio",
    "f15": "quote_high",
    "f16": "quote_low",
    "f17": "quote_open",
    "f18": "quote_prev_close",
    "f20": "total_market_cap",
    "f21": "float_market_cap",
    "f23": "pb",
    "f62": "main_net_inflow",
    "f184": "main_net_inflow_pct",
    "f124": "quote_timestamp",
}

STOCK_HISTORY_DIR = RAW_DIR / "stock_history"
STOCK_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_NOTE = {
    "stock_spot": EASTMONEY_STOCK_LIST_URL,
    "stock_kline": EASTMONEY_KLINE_URL,
    "shareholder": EASTMONEY_DATACENTER_URL,
    "shareholder_reports": "RPT_F10_EH_FREEHOLDERS / RPT_DMSK_HOLDERS",
}


# ---------------------------------------------------------------------------
# Generic helpers


def to_float(value: Any, default: float | None = None) -> float | None:
    """Convert finance API values to finite floats."""

    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if value in {"", "-", "--", "None", "nan", "NaN"}:
            return default
    try:
        result = float(value)
    except Exception:  # noqa: BLE001 - finance APIs return mixed scalar types
        return default
    return result if math.isfinite(result) else default


def _date_only(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return text[:10]


def _fmt_ratio(value: Any) -> str:
    v = to_float(value)
    return "" if v is None else f"{v * 100:.2f}%"


def _fmt_pct_value(value: Any) -> str:
    v = to_float(value)
    return "" if v is None else f"{v:.2f}%"


def _fmt_number(value: Any, digits: int = 2) -> str:
    v = to_float(value)
    return "" if v is None else f"{v:.{digits}f}"


def _normalize_code(value: Any) -> str:
    return str(value or "").strip().zfill(6)


def _tencent_symbol(code: str) -> str:
    code = _normalize_code(code)
    return ("sh" if code.startswith(("5", "6", "9")) else "sz") + code


def _safe_join(values: list[Any], sep: str = ";") -> str:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return sep.join(result)


def _safe_filename(value: Any, *, max_len: int = 80) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "unknown")[:max_len]


def _parse_end_date(end: str | None) -> dt.date:
    if not end:
        return dt.date.today()
    text = str(end)
    if "-" in text:
        return dt.date.fromisoformat(text[:10])
    return dt.date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _history_start_for_end(end: str | None, history_days: int) -> str:
    # Calendar days ~= trading days * 1.8 plus a buffer, which keeps the cache
    # compact while leaving enough observations for 60/120-day features.
    end_date = _parse_end_date(end)
    calendar_days = max(120, int(history_days * 1.9) + 45)
    return (end_date - dt.timedelta(days=calendar_days)).strftime("%Y%m%d")


def _latest_quote_time(row: pd.Series) -> str:
    ts = to_float(row.get("quote_timestamp"))
    if ts is None or ts <= 0:
        return ""
    try:
        return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# ETF signal and related-stock selection


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ETF prediction file not found: {path}")
    df = pd.read_csv(path, dtype={"code": str})
    if "pred_fwd_ret_5" not in df.columns:
        # Allow candidate-only files as a fallback; they will be treated as
        # neutral ETF signals ranked by existing candidate rank.
        df["pred_fwd_ret_5"] = 0.0
    if "pred_score_rank" not in df.columns:
        rank_col = "selected_rank_in_family" if "selected_rank_in_family" in df.columns else None
        if rank_col:
            df["pred_score_rank"] = pd.to_numeric(df[rank_col], errors="coerce")
        else:
            df["pred_score_rank"] = range(1, len(df) + 1)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["pred_fwd_ret_5"] = pd.to_numeric(df["pred_fwd_ret_5"], errors="coerce").fillna(0.0)
    df["pred_score_rank"] = pd.to_numeric(df["pred_score_rank"], errors="coerce").fillna(len(df) + 1)
    return df


def select_etf_signals(
    predictions: pd.DataFrame,
    *,
    top_etfs: int = 8,
    min_etf_pred: float | None = 0.0,
    include_top: bool = True,
) -> pd.DataFrame:
    """Return ETF rows that define the stock deep-dive universe.

    A strict positive-prediction filter can be too sparse when the model is in a
    risk-off regime.  Therefore the default keeps all non-negative predictions
    and also the highest-ranked ETF signals, deduplicating by ETF code.  The
    returned rows carry ``signal_reason`` for traceability.
    """

    if predictions.empty:
        return predictions.copy()
    ordered = predictions.sort_values(["pred_fwd_ret_5", "pred_score_rank"], ascending=[False, True]).copy()
    frames: list[pd.DataFrame] = []
    if min_etf_pred is not None:
        positive = ordered[ordered["pred_fwd_ret_5"] >= float(min_etf_pred)].copy()
        if not positive.empty:
            positive["signal_reason"] = f"pred>= {float(min_etf_pred):.4f}"
            frames.append(positive)
    if include_top and top_etfs > 0:
        top = ordered.head(int(top_etfs)).copy()
        top["signal_reason"] = f"top{int(top_etfs)} ETF signal"
        frames.append(top)
    if frames:
        selected = pd.concat(frames, ignore_index=True, sort=False)
        selected = selected.drop_duplicates("code", keep="first")
    else:
        selected = ordered.head(max(1, int(top_etfs))).copy()
        selected["signal_reason"] = "fallback_top_rank"
    return selected.sort_values(["pred_score_rank", "pred_fwd_ret_5"], ascending=[True, False]).reset_index(drop=True)


def load_related_stocks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Related-stock file not found: {path}")
    df = pd.read_csv(path, dtype={"stock_code": str, "index_component_type": str})
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    for col in ("weight", "change_rate", "free_cap", "pe", "eps", "roe", "rank_in_family"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def refresh_related_stocks_from_predictions(predictions: pd.DataFrame, per_family: int, force: bool = False) -> pd.DataFrame:
    """Fetch related stocks for the families present in ETF predictions."""

    # The lightweight training module already has a stdlib-only component-stock
    # fetcher.  Import lazily so this deep-dive module stays importable for tests
    # that do not need live data.
    from .lite_train import fetch_related_stocks, write_csv_rows  # noqa: PLC0415

    universe = []
    seen: set[str] = set()
    for _, row in predictions.iterrows():
        family_id = str(row.get("family_id") or "")
        if not family_id or family_id in seen:
            continue
        seen.add(family_id)
        universe.append(
            {
                "code": str(row.get("code") or "").zfill(6),
                "name": row.get("name") or "",
                "family_id": family_id,
                "display_name": row.get("display_name") or family_id,
                "index_code": row.get("index_code") or "",
            }
        )
    rows = fetch_related_stocks(universe, per_family=per_family, force=force)
    out = REPORTS_DIR / "related_stocks_lite.csv"
    write_csv_rows(out, rows)
    return pd.DataFrame(rows)


def _family_signal_map(etf_signals: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if etf_signals.empty:
        return result
    ordered = etf_signals.sort_values(["family_id", "pred_score_rank", "pred_fwd_ret_5"], ascending=[True, True, False])
    for family_id, group in ordered.groupby("family_id", sort=False):
        best = group.sort_values(["pred_fwd_ret_5", "pred_score_rank"], ascending=[False, True]).iloc[0]
        result[str(family_id)] = {
            "source_etf_code": str(best.get("code") or "").zfill(6),
            "source_etf_name": best.get("name") or "",
            "source_etf_date": best.get("date") or "",
            "source_etf_pred_fwd_ret_5": to_float(best.get("pred_fwd_ret_5"), 0.0),
            "source_etf_pred_rank": to_float(best.get("pred_score_rank"), 9999.0),
            "source_etf_signal_reason": best.get("signal_reason") or "",
        }
    return result


def build_stock_candidates(
    related_stocks: pd.DataFrame,
    etf_signals: pd.DataFrame,
    *,
    stocks_per_family: int = 25,
    max_candidates: int = 80,
) -> pd.DataFrame:
    """Map selected ETF/family signals to unique component stock candidates."""

    if related_stocks.empty or etf_signals.empty:
        return pd.DataFrame()
    signal_map = _family_signal_map(etf_signals)
    selected_families = set(signal_map)
    df = related_stocks[related_stocks["family_id"].astype(str).isin(selected_families)].copy()
    if df.empty:
        return pd.DataFrame()
    for col in ("rank_in_family", "weight", "free_cap", "change_rate", "pe", "eps", "roe"):
        if col not in df.columns:
            df[col] = math.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["family_id", "rank_in_family", "weight", "free_cap"], ascending=[True, True, False, False])
    if stocks_per_family > 0:
        df = df.groupby("family_id", group_keys=False).head(int(stocks_per_family)).copy()

    grouped: list[dict[str, Any]] = []
    for code, group in df.groupby("stock_code", sort=False):
        code = _normalize_code(code)
        if not code.isdigit() or code == "000000":
            continue
        family_ids = [str(v) for v in group["family_id"].tolist()]
        signal_rows = [signal_map[fid] for fid in family_ids if fid in signal_map]
        best_signal = sorted(signal_rows, key=lambda r: (-(to_float(r.get("source_etf_pred_fwd_ret_5"), 0.0) or 0.0), to_float(r.get("source_etf_pred_rank"), 9999.0) or 9999.0))[0]
        first = group.iloc[0]
        grouped.append(
            {
                "stock_code": code,
                "stock_name": first.get("stock_name") or first.get("SECURITY_NAME_ABBR") or "",
                "family_ids": _safe_join(family_ids),
                "display_names": _safe_join([row for row in group.get("display_name", pd.Series(dtype=str)).tolist()]),
                "industry": first.get("industry") or "",
                "region": first.get("region") or "",
                "component_rank_best": to_float(group["rank_in_family"].min()),
                "index_weight_pct_max": to_float(group["weight"].max()),
                "component_free_cap_max": to_float(group["free_cap"].max()),
                "component_pe": to_float(first.get("pe")),
                "component_eps": to_float(first.get("eps")),
                "component_roe_pct": to_float(first.get("roe")),
                "component_trade_date": first.get("trade_date") or "",
                "source_etf_codes": _safe_join([row.get("source_etf_code") for row in signal_rows]),
                "source_etf_names": _safe_join([row.get("source_etf_name") for row in signal_rows]),
                "source_etf_dates": _safe_join([row.get("source_etf_date") for row in signal_rows]),
                "source_etf_pred_fwd_ret_5_best": to_float(best_signal.get("source_etf_pred_fwd_ret_5"), 0.0),
                "source_etf_pred_rank_best": to_float(best_signal.get("source_etf_pred_rank"), 9999.0),
                "source_etf_signal_reasons": _safe_join([row.get("source_etf_signal_reason") for row in signal_rows]),
            }
        )
    out = pd.DataFrame(grouped)
    if out.empty:
        return out
    out = out.sort_values(
        ["source_etf_pred_rank_best", "source_etf_pred_fwd_ret_5_best", "component_rank_best", "component_free_cap_max"],
        ascending=[True, False, True, False],
    )
    if max_candidates > 0:
        out = out.head(int(max_candidates)).copy()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trading/quote enrichment


def fetch_stock_spot(codes: list[str] | None = None, *, force: bool = False) -> pd.DataFrame:
    """Fetch/cache the A-share spot quote table and optionally filter codes."""

    codes = [_normalize_code(code) for code in (codes or [])]
    code_set = set(codes)
    if code_set:
        return fetch_stock_spot_by_codes(codes, force=force)
    cache_path = RAW_DIR / f"stock_spot_{today_yyyymmdd()}.csv"
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path, dtype={"stock_code": str})
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        return df

    try:
        session = make_session()
        rows: list[dict[str, Any]] = []
        page_size = 500
        total: int | None = None
        page = 1
        params = {
            "pn": page,
            "pz": page_size,
            "po": 1,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f6",
            "fs": A_SHARE_FS,
            "fields": ",".join(STOCK_SPOT_FIELDS),
        }
        while True:
            params["pn"] = page
            payload = _get_json(session, EASTMONEY_STOCK_LIST_URL, params=params, timeout=20)
            data = payload.get("data") or {}
            page_rows = data.get("diff") or []
            if total is None:
                total = int(data.get("total") or 0) or None
            if not page_rows:
                break
            rows.extend(page_rows)
            if total is not None and len(rows) >= total:
                break
            if len(page_rows) < page_size:
                break
            page += 1
        if not rows:
            raise RuntimeError("Eastmoney stock quote endpoint returned no rows")
    except Exception:
        latest = None if force else latest_cache_path("stock_spot_*.csv", exclude=cache_path)
        if latest is not None:
            df = pd.read_csv(latest, dtype={"stock_code": str})
            df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
            return df[df["stock_code"].isin(code_set)].copy() if code_set else df
        raise

    df = _spot_rows_to_dataframe(rows)
    df.to_csv(cache_path, index=False)
    return df


def _spot_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).rename(columns=STOCK_SPOT_FIELDS)
    keep_cols = list(STOCK_SPOT_FIELDS.values())
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    if df.empty:
        return df
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    for col in [c for c in df.columns if c not in {"stock_code", "stock_name_quote"}]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fetched_at"] = dt.datetime.now().isoformat(timespec="seconds")
    if "quote_timestamp" in df.columns:
        df["quote_time"] = df.apply(_latest_quote_time, axis=1)
    return df


def fetch_stock_spot_by_codes(codes: list[str], *, force: bool = False) -> pd.DataFrame:
    """Fetch/cache spot quote rows for selected stock codes.

    The full A-share list endpoint can occasionally reject large paginated
    requests through proxies.  The ulist endpoint accepts explicit ``secids``
    and is a lighter, more reliable path for this deep-dive workflow.
    """

    codes = [_normalize_code(code) for code in codes]
    code_set = set(codes)
    cache_path = RAW_DIR / f"stock_spot_selected_{today_yyyymmdd()}.csv"
    cached = pd.DataFrame()
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path, dtype={"stock_code": str})
        cached["stock_code"] = cached["stock_code"].astype(str).str.zfill(6)
        cached_subset = cached[cached["stock_code"].isin(code_set)].copy()
        has_requested_codes = code_set.issubset(set(cached_subset["stock_code"]))
        has_quote_amount = "quote_amount" in cached_subset.columns and cached_subset["quote_amount"].notna().any()
        if has_requested_codes and has_quote_amount:
            return cached_subset

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    rows: list[dict[str, Any]] = []
    # Keep the URL comfortably short; Eastmoney accepts comma-separated secids.
    chunk_size = 3
    for i in range(0, len(codes), chunk_size):
        chunk = codes[i : i + chunk_size]
        try:
            response = session.get(
                EASTMONEY_STOCK_ULIST_URL,
                params={
                    "fltt": 2,
                    "invt": 2,
                    "fields": ",".join(STOCK_SPOT_FIELDS),
                    "secids": ",".join(secid_for_code(code) for code in chunk),
                },
                timeout=8,
            )
            response.raise_for_status()
            payload = response.json()
            rows.extend(((payload.get("data") or {}).get("diff")) or [])
        except Exception:
            # Quote enrichment is useful but not required: the K-line path below
            # still provides amount/turnover features.  Avoid long proxy retry
            # cascades by falling back to placeholder rows for this chunk.
            for code in chunk:
                rows.append({"f12": code})
    df = _spot_rows_to_dataframe(rows)
    if df.empty or "quote_amount" not in df.columns or df["quote_amount"].isna().all():
        try:
            tencent = fetch_stock_spot_tencent_by_codes(codes)
            if df.empty:
                df = tencent
            else:
                df = pd.concat([df, tencent], ignore_index=True, sort=False)
                df = df.sort_values(
                    ["stock_code", "quote_amount"],
                    ascending=[True, True],
                    na_position="first",
                ).drop_duplicates("stock_code", keep="last")
        except Exception:
            pass
    if not cached.empty:
        df = pd.concat([cached, df], ignore_index=True, sort=False)
        df = df.sort_values(
            ["stock_code", "quote_amount"],
            ascending=[True, True],
            na_position="first",
        ).drop_duplicates("stock_code", keep="last")
    df.to_csv(cache_path, index=False)
    return df[df["stock_code"].isin(code_set)].copy()


def fetch_stock_spot_tencent_by_codes(codes: list[str]) -> pd.DataFrame:
    """Fetch selected real-time quote rows from Tencent as an amount fallback."""

    rows: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    for i in range(0, len(codes), 80):
        chunk = [_tencent_symbol(code) for code in codes[i : i + 80]]
        response = session.get(TENCENT_REALTIME_URL + ",".join(chunk), timeout=10)
        response.raise_for_status()
        text = response.content.decode("gbk", "ignore")
        for line in text.splitlines():
            if '="' not in line:
                continue
            payload = line.split('="', 1)[1].rsplit('"', 1)[0]
            parts = payload.split("~")
            if len(parts) < 40:
                continue
            code = _normalize_code(parts[2])
            trade_parts = parts[35].split("/") if len(parts) > 35 else []
            amount = to_float(trade_parts[2]) if len(trade_parts) >= 3 else None
            volume = to_float(trade_parts[1]) if len(trade_parts) >= 2 else to_float(parts[36])
            amount_10k = to_float(parts[37]) if len(parts) > 37 else None
            if amount is None and amount_10k is not None:
                amount = amount_10k * 10000
            rows.append(
                {
                    "stock_code": code,
                    "stock_name_quote": parts[1],
                    "latest_price": to_float(parts[3]),
                    "quote_pct_chg": to_float(parts[32]),
                    "quote_price_change": to_float(parts[31]),
                    "quote_volume": volume,
                    "quote_amount": amount,
                    "quote_high": to_float(parts[33]),
                    "quote_low": to_float(parts[34]),
                    "quote_open": to_float(parts[5]),
                    "quote_prev_close": to_float(parts[4]),
                    "quote_turnover_rate": to_float(parts[38]),
                    "pe_ttm": to_float(parts[39]),
                    "quote_time": parts[30],
                    "source": "tencent_realtime",
                    "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
    return pd.DataFrame(rows)


def fetch_stock_history(
    code: str,
    *,
    start: str,
    end: str | None = None,
    adjust: str = "qfq",
    force: bool = False,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """Fetch/cache daily A-share K-line history from Eastmoney."""

    code = _normalize_code(code)
    end = end or today_yyyymmdd()
    adjust_key = {"": 0, "none": 0, "raw": 0, "qfq": 1, "hfq": 2}.get(adjust.lower(), 1)
    cache_path = STOCK_HISTORY_DIR / f"{code}_{start}_{end}_{adjust or 'raw'}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, dtype={"stock_code": str}, parse_dates=["date"])
    try:
        if sleep_seconds:
            time.sleep(sleep_seconds)
        session = make_session()
        payload = _get_json(
            session,
            EASTMONEY_KLINE_URL,
            params={
                "secid": secid_for_code(code),
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": 101,
                "fqt": adjust_key,
                "beg": start,
                "end": end,
            },
            timeout=15,
        )
        klines = ((payload.get("data") or {}).get("klines")) or []
        if not klines:
            raise RuntimeError(f"No stock K-line rows for {code}")
        records: list[list[str]] = [line.split(",") for line in klines]
        df = pd.DataFrame(records, columns=HISTORY_COLUMNS)
        df.insert(0, "stock_code", code)
        df["date"] = pd.to_datetime(df["date"])
        for col in HISTORY_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("date").drop_duplicates(["stock_code", "date"], keep="last")
    except Exception:
        df = fetch_stock_history_tencent(code, start=start, end=end, adjust=adjust)
    df.to_csv(cache_path, index=False)
    return df


def fetch_stock_history_tencent(code: str, *, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """Fetch stock daily history from Tencent and approximate amount.

    Tencent's daily endpoint returns volume in lots and no explicit amount.
    ``amount`` is approximated as close * volume * 100, which is sufficient for
    detecting amount expansion when Eastmoney K-line is unavailable.
    """

    code = _normalize_code(code)
    symbol = _tencent_symbol(code)
    fq = "qfq" if adjust.lower() == "qfq" else ""
    param = f"{symbol},day,{_date_only(pd.to_datetime(start))},{_date_only(pd.to_datetime(end))},1000,{fq}".rstrip(",")
    payload = _get_json(make_session(), TENCENT_KLINE_URL, params={"param": param}, timeout=12)
    node = (payload.get("data") or {}).get(symbol) or {}
    lines = node.get("qfqday") or node.get("hfqday") or node.get("day") or []
    if not lines:
        raise RuntimeError(f"No Tencent stock K-line rows for {code}")
    rows: list[dict[str, Any]] = []
    prev_close: float | None = None
    for item in lines:
        if len(item) < 6:
            continue
        date_s, open_v, close_v, high_v, low_v, volume_v = item[:6]
        close_f = to_float(close_v)
        high_f = to_float(high_v)
        low_f = to_float(low_v)
        volume_f = to_float(volume_v)
        row = {
            "stock_code": code,
            "date": date_s,
            "open": to_float(open_v),
            "close": close_f,
            "high": high_f,
            "low": low_f,
            "volume": volume_f,
            "amount": close_f * volume_f * 100 if close_f is not None and volume_f is not None else None,
            "amplitude": None,
            "pct_chg": None,
            "price_change": None,
            "turnover_rate": None,
        }
        if prev_close not in (None, 0) and close_f is not None:
            row["price_change"] = close_f - prev_close
            row["pct_chg"] = row["price_change"] / prev_close * 100
            if high_f is not None and low_f is not None:
                row["amplitude"] = (high_f - low_f) / prev_close * 100
        if close_f is not None:
            prev_close = close_f
        rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for col in HISTORY_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").drop_duplicates(["stock_code", "date"], keep="last")


def stock_history_features(history: pd.DataFrame) -> dict[str, Any]:
    """Calculate recent trading/momentum features from daily stock history."""

    if history.empty:
        return {}
    df = history.sort_values("date").copy()
    latest = df.iloc[-1]
    closes = pd.to_numeric(df["close"], errors="coerce")
    amounts = pd.to_numeric(df["amount"], errors="coerce")

    def ret(days: int) -> float | None:
        if len(closes) <= days:
            return None
        prev = to_float(closes.iloc[-days - 1])
        last = to_float(closes.iloc[-1])
        if prev in (None, 0.0) or last is None:
            return None
        return last / prev - 1.0

    amount_last = to_float(amounts.iloc[-1])
    amount20 = amounts.tail(20).dropna()
    amount_mean_20 = to_float(amount20.mean()) if not amount20.empty else None
    amount_std_20 = to_float(amount20.std(ddof=0)) if len(amount20) > 1 else None
    amount_ratio_20 = None
    if amount_last is not None and amount_mean_20 not in (None, 0.0):
        amount_ratio_20 = amount_last / amount_mean_20
    amount_z20 = None
    if amount_last is not None and amount_mean_20 is not None and amount_std_20 not in (None, 0.0):
        amount_z20 = (amount_last - amount_mean_20) / amount_std_20

    return {
        "history_trade_date": _date_only(latest.get("date")),
        "history_close": to_float(latest.get("close")),
        "history_amount": amount_last,
        "history_amount_mean_20": amount_mean_20,
        "history_amount_ratio_20": amount_ratio_20,
        "history_amount_z20": amount_z20,
        "history_turnover_rate": to_float(latest.get("turnover_rate")),
        "history_amplitude": to_float(latest.get("amplitude")),
        "history_pct_chg": to_float(latest.get("pct_chg")),
        "stock_ret_3": ret(3),
        "stock_ret_5": ret(5),
        "stock_ret_10": ret(10),
        "stock_ret_20": ret(20),
        "stock_ret_60": ret(60),
    }


def merge_quote_and_history(
    candidates: pd.DataFrame,
    *,
    history_days: int,
    end: str | None,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add spot quote and recent K-line features to stock candidates."""

    if candidates.empty:
        return candidates.copy(), {}
    codes = candidates["stock_code"].astype(str).str.zfill(6).tolist()
    spot = fetch_stock_spot(codes, force=force)
    spot = spot.drop_duplicates("stock_code", keep="first")
    merged = candidates.merge(spot, on="stock_code", how="left")
    if "stock_name_quote" in merged.columns:
        merged["stock_name"] = merged["stock_name"].fillna(merged["stock_name_quote"])
        merged.loc[merged["stock_name"].astype(str).str.len() == 0, "stock_name"] = merged.loc[
            merged["stock_name"].astype(str).str.len() == 0, "stock_name_quote"
        ]
    start = _history_start_for_end(end, history_days)
    end_text = end or today_yyyymmdd()
    errors: dict[str, str] = {}
    feature_rows: list[dict[str, Any]] = []
    for code in codes:
        try:
            hist = fetch_stock_history(code, start=start, end=end_text, force=force)
            features = stock_history_features(hist)
            features["stock_code"] = code
            feature_rows.append(features)
        except Exception as exc:  # noqa: BLE001 - keep other symbols progressing
            errors[code] = str(exc)
    if feature_rows:
        features_df = pd.DataFrame(feature_rows)
        merged = merged.merge(features_df, on="stock_code", how="left")
    return merged, errors


# ---------------------------------------------------------------------------
# Shareholder enrichment


def datacenter_get(params: dict[str, Any], *, timeout: int = 20) -> dict[str, Any]:
    session = make_session()
    session.headers.update({"Referer": "https://data.eastmoney.com/gdfx/HoldingDetail.html"})
    response = session.get(EASTMONEY_DATACENTER_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_report_date(report_name: str, *, force: bool = False) -> str:
    """Fetch latest shareholder report date for an Eastmoney holder report."""

    cache = RAW_DIR / f"{report_name.lower()}_{today_yyyymmdd()}.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache)
        if not df.empty and "END_DATE" in df.columns:
            return _date_only(df.iloc[0]["END_DATE"])
    payload = datacenter_get(
        {
            "reportName": report_name,
            "columns": "ALL",
            "source": "WEB",
            "client": "WEB",
            "pageNumber": 1,
            "pageSize": 20,
            "sortColumns": "END_DATE",
            "sortTypes": "-1",
        }
    )
    rows = ((payload.get("result") or {}).get("data")) or []
    if not rows:
        raise RuntimeError(f"No report dates for {report_name}")
    pd.DataFrame(rows).to_csv(cache, index=False)
    return _date_only(rows[0].get("END_DATE"))


def fetch_holder_rows(
    code: str,
    *,
    holder_kind: str,
    report_date: str | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Fetch latest top-holder rows for one stock.

    ``holder_kind`` is ``free`` for top circulating shareholders or ``total``
    for top total shareholders.
    """

    code = _normalize_code(code)
    if holder_kind not in {"free", "total"}:
        raise ValueError("holder_kind must be 'free' or 'total'")
    report = "RPT_F10_EH_FREEHOLDERS" if holder_kind == "free" else "RPT_DMSK_HOLDERS"
    rank_col = "HOLDER_RANK" if holder_kind == "free" else "RANK"
    date_key = (report_date or "latest").replace("-", "")
    cache = RAW_DIR / f"shareholders_{holder_kind}_{code}_{date_key}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache).to_dict("records")
    filters = [f'(SECURITY_CODE="{code}")']
    if report_date:
        filters.append(f'(END_DATE="{report_date}")')
    # Exclude terminated/listing-state rows, matching the public page filter.
    if holder_kind == "free":
        filters.append('(LISTING_STATE<>"10")')
    else:
        filters.extend(['(LISTING_STATE<>"10")', '(LISTING_STATE<>"9")'])
    payload = datacenter_get(
        {
            "reportName": report,
            "columns": "ALL",
            "source": "WEB",
            "client": "WEB",
            "pageNumber": 1,
            "pageSize": 50,
            "sortColumns": f"END_DATE,{rank_col}",
            "sortTypes": "-1,1",
            "filter": "".join(filters),
        }
    )
    rows = ((payload.get("result") or {}).get("data")) or []
    if not rows and report_date:
        # Some symbols may not have data for the global latest quarter.  Retry
        # without date and then aggregate the newest date returned for the stock.
        return fetch_holder_rows(code, holder_kind=holder_kind, report_date=None, force=force)
    if rows:
        pd.DataFrame(rows).to_csv(cache, index=False)
    return rows


def fetch_holder_history_rows(
    code: str,
    *,
    holder_kind: str,
    periods: int = 8,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Fetch multiple report periods of top-holder rows for one stock."""

    code = _normalize_code(code)
    if holder_kind not in {"free", "total"}:
        raise ValueError("holder_kind must be 'free' or 'total'")
    periods = max(1, int(periods))
    report = "RPT_F10_EH_FREEHOLDERS" if holder_kind == "free" else "RPT_DMSK_HOLDERS"
    rank_col = "HOLDER_RANK" if holder_kind == "free" else "RANK"
    cache = RAW_DIR / f"shareholder_history_{holder_kind}_{code}_{periods}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache).to_dict("records")
    filters = [f'(SECURITY_CODE="{code}")']
    if holder_kind == "free":
        filters.append('(LISTING_STATE<>"10")')
    else:
        filters.extend(['(LISTING_STATE<>"10")', '(LISTING_STATE<>"9")'])
    payload = datacenter_get(
        {
            "reportName": report,
            "columns": "ALL",
            "source": "WEB",
            "client": "WEB",
            "pageNumber": 1,
            "pageSize": max(120, periods * 35),
            "sortColumns": f"END_DATE,{rank_col}",
            "sortTypes": "-1,1",
            "filter": "".join(filters),
        }
    )
    rows = ((payload.get("result") or {}).get("data")) or []
    if rows:
        # Keep only the newest N unique report periods and their top-10 rows.
        dates = sorted({_date_only(row.get("END_DATE")) for row in rows if _date_only(row.get("END_DATE"))}, reverse=True)
        keep_dates = set(dates[:periods])
        rows = [row for row in rows if _date_only(row.get("END_DATE")) in keep_dates]
        pd.DataFrame(rows).to_csv(cache, index=False)
    return rows


def _holder_type(row: dict[str, Any], kind: str) -> str:
    return str(
        row.get("HOLDER_NEWTYPE")
        or row.get("HOLDER_TYPE")
        or row.get("HOLDER_NATURE")
        or row.get("HOLDER_TYPE_ORG")
        or ""
    )


def is_institution_holder(row: dict[str, Any], kind: str) -> bool:
    holder_type = _holder_type(row, kind)
    holder_name = str(row.get("HOLDER_NAME") or row.get("HOLDER_NEW") or "")
    if "个人" in holder_type or "自然人" in holder_type:
        return False
    if holder_type.strip():
        return True
    institutional_terms = ("基金", "银行", "保险", "证券", "信托", "社保", "QFII", "香港中央结算", "资管", "资产管理", "投资")
    return any(term in holder_name for term in institutional_terms)


def is_fund_like_holder(row: dict[str, Any]) -> bool:
    text = " ".join(str(row.get(key) or "") for key in ("HOLDER_NAME", "HOLDER_NEW", "HOLDER_NEWTYPE", "HOLDER_TYPE"))
    return any(term in text for term in ("基金", "ETF", "联接", "社保", "QFII"))


def is_individual_holder(row: dict[str, Any], kind: str) -> bool:
    holder_type = _holder_type(row, kind)
    holder_name = str(row.get("HOLDER_NAME") or row.get("HOLDER_NEW") or "")
    if "个人" in holder_type or "自然人" in holder_type:
        return True
    # Eastmoney occasionally leaves the type empty for natural-person names.  Do
    # not infer aggressively for organisations; only use a conservative
    # not-institution fallback for short Chinese names.
    if holder_type.strip():
        return False
    institutional_terms = ("公司", "集团", "基金", "银行", "保险", "证券", "信托", "社保", "QFII", "合伙", "资管", "资产")
    return bool(holder_name) and len(holder_name) <= 4 and not any(term in holder_name for term in institutional_terms)


def _limit_latest_holder_rows(rows: list[dict[str, Any]], rank_col: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    dates = [_date_only(row.get("END_DATE")) for row in rows if _date_only(row.get("END_DATE"))]
    latest = max(dates) if dates else ""
    filtered = [row for row in rows if not latest or _date_only(row.get("END_DATE")) == latest]
    return sorted(filtered, key=lambda row: to_float(row.get(rank_col), 9999.0) or 9999.0)[:10]


def aggregate_holder_rows(
    free_rows: list[dict[str, Any]],
    total_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate top-holder concentration ratios for one stock."""

    free_top = _limit_latest_holder_rows(free_rows, "HOLDER_RANK")
    total_top = _limit_latest_holder_rows(total_rows, "RANK")

    def sum_field(rows: list[dict[str, Any]], field: str) -> float | None:
        vals = [to_float(row.get(field)) for row in rows]
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None

    free_ratio = sum_field(free_top, "FREE_HOLDNUM_RATIO")
    free_total_ratio = sum_field(free_top, "HOLD_RATIO")
    total_ratio = sum_field(total_top, "HOLD_RATIO")
    free_inst_ratio = sum(
        to_float(row.get("FREE_HOLDNUM_RATIO"), 0.0) or 0.0 for row in free_top if is_institution_holder(row, "free")
    )
    total_inst_ratio = sum(to_float(row.get("HOLD_RATIO"), 0.0) or 0.0 for row in total_top if is_institution_holder(row, "total"))
    fund_like_ratio = sum(
        to_float(row.get("FREE_HOLDNUM_RATIO"), 0.0) or 0.0 for row in free_top if is_fund_like_holder(row)
    )
    free_change_sum = sum(to_float(row.get("HOLD_RATIO_CHANGE"), 0.0) or 0.0 for row in free_top)
    total_change_sum = sum(to_float(row.get("HOLD_RATIO_CHANGE"), 0.0) or 0.0 for row in total_top)

    top_free = free_top[0] if free_top else {}
    top_total = total_top[0] if total_top else {}
    return {
        "holder_report_date_free": _date_only(top_free.get("END_DATE")) if top_free else "",
        "holder_report_date_total": _date_only(top_total.get("END_DATE")) if top_total else "",
        "top_free_holder_ratio_sum_pct": free_ratio,
        "top_free_holder_total_ratio_sum_pct": free_total_ratio,
        "top_free_institution_ratio_sum_pct": free_inst_ratio if free_top else None,
        "top_free_fund_like_ratio_sum_pct": fund_like_ratio if free_top else None,
        "top_free_hold_ratio_change_sum_pct": free_change_sum if free_top else None,
        "top_free_holder_name": top_free.get("HOLDER_NAME") or top_free.get("HOLDER_NEW") or "",
        "top_free_holder_type": _holder_type(top_free, "free") if top_free else "",
        "top_free_holder_ratio_pct": to_float(top_free.get("FREE_HOLDNUM_RATIO")) if top_free else None,
        "top_total_holder_ratio_sum_pct": total_ratio,
        "top_total_institution_ratio_sum_pct": total_inst_ratio if total_top else None,
        "top_total_hold_ratio_change_sum_pct": total_change_sum if total_top else None,
        "top_total_holder_name": top_total.get("HOLDER_NAME") or top_total.get("HOLDER_NEW") or "",
        "top_total_holder_type": _holder_type(top_total, "total") if total_top else "",
        "top_total_holder_ratio_pct": to_float(top_total.get("HOLD_RATIO")) if top_total else None,
        "top_holder_rows_free": len(free_top),
        "top_holder_rows_total": len(total_top),
    }


def holder_detail_records(code: str, stock_name: str, holder_kind: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_col = "HOLDER_RANK" if holder_kind == "free" else "RANK"
    top_rows = _limit_latest_holder_rows(rows, rank_col)
    details: list[dict[str, Any]] = []
    for row in top_rows:
        details.append(
            {
                "stock_code": code,
                "stock_name": stock_name,
                "holder_kind": holder_kind,
                "report_date": _date_only(row.get("END_DATE")),
                "rank": to_float(row.get(rank_col)),
                "holder_name": row.get("HOLDER_NAME") or row.get("HOLDER_NEW") or "",
                "holder_type": _holder_type(row, holder_kind),
                "shares_type": row.get("SHARES_TYPE") or "",
                "hold_num": to_float(row.get("HOLD_NUM")),
                "hold_ratio_pct": to_float(row.get("HOLD_RATIO")),
                "free_hold_ratio_pct": to_float(row.get("FREE_HOLDNUM_RATIO")),
                "hold_num_change": row.get("HOLD_NUM_CHANGE") or row.get("HOLD_CHANGE") or "",
                "hold_ratio_change_pct": to_float(row.get("HOLD_RATIO_CHANGE")),
                "direction": row.get("DIRECTION") or row.get("HOLDNUM_CHANGE_NAME") or row.get("HOLD_CHANGE") or "",
                "holder_market_cap": to_float(row.get("HOLDER_MARKET_CAP") or row.get("REFERENCE_MARKET_CAP")),
            }
        )
    return details


def holder_history_records(
    code: str,
    stock_name: str,
    holder_kind: str,
    rows: list[dict[str, Any]],
    *,
    periods: int,
) -> list[dict[str, Any]]:
    """Normalize historical top-holder rows across multiple report periods."""

    rank_col = "HOLDER_RANK" if holder_kind == "free" else "RANK"
    records: list[dict[str, Any]] = []
    dates = sorted({_date_only(row.get("END_DATE")) for row in rows if _date_only(row.get("END_DATE"))}, reverse=True)[: max(1, periods)]
    for period_index, report_date in enumerate(dates, 1):
        period_rows = [row for row in rows if _date_only(row.get("END_DATE")) == report_date]
        period_rows = sorted(period_rows, key=lambda row: to_float(row.get(rank_col), 9999.0) or 9999.0)[:10]
        for row in period_rows:
            records.append(
                {
                    "stock_code": code,
                    "stock_name": stock_name,
                    "holder_kind": holder_kind,
                    "report_date": report_date,
                    "period_index_desc": period_index,
                    "rank": to_float(row.get(rank_col)),
                    "holder_name": row.get("HOLDER_NAME") or row.get("HOLDER_NEW") or "",
                    "holder_type": _holder_type(row, holder_kind),
                    "shares_type": row.get("SHARES_TYPE") or "",
                    "is_individual": is_individual_holder(row, holder_kind),
                    "is_institution": is_institution_holder(row, holder_kind),
                    "is_fund_like": is_fund_like_holder(row),
                    "hold_num": to_float(row.get("HOLD_NUM")),
                    "hold_ratio_pct": to_float(row.get("HOLD_RATIO")),
                    "free_hold_ratio_pct": to_float(row.get("FREE_HOLDNUM_RATIO")),
                    "hold_num_change": row.get("HOLD_NUM_CHANGE") or row.get("HOLD_CHANGE") or "",
                    "hold_ratio_change_pct": to_float(row.get("HOLD_RATIO_CHANGE")),
                    "direction": row.get("DIRECTION") or row.get("HOLDNUM_CHANGE_NAME") or row.get("HOLD_CHANGE") or "",
                    "holder_market_cap": to_float(row.get("HOLDER_MARKET_CAP") or row.get("REFERENCE_MARKET_CAP")),
                }
            )
    return records


def _history_ratio_value(row: pd.Series | dict[str, Any], holder_kind: str) -> float | None:
    get = row.get if isinstance(row, dict) else row.get
    if holder_kind == "free":
        ratio = to_float(get("free_hold_ratio_pct"))
        if ratio is not None:
            return ratio
    return to_float(get("hold_ratio_pct"))


def summarize_holder_history(history: pd.DataFrame) -> pd.DataFrame:
    """Summarize top-holder concentration and individual-holder exposure by period."""

    if history.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (code, stock_name, holder_kind, report_date), group in history.groupby(
        ["stock_code", "stock_name", "holder_kind", "report_date"], sort=False
    ):
        ratios = [_history_ratio_value(row, str(holder_kind)) for _, row in group.iterrows()]
        ratio_sum = sum(v for v in ratios if v is not None)
        individual = group[group.get("is_individual", False).astype(bool)]
        institution = group[group.get("is_institution", False).astype(bool)]
        fund_like = group[group.get("is_fund_like", False).astype(bool)]
        rows.append(
            {
                "stock_code": _normalize_code(code),
                "stock_name": stock_name,
                "holder_kind": holder_kind,
                "report_date": report_date,
                "top10_ratio_sum_pct": ratio_sum,
                "individual_holder_count": int(len(individual)),
                "individual_holder_ratio_sum_pct": sum(
                    _history_ratio_value(row, str(holder_kind)) or 0.0 for _, row in individual.iterrows()
                ),
                "individual_holder_names": _safe_join(individual["holder_name"].tolist(), "、") if not individual.empty else "",
                "institution_holder_ratio_sum_pct": sum(
                    _history_ratio_value(row, str(holder_kind)) or 0.0 for _, row in institution.iterrows()
                ),
                "fund_like_holder_ratio_sum_pct": sum(
                    _history_ratio_value(row, str(holder_kind)) or 0.0 for _, row in fund_like.iterrows()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["stock_code", "holder_kind", "report_date"], ascending=[True, True, False])


def build_individual_holder_changes(history: pd.DataFrame) -> pd.DataFrame:
    """Build period-to-period changes for personal shareholders."""

    if history.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    history = history.copy()
    history["stock_code"] = history["stock_code"].astype(str).str.zfill(6)
    for (code, stock_name, holder_kind), group in history.groupby(["stock_code", "stock_name", "holder_kind"], sort=False):
        dates = sorted({_date_only(value) for value in group["report_date"].tolist() if _date_only(value)})
        for prev_date, curr_date in zip(dates, dates[1:]):
            prev = group[group["report_date"].eq(prev_date)]
            curr = group[group["report_date"].eq(curr_date)]
            prev_ind = {str(row["holder_name"]): row for _, row in prev[prev.get("is_individual", False).astype(bool)].iterrows()}
            curr_ind = {str(row["holder_name"]): row for _, row in curr[curr.get("is_individual", False).astype(bool)].iterrows()}
            for holder_name in sorted(set(prev_ind) | set(curr_ind)):
                prev_row = prev_ind.get(holder_name)
                curr_row = curr_ind.get(holder_name)
                prev_ratio = _history_ratio_value(prev_row, str(holder_kind)) if prev_row is not None else None
                curr_ratio = _history_ratio_value(curr_row, str(holder_kind)) if curr_row is not None else None
                prev_hold = to_float(prev_row.get("hold_num")) if prev_row is not None else None
                curr_hold = to_float(curr_row.get("hold_num")) if curr_row is not None else None
                delta = (curr_ratio or 0.0) - (prev_ratio or 0.0)
                if prev_row is None:
                    status = "新进"
                elif curr_row is None:
                    status = "退出"
                elif delta > 1e-6:
                    status = "增持"
                elif delta < -1e-6:
                    status = "减持"
                else:
                    status = "不变"
                rows.append(
                    {
                        "stock_code": code,
                        "stock_name": stock_name,
                        "holder_kind": holder_kind,
                        "from_report_date": prev_date,
                        "to_report_date": curr_date,
                        "holder_name": holder_name,
                        "holder_type": (
                            (curr_row.get("holder_type") if curr_row is not None else None)
                            or (prev_row.get("holder_type") if prev_row is not None else "")
                        ),
                        "status": status,
                        "prev_rank": to_float(prev_row.get("rank")) if prev_row is not None else None,
                        "current_rank": to_float(curr_row.get("rank")) if curr_row is not None else None,
                        "prev_ratio_pct": prev_ratio,
                        "current_ratio_pct": curr_ratio,
                        "ratio_delta_pct": delta,
                        "prev_hold_num": prev_hold,
                        "current_hold_num": curr_hold,
                        "hold_num_delta": (curr_hold or 0.0) - (prev_hold or 0.0),
                    }
                )
    if not rows:
        return pd.DataFrame()
    changes = pd.DataFrame(rows)
    changes["abs_ratio_delta_pct"] = changes["ratio_delta_pct"].abs()
    return changes.sort_values(
        ["stock_code", "holder_kind", "to_report_date", "abs_ratio_delta_pct"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def enrich_shareholder_history(
    rows: pd.DataFrame,
    *,
    periods: int,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Fetch and calculate historical shareholder changes for output stocks."""

    if rows.empty or periods <= 1:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    errors: dict[str, str] = {}
    history_records: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        code = _normalize_code(row.get("stock_code"))
        stock_name = str(row.get("stock_name") or "")
        for holder_kind in ("free", "total"):
            try:
                raw_rows = fetch_holder_history_rows(code, holder_kind=holder_kind, periods=periods, force=force)
                history_records.extend(holder_history_records(code, stock_name, holder_kind, raw_rows, periods=periods))
            except Exception as exc:  # noqa: BLE001
                errors[f"{code}:{holder_kind}"] = str(exc)
    history = pd.DataFrame(history_records)
    summary = summarize_holder_history(history)
    changes = build_individual_holder_changes(history)
    return history, summary, changes, errors


def enrich_shareholders(
    rows: pd.DataFrame,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Add top-shareholder concentration summaries to bullish stock rows."""

    if rows.empty:
        return rows.copy(), pd.DataFrame(), {}
    errors: dict[str, str] = {}
    try:
        latest_free_date = fetch_report_date("RPT_FREEHOLDERS_HD_REPORTDATE", force=force)
    except Exception as exc:  # noqa: BLE001
        latest_free_date = None
        errors["RPT_FREEHOLDERS_HD_REPORTDATE"] = str(exc)
    try:
        latest_total_date = fetch_report_date("RPT_HOLDERS_HD_REPORTDATE", force=force)
    except Exception as exc:  # noqa: BLE001
        latest_total_date = None
        errors["RPT_HOLDERS_HD_REPORTDATE"] = str(exc)

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        code = _normalize_code(row.get("stock_code"))
        stock_name = str(row.get("stock_name") or "")
        try:
            time.sleep(0.05)
            free_rows = fetch_holder_rows(code, holder_kind="free", report_date=latest_free_date, force=force)
            total_rows = fetch_holder_rows(code, holder_kind="total", report_date=latest_total_date, force=force)
            summary = aggregate_holder_rows(free_rows, total_rows)
            summary["stock_code"] = code
            summaries.append(summary)
            details.extend(holder_detail_records(code, stock_name, "free", free_rows))
            details.extend(holder_detail_records(code, stock_name, "total", total_rows))
        except Exception as exc:  # noqa: BLE001
            errors[code] = str(exc)
    enriched = rows.merge(pd.DataFrame(summaries), on="stock_code", how="left") if summaries else rows.copy()
    details_df = pd.DataFrame(details)
    return enriched, details_df, errors


# ---------------------------------------------------------------------------
# Scoring, mechanism tags, and reporting


def compute_bullish_score(row: pd.Series | dict[str, Any], *, include_shareholders: bool = True) -> float:
    """Explainable score for ranking stock candidates beneath ETF signals."""

    get = row.get if isinstance(row, dict) else row.get
    score = 0.0
    score += max(to_float(get("source_etf_pred_fwd_ret_5_best"), 0.0) or 0.0, 0.0) * 120.0
    rank = to_float(get("source_etf_pred_rank_best"), 9999.0) or 9999.0
    score += max(0.0, 10.0 - min(rank, 10.0)) * 0.35
    for col, weight in (("stock_ret_5", 38.0), ("stock_ret_20", 24.0), ("stock_ret_60", 12.0)):
        score += max(to_float(get(col), 0.0) or 0.0, 0.0) * weight
    quote_pct = to_float(get("quote_pct_chg"))
    if quote_pct is not None:
        score += max(quote_pct / 100.0, 0.0) * 16.0
    amount_ratio = to_float(get("history_amount_ratio_20"), 1.0) or 1.0
    score += max(0.0, min(amount_ratio - 1.0, 3.0)) * 1.25
    amount_z20 = to_float(get("history_amount_z20"), 0.0) or 0.0
    score += max(0.0, min(amount_z20, 5.0)) * 0.4
    turnover = to_float(get("quote_turnover_rate"))
    if turnover is None:
        turnover = to_float(get("history_turnover_rate"))
    if turnover is not None:
        score += min(max(turnover, 0.0), 20.0) * 0.08
    main_flow_pct = to_float(get("main_net_inflow_pct"))
    if main_flow_pct is not None:
        score += max(main_flow_pct / 100.0, 0.0) * 8.0
    index_weight = to_float(get("index_weight_pct_max"))
    if index_weight is not None:
        score += min(max(index_weight, 0.0), 10.0) * 0.05
    if include_shareholders:
        free_ratio = to_float(get("top_free_holder_ratio_sum_pct"))
        inst_ratio = to_float(get("top_free_institution_ratio_sum_pct"))
        fund_ratio = to_float(get("top_free_fund_like_ratio_sum_pct"))
        if free_ratio is not None:
            score += min(max(free_ratio, 0.0), 80.0) * 0.015
        if inst_ratio is not None:
            score += min(max(inst_ratio, 0.0), 80.0) * 0.01
        if fund_ratio is not None:
            score += min(max(fund_ratio, 0.0), 40.0) * 0.012
    return round(score, 6)


def mechanism_tags(row: pd.Series | dict[str, Any]) -> str:
    get = row.get if isinstance(row, dict) else row.get
    tags: list[str] = []
    if (to_float(get("source_etf_pred_fwd_ret_5_best"), 0.0) or 0.0) >= 0:
        tags.append("ETF信号牵引")
    if (to_float(get("stock_ret_5"), 0.0) or 0.0) > 0 and (to_float(get("stock_ret_20"), 0.0) or 0.0) > 0:
        tags.append("价格动量")
    if (to_float(get("history_amount_ratio_20"), 0.0) or 0.0) >= 1.5 or (to_float(get("history_amount_z20"), 0.0) or 0.0) >= 1.0:
        tags.append("成交额放大")
    turnover = to_float(get("quote_turnover_rate"))
    if turnover is None:
        turnover = to_float(get("history_turnover_rate"))
    if turnover is not None and turnover >= 5.0:
        tags.append("高换手交易")
    if (to_float(get("main_net_inflow_pct"), 0.0) or 0.0) > 0:
        tags.append("主力净流入")
    if (to_float(get("top_free_holder_ratio_sum_pct"), 0.0) or 0.0) >= 30:
        tags.append("流通股东集中")
    if (to_float(get("top_free_institution_ratio_sum_pct"), 0.0) or 0.0) >= 10:
        tags.append("机构股东参与")
    if (to_float(get("component_roe_pct"), 0.0) or 0.0) >= 10:
        tags.append("ROE支撑")
    return ";".join(tags) if tags else "待观察"


def rank_bullish_rows(rows: pd.DataFrame, *, top_stocks: int, include_shareholders: bool) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    ranked = rows.copy()
    for col in ("quote_amount", "history_amount", "source_etf_pred_fwd_ret_5_best"):
        if col not in ranked.columns:
            ranked[col] = math.nan
    ranked["bullish_score"] = ranked.apply(lambda row: compute_bullish_score(row, include_shareholders=include_shareholders), axis=1)
    ranked["mechanism_tags"] = ranked.apply(mechanism_tags, axis=1)
    ranked = ranked.sort_values(
        ["bullish_score", "quote_amount", "history_amount", "source_etf_pred_fwd_ret_5_best"],
        ascending=[False, False, False, False],
    )
    if top_stocks > 0:
        ranked = ranked.head(int(top_stocks)).copy()
    if "bullish_rank" in ranked.columns:
        ranked = ranked.drop(columns=["bullish_rank"])
    ranked.insert(0, "bullish_rank", range(1, len(ranked) + 1))
    return ranked.reset_index(drop=True)


def _ordered_output_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "bullish_rank",
        "bullish_score",
        "stock_code",
        "stock_name",
        "family_ids",
        "source_etf_codes",
        "source_etf_names",
        "source_etf_pred_fwd_ret_5_best",
        "source_etf_pred_rank_best",
        "mechanism_tags",
        "industry",
        "region",
        "latest_price",
        "quote_pct_chg",
        "quote_amount",
        "quote_turnover_rate",
        "main_net_inflow",
        "main_net_inflow_pct",
        "history_trade_date",
        "history_amount",
        "history_amount_ratio_20",
        "history_amount_z20",
        "stock_ret_5",
        "stock_ret_20",
        "stock_ret_60",
        "index_weight_pct_max",
        "component_rank_best",
        "component_free_cap_max",
        "component_roe_pct",
        "pe_ttm",
        "pb",
        "top_free_holder_ratio_sum_pct",
        "top_free_institution_ratio_sum_pct",
        "top_free_fund_like_ratio_sum_pct",
        "top_free_hold_ratio_change_sum_pct",
        "top_free_holder_name",
        "top_free_holder_type",
        "top_free_holder_ratio_pct",
        "top_total_holder_ratio_sum_pct",
        "top_total_institution_ratio_sum_pct",
        "top_total_hold_ratio_change_sum_pct",
        "top_total_holder_name",
        "top_total_holder_type",
        "top_total_holder_ratio_pct",
        "holder_report_date_free",
        "holder_report_date_total",
        "quote_time",
        "fetched_at",
    ]
    return [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]


def write_deep_dive_charts(rows: pd.DataFrame, charts_dir: Path) -> dict[str, str]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: dict[str, str] = {}
    if rows.empty:
        return chart_paths
    top_score = [
        (f"{row.stock_code} {row.stock_name}", to_float(row.bullish_score, 0.0) or 0.0)
        for row in rows.head(15).itertuples(index=False)
    ]
    path = write_horizontal_bar_chart(
        charts_dir / "bullish_stock_score_top.svg",
        title="看涨股票综合评分 Top",
        subtitle="综合 ETF 信号、价格动量、成交额放大、换手与股东结构；仅作研究排序。",
        rows=top_score,
        value_kind="number",
    )
    chart_paths["score_top"] = str(path)

    amount_rows = []
    for row in rows.sort_values("quote_amount", ascending=False).head(15).itertuples(index=False):
        amount = to_float(getattr(row, "quote_amount", None))
        if amount is not None:
            amount_rows.append((f"{row.stock_code} {row.stock_name}", amount / 1e8))
    if amount_rows:
        path = write_horizontal_bar_chart(
            charts_dir / "bullish_stock_amount_top.svg",
            title="看涨股票成交额 Top",
            subtitle="单位：亿元；用于确认上涨背后的交易活跃度。",
            rows=amount_rows,
            value_kind="number",
            positive_color="#0891b2",
        )
        chart_paths["amount_top"] = str(path)

    holder_rows = []
    if "top_free_holder_ratio_sum_pct" in rows.columns:
        for row in rows.sort_values("top_free_holder_ratio_sum_pct", ascending=False).head(15).itertuples(index=False):
            ratio = to_float(getattr(row, "top_free_holder_ratio_sum_pct", None))
            if ratio is not None:
                holder_rows.append((f"{row.stock_code} {row.stock_name}", ratio / 100.0))
    if holder_rows:
        path = write_horizontal_bar_chart(
            charts_dir / "bullish_stock_holder_concentration.svg",
            title="看涨股票前十大流通股东占比",
            subtitle="前十大流通股东合计持股占流通股比例。",
            rows=holder_rows,
            value_kind="pct",
            positive_color="#7c3aed",
        )
        chart_paths["holder_concentration"] = str(path)

    industry_counts = rows["industry"].fillna("未知").replace("", "未知").value_counts().head(12)
    if not industry_counts.empty:
        path = write_column_chart(
            charts_dir / "bullish_stock_industry_counts.svg",
            title="看涨股票行业分布",
            subtitle="按自动深挖后的看涨候选股票数量统计。",
            rows=[(idx, float(val)) for idx, val in industry_counts.items()],
            value_kind="number",
        )
        chart_paths["industry_counts"] = str(path)
    return chart_paths


def _holder_ratio_for_chart(detail: pd.Series) -> float | None:
    ratio = to_float(detail.get("free_hold_ratio_pct"))
    if ratio is None:
        ratio = to_float(detail.get("hold_ratio_pct"))
    if ratio is None:
        return None
    return ratio / 100.0


def _holder_detail_table(details: pd.DataFrame, holder_kind: str) -> list[str]:
    sub = details[details["holder_kind"].eq(holder_kind)].copy() if not details.empty else pd.DataFrame()
    title = "十大流通股东" if holder_kind == "free" else "十大股东"
    lines = [f"### {title}", ""]
    if sub.empty:
        lines.append("- 暂无明细。")
        return lines
    sub["rank_sort"] = pd.to_numeric(sub.get("rank"), errors="coerce").fillna(9999)
    sub = sub.sort_values("rank_sort").head(10)
    lines.append("|排名|股东名称|类型|股份类型|持股数|占总股本|占流通股本|占比变化|方向|参考市值|")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---|---:|")
    for _, row in sub.iterrows():
        lines.append(
            f"|{_fmt_number(row.get('rank'), 0)}|{row.get('holder_name', '')}|{row.get('holder_type', '')}|"
            f"{row.get('shares_type', '')}|{_fmt_number(row.get('hold_num'), 0)}|"
            f"{_fmt_pct_value(row.get('hold_ratio_pct'))}|{_fmt_pct_value(row.get('free_hold_ratio_pct'))}|"
            f"{_fmt_pct_value(row.get('hold_ratio_change_pct'))}|{row.get('direction', '')}|"
            f"{human_amount(to_float(row.get('holder_market_cap')))}|"
        )
    return lines


def write_company_shareholder_reports(
    *,
    rows: pd.DataFrame,
    details: pd.DataFrame,
    history_summary: pd.DataFrame | None = None,
    individual_changes: pd.DataFrame | None = None,
    out_dir: Path,
    charts_dir: Path,
) -> list[dict[str, str]]:
    """Write one shareholder-composition report per output stock.

    The deep-dive CSV is useful for screening, while these per-company reports
    make the shareholder composition directly reviewable for each bullish stock.
    They can also be attached to later research notes or PR artifacts.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    if rows.empty:
        return []
    if details.empty:
        details = pd.DataFrame(columns=["stock_code", "holder_kind"])
    if history_summary is None or history_summary.empty:
        history_summary = pd.DataFrame(columns=["stock_code", "holder_kind", "report_date"])
    if individual_changes is None or individual_changes.empty:
        individual_changes = pd.DataFrame(columns=["stock_code", "holder_kind", "to_report_date"])
    reports: list[dict[str, str]] = []
    for _, stock in rows.iterrows():
        code = _normalize_code(stock.get("stock_code"))
        name = str(stock.get("stock_name") or "")
        rank = int(to_float(stock.get("bullish_rank"), 0.0) or 0)
        stem = f"{rank:02d}_{code}_{_safe_filename(name)}"
        md_path = out_dir / f"{stem}_shareholder_composition.md"
        stock_details = details[details["stock_code"].astype(str).str.zfill(6).eq(code)].copy()
        stock_history_summary = history_summary[history_summary["stock_code"].astype(str).str.zfill(6).eq(code)].copy()
        stock_individual_changes = individual_changes[individual_changes["stock_code"].astype(str).str.zfill(6).eq(code)].copy()

        chart_entries: list[tuple[str, Path]] = []
        for holder_kind, label, ratio_col in (
            ("free", "十大流通股东占比", "free_hold_ratio_pct"),
            ("total", "十大股东占比", "hold_ratio_pct"),
        ):
            sub = stock_details[stock_details["holder_kind"].eq(holder_kind)].copy()
            if sub.empty:
                continue
            sub["rank_sort"] = pd.to_numeric(sub.get("rank"), errors="coerce").fillna(9999)
            sub = sub.sort_values("rank_sort").head(10)
            chart_rows: list[tuple[str, float]] = []
            for _, item in sub.iterrows():
                ratio = _holder_ratio_for_chart(item)
                if ratio is None:
                    continue
                chart_rows.append((f"{_fmt_number(item.get('rank'), 0)} {item.get('holder_name', '')}", ratio))
            if chart_rows:
                chart_path = charts_dir / f"{stem}_{holder_kind}_holders_pie.svg"
                write_pie_chart(
                    chart_path,
                    title=f"{code} {name} {label}",
                    subtitle="东方财富股东分析口径；饼图含前十大股东及其他/未披露占比。",
                    rows=chart_rows,
                    add_remainder=True,
                )
                chart_entries.append((label, chart_path))
        if not stock_history_summary.empty:
            series: dict[str, list[tuple[str, float]]] = {}
            for holder_kind, label in (("free", "流通个人股东占比"), ("total", "总股本个人股东占比")):
                sub = stock_history_summary[stock_history_summary["holder_kind"].eq(holder_kind)].copy()
                if sub.empty:
                    continue
                sub = sub.sort_values("report_date")
                points = [
                    (str(row.get("report_date")), (to_float(row.get("individual_holder_ratio_sum_pct"), 0.0) or 0.0) / 100.0)
                    for _, row in sub.iterrows()
                ]
                if points:
                    series[label] = points
            if series:
                chart_path = charts_dir / f"{stem}_individual_holder_history.svg"
                write_line_chart(
                    chart_path,
                    title=f"{code} {name} 个人股东占比历史",
                    subtitle="前十大流通股东/十大股东中个人股东占比的报告期变迁。",
                    series=series,
                    value_kind="pct",
                )
                chart_entries.append(("个人股东历史占比", chart_path))

        lines: list[str] = [
            f"# {code} {name} 股东成分报告",
            "",
            f"- 生成时间：{dt.datetime.now().isoformat(timespec='seconds')}",
            f"- 看涨排名：{rank}；综合评分：{_fmt_number(stock.get('bullish_score'), 3)}。",
            f"- ETF/指数线索：{stock.get('family_ids', '')} / {stock.get('source_etf_codes', '')} {stock.get('source_etf_names', '')}。",
            f"- 增长/交易机制标签：{stock.get('mechanism_tags', '')}。",
            "- 说明：本报告是股东结构研究工件，不构成投资建议。",
            "",
            "## 1. 公司交易与 ETF 线索",
            "",
            "|项目|数值|",
            "|---|---:|",
            f"|行业/地区|{stock.get('industry', '')} / {stock.get('region', '')}|",
            f"|最新价|{_fmt_number(stock.get('latest_price'), 2)}|",
            f"|成交额|{human_amount(to_float(stock.get('quote_amount')) or to_float(stock.get('history_amount')))}|",
            f"|换手率|{_fmt_pct_value(stock.get('quote_turnover_rate') or stock.get('history_turnover_rate'))}|",
            f"|5日收益|{_fmt_ratio(stock.get('stock_ret_5'))}|",
            f"|20日收益|{_fmt_ratio(stock.get('stock_ret_20'))}|",
            f"|20日成交额放大倍数|{_fmt_number(stock.get('history_amount_ratio_20'), 2)}|",
            f"|主力净流入|{human_amount(to_float(stock.get('main_net_inflow')))} ({_fmt_pct_value(stock.get('main_net_inflow_pct'))})|",
            "",
            "## 2. 股东结构快照",
            "",
            "|项目|数值|",
            "|---|---:|",
            f"|流通股东报告期|{stock.get('holder_report_date_free', '')}|",
            f"|前十大流通股东合计占流通股本|{_fmt_pct_value(stock.get('top_free_holder_ratio_sum_pct'))}|",
            f"|其中机构类流通股东占比|{_fmt_pct_value(stock.get('top_free_institution_ratio_sum_pct'))}|",
            f"|其中基金/社保/QFII 类占比|{_fmt_pct_value(stock.get('top_free_fund_like_ratio_sum_pct'))}|",
            f"|前十大流通股东占比变化合计|{_fmt_pct_value(stock.get('top_free_hold_ratio_change_sum_pct'))}|",
            f"|第一大流通股东|{stock.get('top_free_holder_name', '')} / {stock.get('top_free_holder_type', '')} / {_fmt_pct_value(stock.get('top_free_holder_ratio_pct'))}|",
            f"|十大股东报告期|{stock.get('holder_report_date_total', '')}|",
            f"|前十大股东合计占总股本|{_fmt_pct_value(stock.get('top_total_holder_ratio_sum_pct'))}|",
            f"|其中机构类十大股东占比|{_fmt_pct_value(stock.get('top_total_institution_ratio_sum_pct'))}|",
            f"|前十大股东占比变化合计|{_fmt_pct_value(stock.get('top_total_hold_ratio_change_sum_pct'))}|",
            f"|第一大股东|{stock.get('top_total_holder_name', '')} / {stock.get('top_total_holder_type', '')} / {_fmt_pct_value(stock.get('top_total_holder_ratio_pct'))}|",
            "",
        ]
        if chart_entries:
            lines.extend(["## 3. 股东占比饼图", ""])
            for label, chart_path in chart_entries:
                rel = chart_path.relative_to(REPORTS_DIR) if chart_path.is_relative_to(REPORTS_DIR) else chart_path
                lines.append(f"![{label}](../{rel.as_posix()})")
                lines.append("")
        lines.extend(["## 4. 历史股东变迁（个人股东重点）", ""])
        if stock_history_summary.empty:
            lines.append("- 暂无历史股东变迁数据。")
            lines.append("")
        else:
            summary = stock_history_summary.sort_values(["report_date", "holder_kind"], ascending=[False, True])
            lines.append("|报告期|口径|前十大占比|个人股东数|个人股东占比|个人股东|机构占比|基金/社保/QFII占比|")
            lines.append("|---|---|---:|---:|---:|---|---:|---:|")
            for _, item in summary.head(16).iterrows():
                kind_label = "流通股东" if item.get("holder_kind") == "free" else "十大股东"
                lines.append(
                    f"|{item.get('report_date', '')}|{kind_label}|{_fmt_pct_value(item.get('top10_ratio_sum_pct'))}|"
                    f"{int(to_float(item.get('individual_holder_count'), 0.0) or 0)}|"
                    f"{_fmt_pct_value(item.get('individual_holder_ratio_sum_pct'))}|"
                    f"{item.get('individual_holder_names', '')}|"
                    f"{_fmt_pct_value(item.get('institution_holder_ratio_sum_pct'))}|"
                    f"{_fmt_pct_value(item.get('fund_like_holder_ratio_sum_pct'))}|"
                )
            changes = stock_individual_changes.sort_values(
                ["to_report_date", "abs_ratio_delta_pct"], ascending=[False, False]
            ) if "abs_ratio_delta_pct" in stock_individual_changes.columns else stock_individual_changes
            lines.extend(["", "### 个人股东逐期变化", ""])
            if changes.empty:
                lines.append("- 最近报告期内前十大名单未出现个人股东，或个人股东无可计算变化。")
            else:
                lines.append("|从|到|口径|个人股东|状态|上一期占比|本期占比|占比变化|上一期排名|本期排名|")
                lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|")
                for _, item in changes.head(24).iterrows():
                    kind_label = "流通" if item.get("holder_kind") == "free" else "总股本"
                    lines.append(
                        f"|{item.get('from_report_date', '')}|{item.get('to_report_date', '')}|{kind_label}|"
                        f"{item.get('holder_name', '')}|{item.get('status', '')}|"
                        f"{_fmt_pct_value(item.get('prev_ratio_pct'))}|{_fmt_pct_value(item.get('current_ratio_pct'))}|"
                        f"{_fmt_pct_value(item.get('ratio_delta_pct'))}|{_fmt_number(item.get('prev_rank'), 0)}|"
                        f"{_fmt_number(item.get('current_rank'), 0)}|"
                    )
            lines.append("")
        lines.extend(["## 5. 股东明细", ""])
        lines.extend(_holder_detail_table(stock_details, "free"))
        lines.append("")
        lines.extend(_holder_detail_table(stock_details, "total"))
        lines.extend(
            [
                "",
                "## 6. 解读要点",
                "",
                "- 若“成交额放大 + 高换手交易”同时出现，说明上涨背后有真实交易活跃度，但也可能伴随波动放大。",
                "- 前十大流通股东占比越高，筹码越集中；机构/基金占比越高，越需要继续核对定期报告、基金持仓和是否存在被动指数持仓。",
                "- 个人股东历史变迁重点看三类信号：连续增持、新进进入前十大、以及从前十大退出；个人股东占比变化越大，越需要核对公告和实际控制人/高管身份。",
                "- 股东占比变化为增持/减持线索，需结合公告日、股价位置和成交额验证，不应单独作为买卖依据。",
                "",
                "## 数据源",
                "",
                f"- 股东数据：{SOURCE_NOTE['shareholder']} ({SOURCE_NOTE['shareholder_reports']})",
                f"- 行情与 K 线：{SOURCE_NOTE['stock_spot']} / {SOURCE_NOTE['stock_kline']}",
            ]
        )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        reports.append({"stock_code": code, "stock_name": name, "path": str(md_path)})
    index_path = out_dir / "index.md"
    index_lines = ["# 公司股东成分报告索引", ""]
    for item in reports:
        index_lines.append(f"- [{item['stock_code']} {item['stock_name']}]({Path(item['path']).name})")
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    reports.insert(0, {"stock_code": "INDEX", "stock_name": "股东成分报告索引", "path": str(index_path)})
    return reports


def write_markdown_report(
    path: Path,
    *,
    rows: pd.DataFrame,
    etf_signals: pd.DataFrame,
    candidate_count: int,
    details_path: Path,
    history_path: Path,
    individual_changes_path: Path,
    company_reports: list[dict[str, str]],
    chart_paths: dict[str, str],
    errors: dict[str, str],
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.extend(
        [
            "# ETF 自动信号下钻：看涨股票成交额与股东占比深挖",
            "",
            f"- 生成时间：{dt.datetime.now().isoformat(timespec='seconds')}",
            f"- ETF 信号文件：`{args.predictions}`",
            f"- 相关股票文件：`{args.related_stocks}`",
            f"- 入池 ETF 信号：{len(etf_signals)} 只；股票候选池：{candidate_count} 只；输出看涨股票：{len(rows)} 只。",
            "- 口径：成交额/换手来自东方财富 A 股行情与日 K；股东占比来自东方财富股东分析十大流通股东/十大股东。",
            "- 说明：本报告是量化研究工件，不构成投资建议。",
            "",
            "## 1. ETF 信号入口",
            "",
            "|排名|日期|ETF|名称|指数族|预测5日收益|信号原因|",
            "|---:|---|---|---|---|---:|---|",
        ]
    )
    for _, row in etf_signals.head(20).iterrows():
        lines.append(
            f"|{_fmt_number(row.get('pred_score_rank'), 0)}|{row.get('date', '')}|{str(row.get('code', '')).zfill(6)}|"
            f"{row.get('name', '')}|{row.get('family_id', '')}|{_fmt_ratio(row.get('pred_fwd_ret_5'))}|{row.get('signal_reason', '')}|"
        )

    lines.extend(["", "## 2. 看涨股票核心清单", ""])
    if rows.empty:
        lines.append("- 暂无可输出的股票行。")
    else:
        lines.append(
            "|排名|股票|行业|ETF/指数线索|机制标签|评分|成交额|换手|5日收益|20日收益|前十大流通股东占比|机构流通占比|第一流通股东|"
        )
        lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, row in rows.head(30).iterrows():
            amount = human_amount(to_float(row.get("quote_amount")) or to_float(row.get("history_amount")))
            lines.append(
                f"|{int(row.get('bullish_rank') or 0)}|{row.get('stock_code')} {row.get('stock_name')}|{row.get('industry', '')}|"
                f"{row.get('family_ids', '')} / {row.get('source_etf_codes', '')}|{row.get('mechanism_tags', '')}|"
                f"{_fmt_number(row.get('bullish_score'), 3)}|{amount}|{_fmt_pct_value(row.get('quote_turnover_rate') or row.get('history_turnover_rate'))}|"
                f"{_fmt_ratio(row.get('stock_ret_5'))}|{_fmt_ratio(row.get('stock_ret_20'))}|"
                f"{_fmt_pct_value(row.get('top_free_holder_ratio_sum_pct'))}|{_fmt_pct_value(row.get('top_free_institution_ratio_sum_pct'))}|"
                f"{row.get('top_free_holder_name', '')} ({_fmt_pct_value(row.get('top_free_holder_ratio_pct'))})|"
            )

    lines.extend(["", "## 3. 机制拆解", ""])
    if not rows.empty:
        top_industries = rows["industry"].fillna("未知").replace("", "未知").value_counts().head(8)
        if not top_industries.empty:
            lines.append("- 行业集中：" + "，".join(f"{idx}({val})" for idx, val in top_industries.items()) + "。")
        avg_amount = rows["quote_amount"].dropna().mean() if "quote_amount" in rows.columns else math.nan
        avg_turnover = rows["quote_turnover_rate"].dropna().mean() if "quote_turnover_rate" in rows.columns else math.nan
        avg_holder = rows["top_free_holder_ratio_sum_pct"].dropna().mean() if "top_free_holder_ratio_sum_pct" in rows.columns else math.nan
        lines.append(
            f"- 交易确认：输出股票平均成交额约 {human_amount(avg_amount)}，平均换手 {_fmt_pct_value(avg_turnover)}；"
            "成交额放大与高换手会被写入 `mechanism_tags`。"
        )
        lines.append(
            f"- 股东结构：前十大流通股东平均占比 {_fmt_pct_value(avg_holder)}；"
            "机构/基金类持股占比单独保存在 CSV，便于过滤“筹码集中 + 机构参与”的股票。"
        )
        top_amount = rows.sort_values("quote_amount", ascending=False).head(5)
        if not top_amount.empty:
            lines.append(
                "- 成交额最活跃："
                + "，".join(f"{r.stock_code}{r.stock_name}({human_amount(to_float(r.quote_amount))})" for r in top_amount.itertuples(index=False))
                + "。"
            )
        top_holder = rows.sort_values("top_free_holder_ratio_sum_pct", ascending=False).head(5)
        if not top_holder.empty:
            lines.append(
                "- 流通股东最集中："
                + "，".join(
                    f"{r.stock_code}{r.stock_name}({_fmt_pct_value(getattr(r, 'top_free_holder_ratio_sum_pct', None))})"
                    for r in top_holder.itertuples(index=False)
                )
                + "。"
            )
    lines.extend(
        [
            "",
            "## 4. 输出文件",
            "",
            f"- 看涨股票深挖表：`{path.with_suffix('.csv')}`",
            f"- 股东明细表：`{details_path}`",
            f"- 历史股东变迁表：`{history_path}`",
            f"- 个人股东逐期变化表：`{individual_changes_path}`",
            f"- 摘要 JSON：`{path.with_suffix('.json')}`",
        ]
    )
    if company_reports:
        index_report = company_reports[0]
        lines.append(f"- 公司股东成分报告索引：`{index_report.get('path')}`")
        for item in company_reports[1:11]:
            lines.append(f"  - `{item.get('stock_code')}` {item.get('stock_name')}：`{item.get('path')}`")
    if chart_paths:
        for label, chart_path in chart_paths.items():
            rel = Path(chart_path)
            lines.append(f"- 图表 {label}：`{rel}`")
    if errors:
        lines.extend(["", "## 5. 拉取异常", ""])
        for key, value in list(errors.items())[:30]:
            lines.append(f"- `{key}`：{value}")
        if len(errors) > 30:
            lines.append(f"- 其余 {len(errors) - 30} 条异常见摘要 JSON。")
    lines.extend(
        [
            "",
            "## 数据源与风险提示",
            "",
            f"- 行情列表：{SOURCE_NOTE['stock_spot']}",
            f"- 日 K：{SOURCE_NOTE['stock_kline']}",
            f"- 股东分析：{SOURCE_NOTE['shareholder']} ({SOURCE_NOTE['shareholder_reports']})",
            "- 东方财富页面也声明相关信息仅用于传播更多信息，不构成投资建议；实际交易需再核对交易所公告、财报与风险承受能力。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_deep_dive(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path = Path(args.predictions)
    related_path = Path(args.related_stocks)
    predictions = load_predictions(predictions_path)
    etf_signals = select_etf_signals(
        predictions,
        top_etfs=args.top_etfs,
        min_etf_pred=args.min_etf_pred,
        include_top=not args.no_include_top_etfs,
    )
    if args.refresh_related or not related_path.exists():
        related = refresh_related_stocks_from_predictions(predictions, per_family=max(args.stocks_per_family, 1), force=args.force)
        if not related.empty:
            related.to_csv(related_path, index=False)
    else:
        related = load_related_stocks(related_path)
    candidates = build_stock_candidates(
        related,
        etf_signals,
        stocks_per_family=args.stocks_per_family,
        max_candidates=args.max_candidates,
    )
    candidates_path = Path(args.out_prefix).with_name(Path(args.out_prefix).name + "_candidates.csv")
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_path, index=False)

    trading_rows, history_errors = merge_quote_and_history(candidates, history_days=args.history_days, end=args.end, force=args.force)
    prelim = rank_bullish_rows(trading_rows, top_stocks=args.top_stocks, include_shareholders=False)
    shareholder_rows, shareholder_details, shareholder_errors = enrich_shareholders(prelim, force=args.force)
    final = rank_bullish_rows(shareholder_rows, top_stocks=args.top_stocks, include_shareholders=True)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    details_path = out_prefix.with_name(out_prefix.name + "_shareholders.csv")
    holder_history_path = out_prefix.with_name(out_prefix.name + "_holder_history.csv")
    individual_changes_path = out_prefix.with_name(out_prefix.name + "_individual_holder_changes.csv")
    md_path = out_prefix.with_suffix(".md")
    json_path = out_prefix.with_suffix(".json")
    final = final[_ordered_output_columns(final)] if not final.empty else final
    holder_history = pd.DataFrame()
    holder_history_summary = pd.DataFrame()
    individual_changes = pd.DataFrame()
    holder_history_errors: dict[str, str] = {}
    if not args.no_holder_history and args.holder_history_periods > 1:
        holder_history, holder_history_summary, individual_changes, holder_history_errors = enrich_shareholder_history(
            final,
            periods=args.holder_history_periods,
            force=args.force,
        )
    final.to_csv(csv_path, index=False)
    shareholder_details.to_csv(details_path, index=False)
    holder_history.to_csv(holder_history_path, index=False)
    individual_changes.to_csv(individual_changes_path, index=False)
    chart_paths = write_deep_dive_charts(final, REPORTS_DIR / "charts")
    company_reports: list[dict[str, str]] = []
    if not args.no_company_reports:
        company_reports = write_company_shareholder_reports(
            rows=final,
            details=shareholder_details,
            history_summary=holder_history_summary,
            individual_changes=individual_changes,
            out_dir=Path(args.company_report_dir),
            charts_dir=REPORTS_DIR / "charts" / "shareholder_composition",
        )

    errors = {
        **{f"history:{k}": v for k, v in history_errors.items()},
        **{f"shareholder:{k}": v for k, v in shareholder_errors.items()},
        **{f"holder_history:{k}": v for k, v in holder_history_errors.items()},
    }
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "predictions_path": str(predictions_path),
        "related_stocks_path": str(related_path),
        "out_csv": str(csv_path),
        "out_markdown": str(md_path),
        "out_shareholders": str(details_path),
        "out_holder_history": str(holder_history_path),
        "out_individual_holder_changes": str(individual_changes_path),
        "out_candidates": str(candidates_path),
        "company_reports": company_reports,
        "charts": chart_paths,
        "source_note": SOURCE_NOTE,
        "selected_etf_count": int(len(etf_signals)),
        "candidate_stock_count": int(len(candidates)),
        "bullish_stock_count": int(len(final)),
        "holder_history_rows": int(len(holder_history)),
        "individual_holder_change_rows": int(len(individual_changes)),
        "selected_etfs": etf_signals.head(50).to_dict("records"),
        "top_stocks": final.head(50).to_dict("records"),
        "errors": errors,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown_report(
        md_path,
        rows=final,
        etf_signals=etf_signals,
        candidate_count=len(candidates),
        details_path=details_path,
        history_path=holder_history_path,
        individual_changes_path=individual_changes_path,
        company_reports=company_reports,
        chart_paths=chart_paths,
        errors=errors,
        args=args,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deep-dive ETF signals into bullish related stocks with trading amount and shareholder ratios.")
    p.add_argument("--predictions", default=str(REPORTS_DIR / "latest_predictions_lite.csv"), help="ETF prediction/candidate CSV from hp_ml.lite_train.")
    p.add_argument("--related-stocks", default=str(REPORTS_DIR / "related_stocks_lite.csv"), help="Related/component stock CSV from hp_ml.lite_train.")
    p.add_argument("--out-prefix", default=str(REPORTS_DIR / "bullish_stock_deep_dive"), help="Output prefix for CSV/Markdown/JSON.")
    p.add_argument("--top-etfs", type=int, default=8, help="Also include top N ETF prediction rows as signal entry points.")
    p.add_argument("--min-etf-pred", type=float, default=0.0, help="Include ETFs with prediction >= this threshold; set very high with --top-etfs to force top-only.")
    p.add_argument("--no-include-top-etfs", action="store_true", help="Only use the min prediction filter; do not add top-N ETF rows.")
    p.add_argument("--stocks-per-family", type=int, default=25, help="Take top N component/related stocks per selected ETF family before dedupe.")
    p.add_argument("--max-candidates", type=int, default=80, help="Maximum stock candidates to enrich with quote/history before bullish ranking.")
    p.add_argument("--top-stocks", type=int, default=30, help="Number of bullish stock rows to enrich with shareholders and output.")
    p.add_argument("--history-days", type=int, default=120, help="Lookback trading-day budget for stock momentum/amount features.")
    p.add_argument("--end", default=today_yyyymmdd(), help="End date for K-line fetch, YYYYMMDD. Defaults to today.")
    p.add_argument("--company-report-dir", default=str(REPORTS_DIR / "shareholder_composition"), help="Directory for per-company shareholder composition reports.")
    p.add_argument("--no-company-reports", action="store_true", help="Do not write per-company shareholder composition Markdown reports.")
    p.add_argument("--holder-history-periods", type=int, default=8, help="Number of shareholder report periods to keep for historical holder-change analysis.")
    p.add_argument("--no-holder-history", action="store_true", help="Disable historical shareholder and individual-holder change analysis.")
    p.add_argument("--refresh-related", action="store_true", help="Refresh related/component stocks from Eastmoney instead of only reading the CSV.")
    p.add_argument("--force", action="store_true", help="Ignore same-day caches and hit live endpoints.")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    summary = run_deep_dive(args)
    print(f"看涨股票深挖表：{summary['out_csv']}")
    print(f"股东明细表：{summary['out_shareholders']}")
    print(f"历史股东变迁表：{summary['out_holder_history']}")
    print(f"个人股东逐期变化表：{summary['out_individual_holder_changes']}")
    print(f"Markdown 报告：{summary['out_markdown']}")
    print(f"候选池：{summary['out_candidates']}")
    if summary.get("company_reports"):
        print(f"公司股东成分报告索引：{summary['company_reports'][0]['path']}")
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
