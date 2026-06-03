"""Zero-third-party fallback pipeline for CSI broad ETF mining and training.

This module intentionally avoids pandas/numpy/sklearn so the project can still
pull data and train in a fresh Python environment.  The richer `hp_ml.train`
module uses pandas + scikit-learn when those dependencies are installed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pickle
import random
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from .charts import write_column_chart, write_horizontal_bar_chart, write_line_chart
from .config import (
    BROAD_INDEX_FAMILIES,
    ENHANCED_TERMS,
    FALLBACK_BROAD_ETF_SEEDS,
    HISTORY_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    STYLE_OR_THEME_TERMS,
)

ETF_LIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
FUND_CODE_SEARCH_URL = "https://fund.eastmoney.com/js/fundcode_search.js"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
LISTED_ETF_PREFIXES = ("159", "510", "512", "515", "516", "517", "520", "560", "561", "562", "563", "588", "589")
LIQUIDITY_HINTS = {
    "510300": 100, "510310": 96, "510330": 92, "159919": 90, "510350": 82,
    "510500": 100, "512500": 96, "159922": 88, "510510": 84, "510580": 80,
    "512100": 100, "159845": 96, "159629": 88, "159633": 86, "560010": 84,
    "159338": 90, "159353": 88, "159358": 86, "560510": 84, "560530": 82, "563860": 80,
}

SPOT_MAP = {
    "f12": "code",
    "f14": "name",
    "f2": "latest_price",
    "f3": "pct_chg",
    "f4": "price_change",
    "f5": "volume",
    "f6": "amount",
    "f15": "high",
    "f16": "low",
    "f17": "open",
    "f18": "prev_close",
    "f8": "turnover_rate",
}
HISTORY_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "price_change",
    "turnover_rate",
]
BASE_FEATURES = [
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


def today_yyyymmdd() -> str:
    return dt.date.today().strftime("%Y%m%d")


def ensure_dirs() -> None:
    for path in (RAW_DIR, HISTORY_DIR, PROCESSED_DIR, REPORTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def http_json(url: str, params: dict[str, Any], timeout: int = 8, attempts: int = 3) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{url}?{query}",
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", "ignore").strip()
            if text and not text.startswith("{"):
                left = text.find("{")
                right = text.rfind("}")
                text = text[left : right + 1]
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last_error}")


def to_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "-"):
        return default
    try:
        value = float(value)
    except Exception:  # noqa: BLE001
        return default
    if math.isfinite(value):
        return value
    return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKC", str(name or "")).upper().replace(" ", "")


def classify_name(name: str) -> tuple[str | None, str | None, str | None]:
    text = normalize_name(name)
    if "ETF" not in text:
        return None, None, None
    for family in BROAD_INDEX_FAMILIES:
        for pattern in family.patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return family.family_id, family.display_name, pattern
    return None, None, None


def excluded(name: str, include_enhanced: bool, include_style: bool) -> str | None:
    text = normalize_name(name)
    if not include_enhanced and any(term.upper() in text for term in ENHANCED_TERMS):
        return "enhanced"
    if not include_style:
        terms = [term for term in STYLE_OR_THEME_TERMS if term.upper() in text]
        if terms:
            return "style_or_theme:" + "/".join(terms[:4])
    return None


def fetch_spot(force: bool = False) -> list[dict[str, Any]]:
    cache = RAW_DIR / f"etf_spot_{today_yyyymmdd()}.csv"
    if cache.exists() and not force:
        rows = read_csv_rows(cache)
        for row in rows:
            for key in SPOT_MAP.values():
                if key not in {"code", "name", "fetched_at"}:
                    row[key] = to_float(row.get(key))
        return rows
    payload = http_json(
        ETF_LIST_URL,
        {
            "pn": 1,
            "pz": 10000,
            "po": 1,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f6",
            "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024",
            "fields": ",".join(SPOT_MAP),
        },
        timeout=12,
        attempts=4,
    )
    raw_rows = ((payload.get("data") or {}).get("diff")) or []
    if not raw_rows:
        raise RuntimeError("Eastmoney ETF spot endpoint returned no rows")
    rows: list[dict[str, Any]] = []
    fetched_at = dt.datetime.now().isoformat(timespec="seconds")
    for raw in raw_rows:
        row = {dst: raw.get(src) for src, dst in SPOT_MAP.items()}
        row["code"] = str(row.get("code", "")).zfill(6)
        for key in SPOT_MAP.values():
            if key not in {"code", "name"}:
                row[key] = to_float(row.get(key))
        row["fetched_at"] = fetched_at
        rows.append(row)
    write_csv_rows(cache, rows)
    return rows


def fetch_fundcode_search(force: bool = False) -> list[dict[str, Any]]:
    """Fallback discovery source: Eastmoney fund-code search JS.

    This endpoint is slower than the ETF quote list and has no live liquidity
    fields, but it is useful when the quote API returns 5xx.  We keep only
    exchange-listed ETF code prefixes and drop ETF-link funds.
    """

    cache = RAW_DIR / f"fundcode_search_{today_yyyymmdd()}.csv"
    if cache.exists() and not force:
        return read_csv_rows(cache)

    req = urllib.request.Request(
        FUND_CODE_SEARCH_URL,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/125 Safari/537.36"},
    )
    text = ""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                text = resp.read().decode("utf-8-sig", "ignore")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    if not text:
        raise RuntimeError(f"fundcode_search.js failed: {last_error}")

    left = text.find("[[")
    right = text.rfind("]]")
    if left < 0 or right < left:
        raise RuntimeError("fundcode_search.js did not contain a JSON array")
    data = json.loads(text[left : right + 2])
    rows: list[dict[str, Any]] = []
    for item in data:
        if len(item) < 4:
            continue
        code = str(item[0]).zfill(6)
        name = str(item[2])
        fund_type = str(item[3])
        if not code.startswith(LISTED_ETF_PREFIXES):
            continue
        normalized = normalize_name(name)
        if "ETF" not in normalized or "联接" in normalized or "LOF" in normalized:
            continue
        rows.append({"code": code, "name": name, "fund_type": fund_type, "source": "eastmoney_fundcode_search"})
    write_csv_rows(cache, rows)
    return rows


def discover_universe(
    max_per_family: int = 3,
    min_amount: float = 0.0,
    include_enhanced: bool = False,
    include_style: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    source_name = "eastmoney_spot"
    try:
        spot = fetch_spot(force=force)
    except Exception:
        try:
            spot = fetch_fundcode_search(force=force)
            source_name = "eastmoney_fundcode_search"
        except Exception:
            spot = []
    family_meta = {family.family_id: family for family in BROAD_INDEX_FAMILIES}
    rows: list[dict[str, Any]] = []
    for item in spot:
        family_id, display_name, pattern = classify_name(str(item.get("name", "")))
        if family_id is None:
            continue
        reason = excluded(str(item.get("name", "")), include_enhanced, include_style)
        if reason:
            continue
        amount = float(to_float(item.get("amount"), 0.0) or 0.0)
        if amount < min_amount:
            continue
        volume = float(to_float(item.get("volume"), 0.0) or 0.0)
        row = dict(item)
        meta = family_meta[family_id]
        row.update(
            {
                "family_id": family_id,
                "display_name": display_name,
                "index_code": meta.index_code or "",
                "description": meta.description,
                "matched_pattern": pattern or "",
                "source": item.get("source") or source_name,
                "rank_score": (
                    math.log1p(max(amount, 0.0)) + 0.15 * math.log1p(max(volume, 0.0))
                    if amount > 0 or volume > 0
                    else float(LIQUIDITY_HINTS.get(str(item.get("code", "")).zfill(6), 0))
                ),
            }
        )
        rows.append(row)
    if not rows:
        for idx, seed in enumerate(FALLBACK_BROAD_ETF_SEEDS, 1):
            meta = family_meta[seed["family_id"]]
            rows.append(
                {
                    **seed,
                    "display_name": meta.display_name,
                    "index_code": meta.index_code or "",
                    "description": meta.description,
                    "source": "fallback_seed",
                    "rank_score": 0.0,
                    "selected_rank_in_family": idx,
                }
            )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for family_id in sorted(grouped):
        family_rows = sorted(grouped[family_id], key=lambda r: (float(r.get("rank_score") or 0.0), float(r.get("amount") or 0.0)), reverse=True)
        for rank, row in enumerate(family_rows, 1):
            row["selected_rank_in_family"] = rank
            if max_per_family <= 0 or rank <= max_per_family:
                selected.append(row)
    selected.sort(key=lambda r: (str(r.get("family_id")), int(r.get("selected_rank_in_family") or 0)))
    return selected


def market_id(code: str) -> int:
    return 1 if str(code).zfill(6).startswith(("5", "6", "9")) else 0


def market_prefix(code: str) -> str:
    return "sh" if str(code).zfill(6).startswith(("5", "6", "9")) else "sz"


def yyyymmdd_to_iso(value: str) -> str:
    value = str(value)
    if "-" in value:
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def fetch_history_tencent(code: str, start: str, end: str, adjust: str = "qfq") -> list[dict[str, Any]]:
    """Fetch ETF daily K-lines from Tencent as a fallback.

    Tencent returns date/open/close/high/low/volume.  Amount and turnover are
    unavailable, so downstream feature imputation handles them.
    """

    code = str(code).zfill(6)
    symbol = market_prefix(code) + code
    fq = "qfq" if adjust.lower() == "qfq" else ""
    # Tencent rejects very large counts (e.g. 3000) for some ETF symbols; 2000
    # trading days is enough for the default 2018+ daily research window.
    param = f"{symbol},day,{yyyymmdd_to_iso(start)},{yyyymmdd_to_iso(end)},2000,{fq}".rstrip(",")
    payload = http_json(TENCENT_KLINE_URL, {"param": param}, timeout=10, attempts=3)
    node = (payload.get("data") or {}).get(symbol) or {}
    lines = node.get("qfqday") or node.get("hfqday") or node.get("day") or []
    if not lines:
        raise RuntimeError(f"No Tencent K-line rows for {code}")
    rows: list[dict[str, Any]] = []
    prev_close: float | None = None
    for item in lines:
        if len(item) < 6:
            continue
        date_s, open_s, close_s, high_s, low_s, volume_s = item[:6]
        open_v = to_float(open_s)
        close_v = to_float(close_s)
        high_v = to_float(high_s)
        low_v = to_float(low_s)
        volume_v = to_float(volume_s)
        price_change = close_v - prev_close if close_v is not None and prev_close is not None else None
        pct_chg = price_change / prev_close * 100 if price_change is not None and prev_close not in (None, 0) else None
        amplitude = (high_v - low_v) / prev_close * 100 if high_v is not None and low_v is not None and prev_close not in (None, 0) else None
        rows.append(
            {
                "code": code,
                "date": date_s,
                "open": open_v,
                "close": close_v,
                "high": high_v,
                "low": low_v,
                "volume": volume_v,
                "amount": None,
                "amplitude": amplitude,
                "pct_chg": pct_chg,
                "price_change": price_change,
                "turnover_rate": None,
            }
        )
        if close_v is not None:
            prev_close = close_v
    if not rows:
        raise RuntimeError(f"No parsed Tencent K-line rows for {code}")
    return rows


def fetch_history(code: str, start: str, end: str, adjust: str = "qfq", force: bool = False) -> list[dict[str, Any]]:
    code = str(code).zfill(6)
    adjust_key = {"": 0, "none": 0, "raw": 0, "qfq": 1, "hfq": 2}.get(adjust.lower(), 1)
    cache = HISTORY_DIR / f"{code}_{start}_{end}_{adjust}.csv"
    if cache.exists() and not force:
        rows = read_csv_rows(cache)
        for row in rows:
            row["code"] = str(row.get("code", code)).zfill(6)
            for col in HISTORY_COLUMNS[1:]:
                row[col] = to_float(row.get(col))
        return rows
    try:
        payload = http_json(
            KLINE_URL,
            {
                "secid": f"{market_id(code)}.{code}",
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": 101,
                "fqt": adjust_key,
                "beg": start,
                "end": end,
            },
            timeout=8,
            attempts=3,
        )
        lines = ((payload.get("data") or {}).get("klines")) or []
        if not lines:
            raise RuntimeError(f"No Eastmoney K-line rows for {code}")
        rows = []
        for line in lines:
            parts = line.split(",")
            row = {col: parts[i] if i < len(parts) else "" for i, col in enumerate(HISTORY_COLUMNS)}
            row["code"] = code
            for col in HISTORY_COLUMNS[1:]:
                row[col] = to_float(row.get(col))
            rows.append(row)
    except Exception:
        rows = fetch_history_tencent(code, start=start, end=end, adjust=adjust)
    write_csv_rows(cache, rows, ["code"] + HISTORY_COLUMNS)
    return rows


def pct(values: list[float | None], i: int, n: int) -> float | None:
    if i - n < 0:
        return None
    a = values[i - n]
    b = values[i]
    if a in (None, 0) or b is None:
        return None
    return b / a - 1.0


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def std(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def rolling(values: list[float | None], i: int, n: int) -> list[float]:
    left = max(0, i - n + 1)
    return [v for v in values[left : i + 1] if v is not None and math.isfinite(v)]


def make_features_for_history(hist: list[dict[str, Any]], meta: dict[str, Any], horizon: int, family_ids: list[str]) -> list[dict[str, Any]]:
    hist = sorted(hist, key=lambda r: str(r["date"]))
    closes = [to_float(r.get("close")) for r in hist]
    highs = [to_float(r.get("high")) for r in hist]
    lows = [to_float(r.get("low")) for r in hist]
    volumes = [to_float(r.get("volume")) for r in hist]
    amounts = [to_float(r.get("amount")) for r in hist]
    rets1 = [pct(closes, i, 1) for i in range(len(hist))]
    start_date = dt.date.fromisoformat(str(hist[0]["date"])[:10])
    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(hist):
        close = closes[i]
        date_obj = dt.date.fromisoformat(str(raw["date"])[:10])
        row: dict[str, Any] = {
            "date": str(date_obj),
            "code": str(meta.get("code", raw.get("code"))).zfill(6),
            "name": meta.get("name", ""),
            "family_id": meta.get("family_id", ""),
            "close": close,
        }
        for n in (1, 3, 5, 10, 20, 60):
            row[f"ret_{n}"] = pct(closes, i, n)
        for n in (5, 20, 60):
            row[f"vol_{n}"] = std(rolling(rets1, i, n))
        ma5 = mean(rolling(closes, i, 5))
        ma20 = mean(rolling(closes, i, 20))
        ma60 = mean(rolling(closes, i, 60))
        row["ma_gap_5_20"] = ma5 / ma20 - 1 if ma5 is not None and ma20 not in (None, 0) else None
        row["ma_gap_20_60"] = ma20 / ma60 - 1 if ma20 is not None and ma60 not in (None, 0) else None
        for n in (20, 60):
            vals = rolling(closes, i, n)
            mx = max(vals) if vals else None
            row[f"drawdown_{n}"] = close / mx - 1 if close is not None and mx not in (None, 0) else None
        amount = amounts[i]
        amount20 = mean(rolling(amounts, i, 20))
        amount_std20 = std(rolling(amounts, i, 20))
        row["amount_log"] = math.log1p(max(amount or 0.0, 0.0))
        row["amount_z20"] = (amount - amount20) / amount_std20 if amount is not None and amount20 is not None and amount_std20 not in (None, 0) else None
        row["turnover_rate"] = to_float(raw.get("turnover_rate"))
        row["amplitude"] = to_float(raw.get("amplitude"))
        row["intraday_range"] = (highs[i] - lows[i]) / close if highs[i] is not None and lows[i] is not None and close not in (None, 0) else None
        row["volume_chg_5"] = pct(volumes, i, 5)
        row["liquidity_shock_20"] = amount / amount20 - 1 if amount is not None and amount20 not in (None, 0) else None
        month = date_obj.month
        row["month_sin"] = math.sin(2 * math.pi * month / 12.0)
        row["month_cos"] = math.cos(2 * math.pi * month / 12.0)
        row["days_since_start"] = (date_obj - start_date).days
        for fam in family_ids:
            row[f"family_{fam}"] = 1.0 if fam == meta.get("family_id") else 0.0
        target = pct(closes, i + horizon, horizon) if i + horizon < len(closes) else None
        row[f"fwd_ret_{horizon}"] = target
        row["is_trainable"] = target is not None
        rows.append(row)
    return rows


def build_panel(histories: dict[str, list[dict[str, Any]]], universe: list[dict[str, Any]], horizon: int) -> tuple[list[dict[str, Any]], list[str], str]:
    meta_by_code = {str(row["code"]).zfill(6): row for row in universe}
    family_ids = sorted({str(row["family_id"]) for row in universe})
    feature_cols = BASE_FEATURES + [f"family_{fam}" for fam in family_ids]
    panel: list[dict[str, Any]] = []
    for code, hist in histories.items():
        meta = meta_by_code.get(str(code).zfill(6), {"code": code})
        panel.extend(make_features_for_history(hist, meta, horizon, family_ids))
    panel.sort(key=lambda r: (str(r["date"]), str(r["code"])))
    return panel, feature_cols, f"fwd_ret_{horizon}"


def time_split(rows: list[dict[str, Any]], target_col: str, test_days: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    trainable = [r for r in rows if r.get("is_trainable") and to_float(r.get(target_col)) is not None]
    dates = sorted({str(r["date"]) for r in trainable})
    if len(dates) < 20:
        raise RuntimeError(f"Too few unique dates for training: {len(dates)}")
    split_idx = max(1, len(dates) - test_days)
    if split_idx <= 5 or len(dates) - split_idx < 5:
        split_idx = max(1, int(len(dates) * 0.75))
    split_date = dates[split_idx]
    train = [r for r in trainable if str(r["date"]) < split_date]
    test = [r for r in trainable if str(r["date"]) >= split_date]
    if not train or not test:
        raise RuntimeError("Time split produced empty train/test")
    return train, test, split_date


def fit_preprocessor(train: list[dict[str, Any]], feature_cols: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in feature_cols:
        vals = [to_float(r.get(col)) for r in train]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        m = sum(vals) / len(vals) if vals else 0.0
        if len(vals) > 1:
            s = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
        else:
            s = 1.0
        means[col] = m
        stds[col] = s if s > 1e-12 and math.isfinite(s) else 1.0
    return means, stds


def vectorize(row: dict[str, Any], feature_cols: list[str], means: dict[str, float], stds: dict[str, float]) -> list[float]:
    xs = [1.0]
    for col in feature_cols:
        val = to_float(row.get(col), means[col])
        if val is None or not math.isfinite(val):
            val = means[col]
        xs.append((val - means[col]) / stds[col])
    return xs


def solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    # Augmented matrix with partial pivoting.
    mat = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-12:
            mat[pivot][col] += 1e-8
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
        div = mat[col][col]
        if abs(div) < 1e-12:
            div = 1e-12
        for j in range(col, n + 1):
            mat[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = mat[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                mat[r][j] -= factor * mat[col][j]
    return [mat[i][n] for i in range(n)]


def train_ridge(train: list[dict[str, Any]], feature_cols: list[str], target_col: str, l2: float = 3.0) -> dict[str, Any]:
    means, stds = fit_preprocessor(train, feature_cols)
    p = len(feature_cols) + 1
    a = [[0.0 for _ in range(p)] for _ in range(p)]
    b = [0.0 for _ in range(p)]
    for row in train:
        x = vectorize(row, feature_cols, means, stds)
        y = float(row[target_col])
        for j in range(p):
            b[j] += x[j] * y
            for k in range(p):
                a[j][k] += x[j] * x[k]
    for j in range(1, p):  # do not penalize intercept
        a[j][j] += l2
    weights = solve_linear_system(a, b)
    return {"feature_cols": feature_cols, "means": means, "stds": stds, "weights": weights, "target_col": target_col, "l2": l2}


def predict_one(model: dict[str, Any], row: dict[str, Any]) -> float:
    x = vectorize(row, model["feature_cols"], model["means"], model["stds"])
    return sum(w * xi for w, xi in zip(model["weights"], x))


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def evaluate(test: list[dict[str, Any]], model: dict[str, Any], target_col: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preds: list[dict[str, Any]] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    for row in test:
        pred = predict_one(model, row)
        y = float(row[target_col])
        out = dict(row)
        out["prediction"] = pred
        preds.append(out)
        y_true.append(y)
        y_pred.append(pred)
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))
    mean_y = sum(y_true) / len(y_true)
    denom = sum((v - mean_y) ** 2 for v in y_true)
    r2 = 1.0 - sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / denom if denom > 0 else None
    directional = sum(1 for a, b in zip(y_true, y_pred) if (a >= 0) == (b >= 0)) / len(y_true)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preds:
        grouped[str(row["date"])].append(row)
    ics: list[float] = []
    top_returns: list[float] = []
    bottom_returns: list[float] = []
    all_returns: list[float] = []
    for group in grouped.values():
        if len(group) >= 2:
            ic = corr(rankdata([float(r["prediction"]) for r in group]), rankdata([float(r[target_col]) for r in group]))
            if ic is not None:
                ics.append(ic)
        top = max(group, key=lambda r: float(r["prediction"]))
        bottom = min(group, key=lambda r: float(r["prediction"]))
        top_returns.append(float(top[target_col]))
        bottom_returns.append(float(bottom[target_col]))
        all_returns.append(sum(float(r[target_col]) for r in group) / len(group))
    metrics = {
        "rows": len(test),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": directional,
        "mean_target": mean_y,
        "mean_prediction": sum(y_pred) / len(y_pred),
        "spearman_ic_by_date": sum(ics) / len(ics) if ics else None,
        "top1_mean_fwd_ret": sum(top_returns) / len(top_returns) if top_returns else None,
        "bottom1_mean_fwd_ret": sum(bottom_returns) / len(bottom_returns) if bottom_returns else None,
        "top_bottom_spread": (sum(top_returns) / len(top_returns) - sum(bottom_returns) / len(bottom_returns)) if top_returns and bottom_returns else None,
        "top_vs_all": (sum(top_returns) / len(top_returns) - sum(all_returns) / len(all_returns)) if top_returns and all_returns else None,
    }
    return metrics, preds


def latest_predictions(panel: list[dict[str, Any]], model: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in panel:
        code = str(row["code"])
        if code not in latest or str(row["date"]) > str(latest[code]["date"]):
            latest[code] = row
    rows = []
    pred_col = f"pred_fwd_ret_{horizon}"
    for row in latest.values():
        out = {k: row.get(k) for k in ["date", "code", "name", "family_id", "close", "ret_5", "ret_20", "vol_20", "drawdown_20", "turnover_rate"]}
        out[pred_col] = predict_one(model, row)
        rows.append(out)
    rows.sort(key=lambda r: float(r[pred_col]), reverse=True)
    for i, row in enumerate(rows, 1):
        row["pred_score_rank"] = i
    return rows


def human_amount(value: Any) -> str:
    v = to_float(value)
    if v is None:
        return ""
    if abs(v) >= 1e8:
        return f"{v / 1e8:.2f}亿"
    if abs(v) >= 1e4:
        return f"{v / 1e4:.2f}万"
    return f"{v:.0f}"


def fmt_pct(value: Any) -> str:
    v = to_float(value)
    return "" if v is None else f"{v * 100:.2f}%"


def _markdown_chart(path_text: str) -> str:
    path = Path(path_text)
    try:
        return str(path.relative_to(REPORTS_DIR))
    except ValueError:
        return str(path)


def _holdout_cumulative_series(holdout: list[dict[str, Any]], target_col: str) -> dict[str, list[tuple[str, float]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in holdout:
        if to_float(row.get("prediction")) is None or to_float(row.get(target_col)) is None:
            continue
        grouped[str(row["date"])].append(row)

    top_value = 1.0
    all_value = 1.0
    bottom_value = 1.0
    top_points: list[tuple[str, float]] = []
    all_points: list[tuple[str, float]] = []
    bottom_points: list[tuple[str, float]] = []
    for date_s in sorted(grouped):
        group = grouped[date_s]
        top = max(group, key=lambda r: float(r["prediction"]))
        bottom = min(group, key=lambda r: float(r["prediction"]))
        top_ret = float(top[target_col])
        bottom_ret = float(bottom[target_col])
        all_ret = sum(float(r[target_col]) for r in group) / len(group)
        top_value *= 1.0 + top_ret
        all_value *= 1.0 + all_ret
        bottom_value *= 1.0 + bottom_ret
        top_points.append((date_s, top_value - 1.0))
        all_points.append((date_s, all_value - 1.0))
        bottom_points.append((date_s, bottom_value - 1.0))
    return {"Top1策略": top_points, "全池平均": all_points, "Bottom1": bottom_points}


def generate_charts(
    universe: list[dict[str, Any]],
    metrics: dict[str, Any],
    preds: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    target_col: str,
    horizon: int,
) -> dict[str, str]:
    charts_dir = REPORTS_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    pred_col = f"pred_fwd_ret_{horizon}"

    prediction_rows = [
        (f"{int(row.get('pred_score_rank', 0))}. {row.get('code', '')} {row.get('name', '')}", float(row[pred_col]))
        for row in sorted(preds, key=lambda r: int(r.get("pred_score_rank", 9999)))
        if to_float(row.get(pred_col)) is not None
    ]
    prediction_path = write_horizontal_bar_chart(
        charts_dir / "latest_prediction_rank_lite.svg",
        title=f"最新中证宽基 ETF 预测排序（未来 {horizon} 日）",
        subtitle="横轴为模型预测未来收益率；排序越靠上代表模型分数越高。",
        rows=prediction_rows,
    )

    fam_counts: dict[str, int] = defaultdict(int)
    for row in universe:
        fam_counts[str(row.get("family_id"))] += 1
    family_path = write_column_chart(
        charts_dir / "candidate_family_counts_lite.svg",
        title="候选 ETF 指数族覆盖",
        subtitle="自动发现后按每个指数族保留的 ETF 数量。",
        rows=sorted((family, float(count)) for family, count in fam_counts.items()),
        value_kind="number",
    )

    metric_rows = [
        ("方向准确率", metrics.get("directional_accuracy")),
        ("Top1均值", metrics.get("top1_mean_fwd_ret")),
        ("Top1-全池", metrics.get("top_vs_all")),
        ("Top-Bottom", metrics.get("top_bottom_spread")),
        ("平均真实收益", metrics.get("mean_target")),
        ("平均预测收益", metrics.get("mean_prediction")),
    ]
    metric_path = write_horizontal_bar_chart(
        charts_dir / "holdout_metric_snapshot_lite.svg",
        title="时间切分验证核心指标",
        subtitle="所有数值均按百分比展示；Top1 等指标基于每日预测排序计算。",
        rows=[(name, float(value)) for name, value in metric_rows if to_float(value) is not None],
        value_kind="pct",
        positive_color="#16a34a",
    )

    cumulative_path = write_line_chart(
        charts_dir / "holdout_cumulative_lite.svg",
        title="Holdout 排序策略累计曲线",
        subtitle=f"按测试期每日预测排序选 Top1 / Bottom1，并与全池平均的未来 {horizon} 日收益累计对比。",
        series=_holdout_cumulative_series(holdout, target_col),
        value_kind="pct",
    )
    return {
        "latest_prediction_rank": str(prediction_path),
        "candidate_family_counts": str(family_path),
        "holdout_metric_snapshot": str(metric_path),
        "holdout_cumulative": str(cumulative_path),
    }


def write_summary(
    universe: list[dict[str, Any]],
    metrics: dict[str, Any],
    preds: list[dict[str, Any]],
    feature_cols: list[str],
    horizon: int,
    chart_paths: dict[str, str] | None = None,
) -> None:
    fam_counts: dict[str, int] = defaultdict(int)
    for row in universe:
        fam_counts[str(row.get("family_id"))] += 1
    lines = [
        "# 中证宽基 ETF 自动挖掘与轻量模型训练报告",
        "",
        "本报告由零第三方依赖的 `python -m hp_ml.lite_train` 自动生成；输出用于量化研究，不构成投资建议。",
        "",
        "## 1. 候选 ETF 池",
        "",
        f"- 候选 ETF 数量：{len(universe)}",
        "- 覆盖指数族：" + ", ".join(f"{k}({v})" for k, v in sorted(fam_counts.items())),
        "- 发现方式：扫描东方财富 ETF 行情/基金代码列表，并用中证/沪深宽基名称模式聚焦纯宽基产品。",
        "",
        "|代码|名称|指数族|最新价|成交额|族内排名|",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in sorted(universe, key=lambda r: float(r.get("amount") or 0.0), reverse=True)[:12]:
        lines.append(
            f"|{row.get('code', '')}|{row.get('name', '')}|{row.get('family_id', '')}|"
            f"{row.get('latest_price', '')}|{human_amount(row.get('amount'))}|{row.get('selected_rank_in_family', '')}|"
        )
    lines.extend(
        [
            "",
            "## 2. 训练设置与验证",
            "",
            f"- 预测目标：未来 {horizon} 个交易日收益。",
            f"- 模型：标准化特征 + 岭回归（纯 Python 线性方程求解）。",
            f"- 特征数量：{len(feature_cols)}",
            "",
            "|指标|值|",
            "|---|---:|",
        ]
    )
    for key in ["train_rows", "test_rows", "train_start", "train_end", "test_start", "test_end", "mae", "rmse", "r2", "directional_accuracy", "spearman_ic_by_date", "top1_mean_fwd_ret", "top_vs_all", "top_bottom_spread"]:
        value = metrics.get(key)
        if isinstance(value, float):
            text = f"{value:.6f}"
        else:
            text = str(value)
        lines.append(f"|{key}|{text}|")
    lines.extend(
        [
            "",
            "## 3. 图表结果",
            "",
        ]
    )
    if chart_paths:
        chart_items = [
            ("最新预测排序图", chart_paths.get("latest_prediction_rank")),
            ("候选指数族覆盖图", chart_paths.get("candidate_family_counts")),
            ("验证指标快照图", chart_paths.get("holdout_metric_snapshot")),
            ("Holdout累计曲线", chart_paths.get("holdout_cumulative")),
        ]
        for title, path_text in chart_items:
            if path_text:
                lines.append(f"### {title}")
                lines.append("")
                lines.append(f"![{title}]({_markdown_chart(path_text)})")
                lines.append("")
    else:
        lines.append("- 本次未生成图表。")
        lines.append("")
    lines.extend(
        [
            "## 4. 最新预测排序",
            "",
            "|排名|日期|代码|名称|指数族|收盘价|预测未来收益|近5日|近20日|20日波动|20日回撤|",
            "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    pred_col = f"pred_fwd_ret_{horizon}"
    for row in preds[:12]:
        lines.append(
            f"|{row.get('pred_score_rank')}|{row.get('date')}|{row.get('code')}|{row.get('name')}|"
            f"{row.get('family_id')}|{row.get('close')}|{fmt_pct(row.get(pred_col))}|"
            f"{fmt_pct(row.get('ret_5'))}|{fmt_pct(row.get('ret_20'))}|{fmt_pct(row.get('vol_20'))}|"
            f"{fmt_pct(row.get('drawdown_20'))}|"
        )
    (REPORTS_DIR / "training_summary_lite.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zero-dependency CSI broad ETF discovery and ridge-model training.")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default=today_yyyymmdd())
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--test-days", type=int, default=252)
    p.add_argument("--max-etfs-per-index", type=int, default=3)
    p.add_argument("--min-amount", type=float, default=0.0)
    p.add_argument("--include-enhanced", action="store_true")
    p.add_argument("--include-style", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", "none", "raw"])
    p.add_argument("--l2", type=float, default=3.0)
    p.add_argument("--model-out", default=str(MODELS_DIR / "csi_broad_etf_model_lite.pkl"))
    return p


def main(argv: list[str] | None = None) -> None:
    ensure_dirs()
    args = build_parser().parse_args(argv)
    universe = discover_universe(args.max_etfs_per_index, args.min_amount, args.include_enhanced, args.include_style, args.force)
    write_csv_rows(REPORTS_DIR / "latest_candidates_lite.csv", universe)
    histories: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    for idx, row in enumerate(universe, 1):
        code = str(row["code"]).zfill(6)
        try:
            print(f"[{idx}/{len(universe)}] 拉取 {code} {row.get('name', '')} ...", flush=True)
            histories[code] = fetch_history(code, args.start, args.end, args.adjust, args.force)
            time.sleep(0.08)
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {code}: {exc}", flush=True)
            errors[code] = str(exc)
    if errors:
        (RAW_DIR / "history_errors_lite.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    if not histories:
        raise RuntimeError("No histories fetched; cannot train")
    panel, feature_cols, target_col = build_panel(histories, universe, args.horizon)
    write_csv_rows(PROCESSED_DIR / "training_panel_lite.csv", panel)
    train, test, split_date = time_split(panel, target_col, args.test_days)
    model = train_ridge(train, feature_cols, target_col, l2=args.l2)
    metrics, holdout = evaluate(test, model, target_col)
    metrics.update(
        {
            "model_type": "stdlib_ridge",
            "target_col": target_col,
            "feature_count": len(feature_cols),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": min(r["date"] for r in train),
            "train_end": max(r["date"] for r in train),
            "test_start": min(r["date"] for r in test),
            "test_end": max(r["date"] for r in test),
            "split_date": split_date,
            "candidate_count": len(universe),
            "history_count": len(histories),
            "panel_rows": len(panel),
            "trainable_rows": sum(1 for r in panel if r.get("is_trainable")),
            "feature_cols": feature_cols,
            "latest_candidates_path": str(REPORTS_DIR / "latest_candidates_lite.csv"),
            "latest_predictions_path": str(REPORTS_DIR / "latest_predictions_lite.csv"),
            "training_panel_path": str(PROCESSED_DIR / "training_panel_lite.csv"),
            "model_path": args.model_out,
        }
    )
    preds = latest_predictions(panel, model, args.horizon)
    write_csv_rows(REPORTS_DIR / "latest_predictions_lite.csv", preds)
    write_csv_rows(REPORTS_DIR / "holdout_predictions_lite.csv", holdout)
    chart_paths = generate_charts(universe, metrics, preds, holdout, target_col, args.horizon)
    metrics["chart_paths"] = chart_paths
    (REPORTS_DIR / "training_metrics_lite.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_summary(universe, metrics, preds, feature_cols, args.horizon, chart_paths)
    artifact = {"model": model, "metrics": metrics, "universe": universe, "created_at": dt.datetime.now().isoformat(timespec="seconds")}
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(artifact, fh)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    print(f"\n轻量模型已保存: {model_path}")
    print(f"候选池: {REPORTS_DIR / 'latest_candidates_lite.csv'}")
    print(f"最新预测: {REPORTS_DIR / 'latest_predictions_lite.csv'}")
    print(f"报告: {REPORTS_DIR / 'training_summary_lite.md'}")
    print(f"图表目录: {REPORTS_DIR / 'charts'}")


if __name__ == "__main__":
    main()
