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
EASTMONEY_DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
LISTED_ETF_PREFIXES = ("159", "510", "512", "515", "516", "517", "520", "560", "561", "562", "563", "588", "589")
INDEX_COMPONENT_TYPE_MAP = {
    "CSI_300": ("1",),
    "CSI_500": ("3",),
    "CSI_1000": ("7",),
    "CSI_2000": ("13",),
    "CSI_800": ("1", "3"),  # 中证 800 ~= 沪深 300 + 中证 500
    "CSI_A500": ("6",),
    "CSI_A50": ("8",),
    "CSI_100": ("12",),
}
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


def parse_float_grid(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value not in values:
            values.append(value)
    return values


def parse_int_grid(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in values:
            values.append(value)
    return values


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


def fetch_index_component_type(type_code: str, force: bool = False) -> list[dict[str, Any]]:
    """Fetch current component stocks for one Eastmoney index-component TYPE."""

    type_code = str(type_code)
    cache = RAW_DIR / f"index_components_type_{type_code}_{today_yyyymmdd()}.csv"
    if cache.exists() and not force:
        rows = read_csv_rows(cache)
        for row in rows:
            for key in ("weight", "close_price", "change_rate", "free_cap", "pe", "eps", "roe"):
                if key in row:
                    row[key] = to_float(row.get(key))
        return rows

    payload = http_json(
        EASTMONEY_DATACENTER_URL,
        {
            "reportName": "RPT_INDEX_TS_COMPONENT",
            "columns": "ALL",
            "filter": f'(TYPE="{type_code}")',
            "pageNumber": 1,
            "pageSize": 2500,
            "sortColumns": "WEIGHT",
            "sortTypes": "-1",
            "source": "WEB",
            "client": "WEB",
        },
        timeout=15,
        attempts=3,
    )
    raw_rows = ((payload.get("result") or {}).get("data")) or []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = {
            "index_component_type": type_code,
            "secucode": raw.get("SECUCODE"),
            "stock_code": str(raw.get("SECURITY_CODE", "")).zfill(6),
            "stock_name": raw.get("SECURITY_NAME_ABBR"),
            "close_price": to_float(raw.get("CLOSE_PRICE")),
            "change_rate": to_float(raw.get("CHANGE_RATE")),
            "trade_date": raw.get("MAXTRADEDATE"),
            "industry": raw.get("INDUSTRY"),
            "region": raw.get("REGION"),
            "weight": to_float(raw.get("WEIGHT")),
            "eps": to_float(raw.get("EPS")),
            "roe": to_float(raw.get("ROE")),
            "free_cap": to_float(raw.get("FREE_CAP")),
            "pe": to_float(raw.get("PE")),
            "source": "eastmoney_datacenter:RPT_INDEX_TS_COMPONENT",
        }
        rows.append(row)
    write_csv_rows(cache, rows)
    return rows


def fetch_related_stocks(universe: list[dict[str, Any]], per_family: int, force: bool = False) -> list[dict[str, Any]]:
    """Fetch more related stocks for discovered broad-index ETF families."""

    family_meta = {family.family_id: family for family in BROAD_INDEX_FAMILIES}
    family_ids = sorted({str(row.get("family_id")) for row in universe if row.get("family_id")})
    related: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for family_id in family_ids:
        type_codes = INDEX_COMPONENT_TYPE_MAP.get(family_id, ())
        if not type_codes:
            continue
        stock_by_code: dict[str, dict[str, Any]] = {}
        for type_code in type_codes:
            try:
                rows = fetch_index_component_type(type_code, force=force)
            except Exception as exc:  # noqa: BLE001
                errors[f"{family_id}:{type_code}"] = str(exc)
                continue
            for row in rows:
                code = str(row.get("stock_code", "")).zfill(6)
                current = stock_by_code.get(code)
                if current is None:
                    stock_by_code[code] = dict(row)
                else:
                    # Prefer rows carrying explicit index weight.
                    if to_float(current.get("weight")) is None and to_float(row.get("weight")) is not None:
                        stock_by_code[code] = dict(row)
        rows = list(stock_by_code.values())
        has_weight = any(to_float(row.get("weight")) is not None for row in rows)
        rows.sort(
            key=lambda row: (
                to_float(row.get("weight"), -1.0) if has_weight else to_float(row.get("free_cap"), -1.0),
                to_float(row.get("free_cap"), -1.0),
            ),
            reverse=True,
        )
        meta = family_meta.get(family_id)
        for rank, row in enumerate(rows[: max(0, per_family)], 1):
            item = dict(row)
            item.update(
                {
                    "family_id": family_id,
                    "display_name": meta.display_name if meta else family_id,
                    "index_code": meta.index_code if meta else "",
                    "rank_in_family": rank,
                    "rank_basis": "weight" if has_weight else "free_cap",
                }
            )
            related.append(item)
    if errors:
        (RAW_DIR / "related_stock_errors_lite.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    return related


def market_id(code: str) -> int:
    return 1 if str(code).zfill(6).startswith(("5", "6", "9")) else 0


def market_prefix(code: str) -> str:
    return "sh" if str(code).zfill(6).startswith(("5", "6", "9")) else "sz"


def yyyymmdd_to_iso(value: str) -> str:
    value = str(value)
    if "-" in value:
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def iso_to_yyyymmdd(value: str | dt.date) -> str:
    if isinstance(value, dt.date):
        return value.strftime("%Y%m%d")
    value = str(value)
    if "-" not in value:
        return value
    return value.replace("-", "")[:8]


def _date_from_any(value: str | dt.date) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(yyyymmdd_to_iso(str(value))[:10])


def _add_years(value: dt.date, years: int) -> dt.date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        # 2 月 29 日跨到平年时回退到 2 月 28 日。
        return value.replace(year=value.year + years, day=28)


def _calendar_chunks(start: str, end: str, years: int = 4) -> list[tuple[str, str]]:
    """Split a long date range into chunks accepted by fallback K-line APIs."""

    start_date = _date_from_any(start)
    end_date = _date_from_any(end)
    chunks: list[tuple[str, str]] = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(_add_years(cursor, years) - dt.timedelta(days=1), end_date)
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + dt.timedelta(days=1)
    return chunks


def fetch_history_tencent(code: str, start: str, end: str, adjust: str = "qfq") -> list[dict[str, Any]]:
    """Fetch ETF daily K-lines from Tencent as a fallback.

    Tencent returns date/open/close/high/low/volume.  Amount and turnover are
    unavailable, so downstream feature imputation handles them.
    """

    code = str(code).zfill(6)
    symbol = market_prefix(code) + code
    fq = "qfq" if adjust.lower() == "qfq" else ""
    # Tencent rejects very large counts (e.g. 3000) for some ETF symbols.  For
    # long-history research we query four-calendar-year chunks and de-duplicate
    # the result, which keeps every request comfortably below 2000 trading days.
    raw_by_date: dict[str, dict[str, Any]] = {}
    for chunk_start, chunk_end in _calendar_chunks(start, end, years=4):
        param = f"{symbol},day,{chunk_start},{chunk_end},2000,{fq}".rstrip(",")
        payload = http_json(TENCENT_KLINE_URL, {"param": param}, timeout=10, attempts=3)
        node = (payload.get("data") or {}).get(symbol) or {}
        lines = node.get("qfqday") or node.get("hfqday") or node.get("day") or []
        for item in lines:
            if len(item) < 6:
                continue
            date_s, open_s, close_s, high_s, low_s, volume_s = item[:6]
            raw_by_date[str(date_s)[:10]] = {
                "code": code,
                "date": str(date_s)[:10],
                "open": to_float(open_s),
                "close": to_float(close_s),
                "high": to_float(high_s),
                "low": to_float(low_s),
                "volume": to_float(volume_s),
                "amount": None,
                "turnover_rate": None,
            }
        time.sleep(0.03)
    if not raw_by_date:
        raise RuntimeError(f"No Tencent K-line rows for {code}")
    rows: list[dict[str, Any]] = []
    prev_close: float | None = None
    for item in sorted(raw_by_date.values(), key=lambda r: str(r["date"])):
        close_v = to_float(item.get("close"))
        high_v = to_float(item.get("high"))
        low_v = to_float(item.get("low"))
        price_change = close_v - prev_close if close_v is not None and prev_close is not None else None
        pct_chg = price_change / prev_close * 100 if price_change is not None and prev_close not in (None, 0) else None
        amplitude = (high_v - low_v) / prev_close * 100 if high_v is not None and low_v is not None and prev_close not in (None, 0) else None
        row = dict(item)
        row["amplitude"] = amplitude
        row["pct_chg"] = pct_chg
        row["price_change"] = price_change
        rows.append(row)
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
                "lmt": 1000000,
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


def predict_dataset(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach model predictions to arbitrary panel rows."""

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["prediction"] = predict_one(model, row)
        out_rows.append(out)
    return out_rows


def split_datasets(panel: list[dict[str, Any]], train: list[dict[str, Any]], test: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Split the panel into historical and future/inference datasets.

    historical_dataset: rows with a known forward-return label.
    future_dataset: rows whose future label is not yet available; these are
    inference-only rows and get a model prediction.
    """

    historical = []
    future = []
    for row in panel:
        item = dict(row)
        if row.get("is_trainable") and to_float(row.get(model["target_col"])) is not None:
            item["dataset_split"] = "historical"
            historical.append(item)
        else:
            item["dataset_split"] = "future_unlabeled"
            future.append(item)

    train_rows = [dict(row, dataset_split="train") for row in train]
    test_rows = [dict(row, dataset_split="test") for row in test]
    future_predicted = predict_dataset(future, model)
    return {
        "historical": historical,
        "train": train_rows,
        "test": test_rows,
        "future": future_predicted,
    }


def median(values: list[float]) -> float | None:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return None
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def stdev_population(values: list[float]) -> float | None:
    values = [v for v in values if math.isfinite(v)]
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def max_drawdown_from_curve(values: list[float]) -> float | None:
    if not values:
        return None
    peak = values[0]
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, value / peak - 1.0)
    return max_dd


def _valid_price_history(hist: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    rows = sorted(hist, key=lambda r: str(r.get("date", "")))
    dates: list[str] = []
    closes: list[float] = []
    for row in rows:
        close = to_float(row.get("close"))
        if close is None or close <= 0:
            continue
        dates.append(str(row.get("date", ""))[:10])
        closes.append(close)
    return dates, closes


def _summarize_returns(values: list[float], horizon: int) -> dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    observations = len(vals)
    avg_value = sum(vals) / observations if observations else None
    vol_value = stdev_population(vals) if observations >= 2 else None
    return {
        "horizon_days": horizon,
        "observations": observations,
        "win_rate": sum(1 for value in vals if value > 0) / observations if observations else None,
        "avg_forward_return": avg_value,
        "median_forward_return": median(vals),
        "volatility": vol_value,
        "sharpe_like": (avg_value / vol_value * math.sqrt(252 / max(1, horizon))) if avg_value is not None and vol_value not in (None, 0) else None,
        "best_forward_return": max(vals) if vals else None,
        "worst_forward_return": min(vals) if vals else None,
        "positive_avg_return": mean([value for value in vals if value > 0]),
        "negative_avg_return": mean([value for value in vals if value <= 0]),
    }


def build_multi_scale_analysis(
    histories: dict[str, list[dict[str, Any]]],
    universe: list[dict[str, Any]],
    horizons: list[int],
) -> dict[str, list[dict[str, Any]]]:
    """Summarize long-history ETF behavior across holding horizons.

    For each ETF and horizon we compute all overlapping forward returns, then
    aggregate them by ETF, index family and calendar year.  These descriptive
    statistics are deliberately independent from the ML train/test split so the
    report can show the longer historical regime coverage.
    """

    horizons = sorted({int(h) for h in horizons if int(h) > 0})
    meta_by_code = {str(row.get("code", "")).zfill(6): row for row in universe}
    coverage_rows: list[dict[str, Any]] = []
    etf_rows: list[dict[str, Any]] = []
    family_groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    family_etfs: dict[tuple[str, int], set[str]] = defaultdict(set)
    period_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    period_dates: dict[tuple[str, str, int], list[str]] = defaultdict(list)

    for code, hist in sorted(histories.items()):
        code = str(code).zfill(6)
        meta = meta_by_code.get(code, {"code": code, "name": "", "family_id": ""})
        dates, closes = _valid_price_history(hist)
        if not dates:
            continue
        start_date = _date_from_any(dates[0])
        end_date = _date_from_any(dates[-1])
        history_years = (end_date - start_date).days / 365.25 if end_date >= start_date else 0.0
        full_return = closes[-1] / closes[0] - 1.0 if closes[0] else None
        coverage_rows.append(
            {
                "code": code,
                "name": meta.get("name", ""),
                "family_id": meta.get("family_id", ""),
                "start_date": dates[0],
                "end_date": dates[-1],
                "rows": len(closes),
                "history_years": history_years,
                "full_period_return": full_return,
                "price_max_drawdown": max_drawdown_from_curve(closes),
                "latest_close": closes[-1],
            }
        )
        for horizon in horizons:
            forward_values: list[float] = []
            for i in range(0, len(closes) - horizon):
                start_close = closes[i]
                end_close = closes[i + horizon]
                if start_close <= 0 or not math.isfinite(start_close) or not math.isfinite(end_close):
                    continue
                ret = end_close / start_close - 1.0
                if not math.isfinite(ret):
                    continue
                date_s = dates[i]
                year = date_s[:4]
                family_id = str(meta.get("family_id", ""))
                forward_values.append(ret)
                for aggregate_family in (family_id, "ALL"):
                    family_groups[(aggregate_family, horizon)].append(ret)
                    family_etfs[(aggregate_family, horizon)].add(code)
                    period_groups[(year, aggregate_family, horizon)].append(ret)
                    period_dates[(year, aggregate_family, horizon)].append(date_s)
            summary = _summarize_returns(forward_values, horizon)
            trailing_return = closes[-1] / closes[-1 - horizon] - 1.0 if len(closes) > horizon and closes[-1 - horizon] else None
            etf_rows.append(
                {
                    "code": code,
                    "name": meta.get("name", ""),
                    "family_id": meta.get("family_id", ""),
                    "start_date": dates[0],
                    "end_date": dates[-1],
                    "history_rows": len(closes),
                    "history_years": history_years,
                    "trailing_horizon_return": trailing_return,
                    "price_max_drawdown": max_drawdown_from_curve(closes),
                    **summary,
                }
            )

    family_rows: list[dict[str, Any]] = []
    for (family_id, horizon), values in sorted(family_groups.items(), key=lambda item: (item[0][0] != "ALL", item[0][0], item[0][1])):
        summary = _summarize_returns(values, horizon)
        family_rows.append(
            {
                "family_id": family_id,
                "horizon_days": horizon,
                "etf_count": len(family_etfs[(family_id, horizon)]),
                **summary,
            }
        )

    period_rows: list[dict[str, Any]] = []
    for (year, family_id, horizon), values in sorted(period_groups.items()):
        summary = _summarize_returns(values, horizon)
        dates_for_period = period_dates[(year, family_id, horizon)]
        period_rows.append(
            {
                "period": year,
                "family_id": family_id,
                "horizon_days": horizon,
                "period_start_date": min(dates_for_period) if dates_for_period else "",
                "period_end_date": max(dates_for_period) if dates_for_period else "",
                **summary,
            }
        )

    coverage_rows.sort(key=lambda r: (float(r.get("history_years") or 0.0), int(r.get("rows") or 0)), reverse=True)
    etf_rows.sort(key=lambda r: (str(r.get("family_id")), int(r.get("horizon_days") or 0), str(r.get("code"))))
    return {
        "coverage": coverage_rows,
        "etf_metrics": etf_rows,
        "family_metrics": family_rows,
        "period_metrics": period_rows,
    }


def build_buy_signals(preds: list[dict[str, Any]], horizon: int, top_k: int, min_pred: float) -> list[dict[str, Any]]:
    """Create current research buy signals from latest predictions."""

    pred_col = f"pred_fwd_ret_{horizon}"
    eligible = [row for row in preds if to_float(row.get(pred_col)) is not None and float(row[pred_col]) >= min_pred]
    selected = sorted(eligible, key=lambda r: int(r.get("pred_score_rank", 9999)))[: max(0, top_k)]
    if not selected:
        return [
            {
                "date": preds[0].get("date") if preds else "",
                "action": "CASH",
                "allocation_pct": 100.0,
                "reason": f"无 ETF 同时满足 rank<= {top_k} 与 prediction >= {min_pred:.4f}",
            }
        ]
    allocation = 100.0 / len(selected)
    signals = []
    for row in selected:
        signals.append(
            {
                "date": row.get("date"),
                "action": "BUY",
                "allocation_pct": allocation,
                "rank": row.get("pred_score_rank"),
                "code": row.get("code"),
                "name": row.get("name"),
                "family_id": row.get("family_id"),
                "close": row.get("close"),
                "predicted_forward_return": row.get(pred_col),
                "holding_horizon_days": horizon,
                "reason": f"预测排序 Top{top_k} 且预测收益 >= {min_pred:.2%}",
            }
        )
    return signals


def backtest_purchase_strategy(
    holdout: list[dict[str, Any]],
    target_col: str,
    *,
    top_k: int,
    min_pred: float,
    round_trip_cost_bps: float,
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Backtest a simple equal-weight top-K purchase strategy on holdout rows.

    Rule:
    - On each decision date, rank ETFs by model prediction.
    - Buy equal-weight top-K ETFs whose prediction is at least min_pred.
    - Hold for the labelled forward horizon and subtract round-trip cost.

    The return series is a daily rolling-decision evaluation; because forward
    windows overlap, it is a research backtest signal, not broker-executable
    accounting.
    """

    cost = float(round_trip_cost_bps) / 10000.0
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in holdout:
        if to_float(row.get("prediction")) is None or to_float(row.get(target_col)) is None:
            continue
        grouped[str(row["date"])].append(row)

    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    strategy_value = 1.0
    benchmark_value = 1.0
    strategy_curve = [strategy_value]
    benchmark_curve = [benchmark_value]
    all_trade_returns: list[float] = []
    active_decision_returns: list[float] = []
    all_decision_returns: list[float] = []
    benchmark_returns: list[float] = []
    outperform_flags: list[bool] = []

    for date_s in sorted(grouped):
        group = sorted(grouped[date_s], key=lambda r: float(r["prediction"]), reverse=True)
        benchmark_ret = sum(float(r[target_col]) for r in group) / len(group)
        selected = [row for row in group if float(row["prediction"]) >= min_pred][: max(0, top_k)]
        selected_returns = [float(row[target_col]) - cost for row in selected]
        strategy_ret = sum(selected_returns) / len(selected_returns) if selected_returns else 0.0
        strategy_value *= 1.0 + strategy_ret
        benchmark_value *= 1.0 + benchmark_ret
        strategy_curve.append(strategy_value)
        benchmark_curve.append(benchmark_value)
        all_decision_returns.append(strategy_ret)
        benchmark_returns.append(benchmark_ret)
        outperform_flags.append(strategy_ret > benchmark_ret)
        if selected_returns:
            active_decision_returns.append(strategy_ret)
            all_trade_returns.extend(selected_returns)

        wins = sum(1 for value in selected_returns if value > 0)
        daily_rows.append(
            {
                "date": date_s,
                "strategy": f"top{top_k}_minpred_{min_pred:.4f}",
                "selected_count": len(selected),
                "selected_codes": ",".join(str(r.get("code", "")) for r in selected),
                "selected_names": ",".join(str(r.get("name", "")) for r in selected),
                "avg_prediction": mean([float(r["prediction"]) for r in selected]) if selected else None,
                "strategy_return": strategy_ret,
                "benchmark_equal_weight_return": benchmark_ret,
                "excess_return": strategy_ret - benchmark_ret,
                "trade_win_count": wins,
                "trade_count": len(selected),
                "trade_win_rate": wins / len(selected) if selected else None,
                "strategy_cumulative_return": strategy_value - 1.0,
                "benchmark_cumulative_return": benchmark_value - 1.0,
            }
        )
        for rank, row in enumerate(selected, 1):
            raw_ret = float(row[target_col])
            net_ret = raw_ret - cost
            trade_rows.append(
                {
                    "date": date_s,
                    "action": "BUY",
                    "rank": rank,
                    "code": row.get("code"),
                    "name": row.get("name"),
                    "family_id": row.get("family_id"),
                    "close": row.get("close"),
                    "prediction": row.get("prediction"),
                    "actual_forward_return": raw_ret,
                    "round_trip_cost_bps": round_trip_cost_bps,
                    "net_forward_return": net_ret,
                    "win": net_ret > 0,
                    "holding_horizon_days": horizon,
                }
            )

    active_count = len(active_decision_returns)
    decision_count = len(all_decision_returns)
    trade_count = len(all_trade_returns)
    trade_win_rate = sum(1 for value in all_trade_returns if value > 0) / trade_count if trade_count else None
    decision_win_rate = sum(1 for value in active_decision_returns if value > 0) / active_count if active_count else None
    benchmark_win_rate = sum(1 for value in benchmark_returns if value > 0) / len(benchmark_returns) if benchmark_returns else None
    daily_std = stdev_population(active_decision_returns)
    sharpe_like = None
    if daily_std and daily_std > 0 and active_decision_returns:
        sharpe_like = (sum(active_decision_returns) / len(active_decision_returns)) / daily_std * math.sqrt(252 / max(1, horizon))
    metrics = {
        "strategy_name": f"Top{top_k} equal-weight, min_pred={min_pred:.4f}, cost={round_trip_cost_bps:.1f}bps",
        "strategy_rule": f"每个决策日按预测收益排序，买入预测值 >= {min_pred:.2%} 的前 {top_k} 只 ETF，等权持有 {horizon} 个交易日。",
        "strategy_top_k": top_k,
        "strategy_min_prediction": min_pred,
        "round_trip_cost_bps": round_trip_cost_bps,
        "decision_count": decision_count,
        "active_decision_count": active_count,
        "no_trade_count": decision_count - active_count,
        "trade_count": trade_count,
        "trade_win_rate": trade_win_rate,
        "decision_win_rate": decision_win_rate,
        "benchmark_win_rate": benchmark_win_rate,
        "benchmark_outperform_rate": sum(1 for flag in outperform_flags if flag) / len(outperform_flags) if outperform_flags else None,
        "avg_trade_return": sum(all_trade_returns) / trade_count if trade_count else None,
        "median_trade_return": median(all_trade_returns),
        "avg_active_decision_return": sum(active_decision_returns) / active_count if active_count else None,
        "avg_all_decision_return": sum(all_decision_returns) / decision_count if decision_count else None,
        "avg_benchmark_return": sum(benchmark_returns) / len(benchmark_returns) if benchmark_returns else None,
        "strategy_cumulative_return": strategy_value - 1.0,
        "benchmark_cumulative_return": benchmark_value - 1.0,
        "excess_cumulative_return": (strategy_value - benchmark_value),
        "strategy_max_drawdown": max_drawdown_from_curve(strategy_curve),
        "benchmark_max_drawdown": max_drawdown_from_curve(benchmark_curve),
        "sharpe_like": sharpe_like,
        "backtest_note": "收益为每日滚动 horizon 前瞻收益评估，窗口存在重叠；用于研究排序信号，不等同真实账户流水。",
    }
    return metrics, daily_rows, trade_rows


def model_tune_score(metrics: dict[str, Any]) -> float:
    directional = to_float(metrics.get("directional_accuracy"), 0.0) or 0.0
    ic = to_float(metrics.get("spearman_ic_by_date"), 0.0) or 0.0
    top_vs_all = to_float(metrics.get("top_vs_all"), 0.0) or 0.0
    rmse = to_float(metrics.get("rmse"), 0.0) or 0.0
    return directional + 0.8 * ic + 4.0 * top_vs_all - 0.5 * rmse


def strategy_tune_score(metrics: dict[str, Any]) -> float:
    cumulative = to_float(metrics.get("strategy_cumulative_return"), 0.0) or 0.0
    win_rate = to_float(metrics.get("trade_win_rate"), 0.0) or 0.0
    outperform = to_float(metrics.get("benchmark_outperform_rate"), 0.0) or 0.0
    drawdown = to_float(metrics.get("strategy_max_drawdown"), 0.0) or 0.0
    sharpe_like = to_float(metrics.get("sharpe_like"), 0.0) or 0.0
    # Drawdown is negative, so this penalizes deeper drawdowns.
    return cumulative + 0.25 * win_rate + 0.25 * outperform + 0.05 * sharpe_like + 0.25 * drawdown


def tune_model_and_strategy(
    train_rows: list[dict[str, Any]],
    feature_cols: list[str],
    target_col: str,
    *,
    l2_grid: list[float],
    top_k_grid: list[int],
    min_pred_grid: list[float],
    tune_days: int,
    round_trip_cost_bps: float,
    horizon: int,
) -> tuple[float, int, float, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Tune ridge L2 and purchase-strategy parameters on an inner validation split."""

    inner_train, validation, validation_split_date = time_split(train_rows, target_col, max(20, int(tune_days)))
    model_rows: list[dict[str, Any]] = []
    best_l2 = l2_grid[0] if l2_grid else 3.0
    best_model_score = -1e18
    best_validation_preds: list[dict[str, Any]] = []
    for l2 in l2_grid or [3.0]:
        model = train_ridge(inner_train, feature_cols, target_col, l2=l2)
        metrics, validation_preds = evaluate(validation, model, target_col)
        score = model_tune_score(metrics)
        row = {
            "tune_type": "model_l2",
            "l2": l2,
            "validation_split_date": validation_split_date,
            "score": score,
            "directional_accuracy": metrics.get("directional_accuracy"),
            "spearman_ic_by_date": metrics.get("spearman_ic_by_date"),
            "top_vs_all": metrics.get("top_vs_all"),
            "rmse": metrics.get("rmse"),
            "rows": metrics.get("rows"),
        }
        model_rows.append(row)
        if score > best_model_score:
            best_model_score = score
            best_l2 = l2
            best_validation_preds = validation_preds

    strategy_rows: list[dict[str, Any]] = []
    best_top_k = top_k_grid[0] if top_k_grid else 3
    best_min_pred = min_pred_grid[0] if min_pred_grid else 0.0
    best_strategy_score = -1e18
    for top_k in top_k_grid or [3]:
        for min_pred in min_pred_grid or [0.0]:
            strategy_metrics, _, _ = backtest_purchase_strategy(
                best_validation_preds,
                target_col,
                top_k=top_k,
                min_pred=min_pred,
                round_trip_cost_bps=round_trip_cost_bps,
                horizon=horizon,
            )
            score = strategy_tune_score(strategy_metrics)
            row = {
                "tune_type": "strategy",
                "top_k": top_k,
                "min_pred": min_pred,
                "score": score,
                "trade_win_rate": strategy_metrics.get("trade_win_rate"),
                "decision_win_rate": strategy_metrics.get("decision_win_rate"),
                "benchmark_outperform_rate": strategy_metrics.get("benchmark_outperform_rate"),
                "strategy_cumulative_return": strategy_metrics.get("strategy_cumulative_return"),
                "strategy_max_drawdown": strategy_metrics.get("strategy_max_drawdown"),
                "sharpe_like": strategy_metrics.get("sharpe_like"),
                "trade_count": strategy_metrics.get("trade_count"),
                "active_decision_count": strategy_metrics.get("active_decision_count"),
            }
            strategy_rows.append(row)
            if score > best_strategy_score:
                best_strategy_score = score
                best_top_k = top_k
                best_min_pred = min_pred

    best = {
        "auto_tune": True,
        "validation_split_date": validation_split_date,
        "inner_train_rows": len(inner_train),
        "validation_rows": len(validation),
        "best_l2": best_l2,
        "best_l2_score": best_model_score,
        "best_strategy_top_k": best_top_k,
        "best_strategy_min_pred": best_min_pred,
        "best_strategy_score": best_strategy_score,
        "model_objective": "directional_accuracy + 0.8*IC + 4*top_vs_all - 0.5*RMSE",
        "strategy_objective": "cum_return + 0.25*win_rate + 0.25*outperform + 0.05*sharpe + 0.25*max_drawdown",
    }
    return best_l2, best_top_k, best_min_pred, best, model_rows, strategy_rows


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
    return {"Top1 策略": top_points, "全池平均": all_points, "Bottom1": bottom_points}


def _backtest_cumulative_series(backtest_rows: list[dict[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    strategy = []
    benchmark = []
    excess = []
    for row in backtest_rows:
        date_s = str(row.get("date", ""))
        strategy_value = to_float(row.get("strategy_cumulative_return"))
        benchmark_value = to_float(row.get("benchmark_cumulative_return"))
        if strategy_value is not None:
            strategy.append((date_s, strategy_value))
        if benchmark_value is not None:
            benchmark.append((date_s, benchmark_value))
        if strategy_value is not None and benchmark_value is not None:
            excess.append((date_s, strategy_value - benchmark_value))
    return {"购买策略": strategy, "全池等权基准": benchmark, "累计超额": excess}


def _multi_scale_period_series(period_rows: list[dict[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    series: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in period_rows:
        if str(row.get("family_id")) != "ALL":
            continue
        value = to_float(row.get("avg_forward_return"))
        if value is None:
            continue
        horizon = int(to_float(row.get("horizon_days"), 0) or 0)
        if horizon <= 0:
            continue
        series[f"{horizon}日"].append((str(row.get("period")), value))
    return {name: sorted(points) for name, points in sorted(series.items(), key=lambda item: int(item[0].replace("日", "")))}


def generate_charts(
    universe: list[dict[str, Any]],
    metrics: dict[str, Any],
    preds: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    target_col: str,
    horizon: int,
    backtest_metrics: dict[str, Any] | None = None,
    backtest_rows: list[dict[str, Any]] | None = None,
    tuning_model_rows: list[dict[str, Any]] | None = None,
    tuning_strategy_rows: list[dict[str, Any]] | None = None,
    related_stocks: list[dict[str, Any]] | None = None,
    multi_scale_family_rows: list[dict[str, Any]] | None = None,
    multi_scale_period_rows: list[dict[str, Any]] | None = None,
    history_coverage_rows: list[dict[str, Any]] | None = None,
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
        ("交易胜率", (backtest_metrics or {}).get("trade_win_rate")),
        ("调仓胜率", (backtest_metrics or {}).get("decision_win_rate")),
        ("跑赢基准率", (backtest_metrics or {}).get("benchmark_outperform_rate")),
        ("策略累计", (backtest_metrics or {}).get("strategy_cumulative_return")),
        ("Top1 均值", metrics.get("top1_mean_fwd_ret")),
        ("Top1-全池", metrics.get("top_vs_all")),
        ("Top-Bottom", metrics.get("top_bottom_spread")),
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
    strategy_path = write_line_chart(
        charts_dir / "strategy_backtest_cumulative_lite.svg",
        title="购买策略回测累计曲线",
        subtitle="TopK 等权买入策略 vs 全池等权基准；曲线为滚动决策累计收益。",
        series=_backtest_cumulative_series(backtest_rows or []),
        value_kind="pct",
    )
    chart_paths = {
        "latest_prediction_rank": str(prediction_path),
        "candidate_family_counts": str(family_path),
        "holdout_metric_snapshot": str(metric_path),
        "holdout_cumulative": str(cumulative_path),
        "strategy_backtest_cumulative": str(strategy_path),
    }
    if tuning_model_rows:
        tune_path = write_column_chart(
            charts_dir / "model_l2_tuning_lite.svg",
            title="自动调参：Ridge L2 验证分数",
            subtitle="越高越好；用于选择最终岭回归正则强度。",
            rows=[(f"L2={row.get('l2')}", float(row.get("score", 0.0))) for row in tuning_model_rows],
            value_kind="number",
            width=1120,
        )
        chart_paths["model_l2_tuning"] = str(tune_path)
    if tuning_strategy_rows:
        top_rows = sorted(tuning_strategy_rows, key=lambda r: float(r.get("score") or -1e18), reverse=True)[:12]
        strategy_tune_path = write_horizontal_bar_chart(
            charts_dir / "strategy_parameter_tuning_lite.svg",
            title="自动调参：购买策略参数 Top 组合",
            subtitle="按验证期综合目标排序，标签为 TopK / 最低预测收益。",
            rows=[(f"K={r.get('top_k')} min={float(r.get('min_pred') or 0.0):.2%}", float(r.get("score") or 0.0)) for r in top_rows],
            value_kind="number",
        )
        chart_paths["strategy_parameter_tuning"] = str(strategy_tune_path)
    if related_stocks:
        top_related = sorted(
            related_stocks,
            key=lambda r: (
                to_float(r.get("weight"), -1.0) if r.get("rank_basis") == "weight" else to_float(r.get("free_cap"), -1.0),
                to_float(r.get("free_cap"), -1.0),
            ),
            reverse=True,
        )[:18]
        related_path = write_horizontal_bar_chart(
            charts_dir / "related_stock_top_exposure_lite.svg",
            title="相关股票：宽基指数高权重/高市值成分",
            subtitle="有权重的指数按权重排序；无权重数据的指数按自由流通市值排序。",
            rows=[
                (
                    f"{row.get('family_id')} {row.get('stock_code')} {row.get('stock_name')}",
                    float(to_float(row.get("weight"), None) or (to_float(row.get("free_cap"), 0.0) or 0.0) / 10000.0),
                )
                for row in top_related
            ],
            value_kind="number",
        )
        chart_paths["related_stock_top_exposure"] = str(related_path)
    if multi_scale_family_rows:
        all_family_rows = [
            row for row in multi_scale_family_rows
            if str(row.get("family_id")) == "ALL" and to_float(row.get("avg_forward_return")) is not None
        ]
        all_family_rows.sort(key=lambda r: int(to_float(r.get("horizon_days"), 0) or 0))
        if all_family_rows:
            return_path = write_column_chart(
                charts_dir / "multi_scale_horizon_return_lite.svg",
                title="跨时间尺度：全池平均前瞻收益",
                subtitle="按不同持有交易日 horizon 聚合全部候选 ETF 的历史前瞻收益。",
                rows=[(f"{int(row.get('horizon_days'))}日", float(row.get("avg_forward_return") or 0.0)) for row in all_family_rows],
                value_kind="pct",
                width=1040,
            )
            win_path = write_column_chart(
                charts_dir / "multi_scale_win_rate_lite.svg",
                title="跨时间尺度：全池历史胜率",
                subtitle="胜率=对应 horizon 前瞻收益大于 0 的样本占比。",
                rows=[(f"{int(row.get('horizon_days'))}日", float(row.get("win_rate") or 0.0)) for row in all_family_rows],
                value_kind="pct",
                width=1040,
            )
            chart_paths["multi_scale_horizon_return"] = str(return_path)
            chart_paths["multi_scale_win_rate"] = str(win_path)
    if history_coverage_rows:
        top_coverage = sorted(history_coverage_rows, key=lambda r: float(r.get("history_years") or 0.0), reverse=True)[:24]
        coverage_path = write_horizontal_bar_chart(
            charts_dir / "long_history_coverage_lite.svg",
            title="长期历史数据覆盖年限",
            subtitle="从本次长起点抓取后，各 ETF 可用日线历史的覆盖年数（受上市日期限制）。",
            rows=[
                (
                    f"{row.get('code')} {row.get('name')}",
                    float(row.get("history_years") or 0.0),
                )
                for row in top_coverage
            ],
            value_kind="number",
        )
        chart_paths["long_history_coverage"] = str(coverage_path)
    if multi_scale_period_rows:
        period_path = write_line_chart(
            charts_dir / "multi_scale_period_return_lite.svg",
            title="跨年份/跨周期平均前瞻收益",
            subtitle="按日历年聚合全池 ETF 前瞻收益，展示不同 horizon 在市场阶段中的变化。",
            series=_multi_scale_period_series(multi_scale_period_rows),
            value_kind="pct",
        )
        chart_paths["multi_scale_period_return"] = str(period_path)
    return chart_paths


def write_summary(
    universe: list[dict[str, Any]],
    metrics: dict[str, Any],
    preds: list[dict[str, Any]],
    feature_cols: list[str],
    horizon: int,
    chart_paths: dict[str, str] | None = None,
    backtest_metrics: dict[str, Any] | None = None,
    buy_signals: list[dict[str, Any]] | None = None,
    dataset_paths: dict[str, str] | None = None,
    tuning_best: dict[str, Any] | None = None,
    tuning_model_rows: list[dict[str, Any]] | None = None,
    tuning_strategy_rows: list[dict[str, Any]] | None = None,
    related_stocks: list[dict[str, Any]] | None = None,
    multi_scale_family_rows: list[dict[str, Any]] | None = None,
    history_coverage_rows: list[dict[str, Any]] | None = None,
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
        "|代码 | 名称 | 指数族 | 最新价 | 成交额 | 族内排名|",
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
            "|指标 | 值|",
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
            "## 3. 长历史与跨时间尺度分析",
            "",
        ]
    )
    all_scale_rows = [
        row for row in (multi_scale_family_rows or [])
        if str(row.get("family_id")) == "ALL" and to_float(row.get("avg_forward_return")) is not None
    ]
    all_scale_rows.sort(key=lambda r: int(to_float(r.get("horizon_days"), 0) or 0))
    if all_scale_rows:
        lines.append(
            f"- 本次长历史请求区间：{metrics.get('requested_start')} 至 {metrics.get('requested_end')}；"
            f"实际最早 ETF 日线：{metrics.get('long_history_start')}，最新：{metrics.get('long_history_end')}。"
        )
        lines.append(f"- 可用历史覆盖：最长 {to_float(metrics.get('long_history_max_years'), 0.0):.2f} 年；跨周期 horizon：{', '.join(str(h) for h in metrics.get('scale_horizons', []))} 个交易日。")
        lines.append("- 统计口径：对每个 ETF 的所有重叠前瞻收益做描述统计；用于观察不同持有周期的历史胜率、波动和阶段变化。")
        lines.append("")
        lines.append("|持有周期|ETF 数 | 样本数 | 平均前瞻收益 | 中位数 | 胜率 | 波动 | 类 Sharpe|最佳 | 最差|")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in all_scale_rows:
            lines.append(
                f"|{row.get('horizon_days')}日|{row.get('etf_count')}|{row.get('observations')}|"
                f"{fmt_pct(row.get('avg_forward_return'))}|{fmt_pct(row.get('median_forward_return'))}|"
                f"{fmt_pct(row.get('win_rate'))}|{fmt_pct(row.get('volatility'))}|"
                f"{to_float(row.get('sharpe_like'), 0.0):.4f}|{fmt_pct(row.get('best_forward_return'))}|{fmt_pct(row.get('worst_forward_return'))}|"
            )
        lines.append("")
        lines.append("### 历史覆盖最长的 ETF")
        lines.append("")
        lines.append("|代码 | 名称 | 指数族 | 起始 | 截止 | 日线数 | 年限 | 全期收益 | 最大回撤|")
        lines.append("|---|---|---|---|---|---:|---:|---:|---:|")
        for row in (history_coverage_rows or [])[:12]:
            lines.append(
                f"|{row.get('code')}|{row.get('name')}|{row.get('family_id')}|{row.get('start_date')}|{row.get('end_date')}|"
                f"{row.get('rows')}|{to_float(row.get('history_years'), 0.0):.2f}|{fmt_pct(row.get('full_period_return'))}|{fmt_pct(row.get('price_max_drawdown'))}|"
            )
        lines.append("")
        lines.append("### 多尺度输出文件")
        lines.append("")
        lines.append("|文件 | 说明|")
        lines.append("|---|---|")
        output_labels = {
            "history_coverage_path": "每只 ETF 的历史覆盖、全期收益和价格最大回撤",
            "multi_scale_etf_metrics_path": "ETF × horizon 的收益/胜率/波动/回撤统计",
            "multi_scale_family_metrics_path": "指数族/全池 × horizon 聚合统计",
            "multi_scale_period_metrics_path": "年份 × 指数族 × horizon 的阶段统计",
        }
        for key, label in output_labels.items():
            path_text = metrics.get(key)
            if path_text:
                lines.append(f"|`{path_text}`|{label}|")
    else:
        lines.append("- 本次未生成跨时间尺度统计。")
    lines.extend(
        [
            "",
            "## 4. 自动参数调优",
            "",
        ]
    )
    if tuning_best:
        lines.append("- 调优方式：在训练集内部再切出靠后的验证期，先调 Ridge L2，再调购买策略 TopK / 最低预测收益阈值。")
        lines.append(f"- 模型目标：{tuning_best.get('model_objective')}")
        lines.append(f"- 策略目标：{tuning_best.get('strategy_objective')}")
        lines.append(f"- 最佳 L2：{tuning_best.get('best_l2')}；最佳策略：Top{tuning_best.get('best_strategy_top_k')}，最低预测收益 {fmt_pct(tuning_best.get('best_strategy_min_pred'))}。")
        lines.append("")
        lines.append("### L2 调优结果")
        lines.append("")
        lines.append("|L2|验证分数 | 方向准确率|IC|Top1-全池|RMSE|")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in (tuning_model_rows or [])[:20]:
            lines.append(
                f"|{row.get('l2')}|{to_float(row.get('score'), 0.0):.6f}|{fmt_pct(row.get('directional_accuracy'))}|"
                f"{to_float(row.get('spearman_ic_by_date'), 0.0):.4f}|{fmt_pct(row.get('top_vs_all'))}|{to_float(row.get('rmse'), 0.0):.6f}|"
            )
        lines.append("")
        lines.append("### 策略参数 Top 结果")
        lines.append("")
        lines.append("|TopK|最低预测 | 调优分数 | 交易胜率 | 跑赢基准率 | 累计收益 | 最大回撤|")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in sorted(tuning_strategy_rows or [], key=lambda r: float(r.get("score") or -1e18), reverse=True)[:12]:
            lines.append(
                f"|{row.get('top_k')}|{fmt_pct(row.get('min_pred'))}|{to_float(row.get('score'), 0.0):.6f}|"
                f"{fmt_pct(row.get('trade_win_rate'))}|{fmt_pct(row.get('benchmark_outperform_rate'))}|"
                f"{fmt_pct(row.get('strategy_cumulative_return'))}|{fmt_pct(row.get('strategy_max_drawdown'))}|"
            )
    else:
        lines.append("- 本次未启用自动调参，使用命令行传入的 L2 与策略参数。")
    lines.extend(
        [
            "",
            "## 5. 相关股票/成分股扩展",
            "",
        ]
    )
    if related_stocks:
        family_counts: dict[str, int] = defaultdict(int)
        for row in related_stocks:
            family_counts[str(row.get("family_id"))] += 1
        lines.append(f"- 已拉取相关股票/指数成分股：{len(related_stocks)} 条。")
        lines.append("- 覆盖：" + ", ".join(f"{k}({v})" for k, v in sorted(family_counts.items())))
        lines.append("- 数据源：东方财富指数成分股数据；有权重字段时按权重排序，否则按自由流通市值排序。")
        lines.append("")
        lines.append("|指数族 | 排名 | 股票代码 | 股票名称 | 行业 | 权重 | 自由流通市值 | 涨跌幅|PE|")
        lines.append("|---|---:|---|---|---|---:|---:|---:|---:|")
        for row in related_stocks[:30]:
            lines.append(
                f"|{row.get('family_id', '')}|{row.get('rank_in_family', '')}|{row.get('stock_code', '')}|{row.get('stock_name', '')}|"
                f"{row.get('industry', '')}|{fmt_pct((to_float(row.get('weight')) or 0.0) / 100 if to_float(row.get('weight')) is not None else None)}|"
                f"{to_float(row.get('free_cap'), 0.0):.2f}|{fmt_pct((to_float(row.get('change_rate')) or 0.0) / 100 if to_float(row.get('change_rate')) is not None else None)}|"
                f"{to_float(row.get('pe'), 0.0):.2f}|"
            )
    else:
        lines.append("- 本次未拉取相关股票。")
    lines.extend(
        [
            "",
            "## 6. 历史/未来数据集拆分",
            "",
            f"- 历史有标签数据：{metrics.get('historical_rows')} 行，可用于训练、验证和回测。",
            f"- 训练集：{metrics.get('train_rows')} 行；测试/回测集：{metrics.get('test_rows')} 行。",
            f"- 未来未标注数据：{metrics.get('future_rows')} 行，目标收益尚未发生，只用于推理和生成买入信号。",
            "",
            "|数据集 | 用途 | 文件|",
            "|---|---|---|",
        ]
    )
    if dataset_paths:
        dataset_labels = {
            "historical": ("历史有标签全集", "训练/验证/回测的监督学习样本"),
            "train": ("历史训练集", "拟合模型"),
            "test": ("历史测试/回测集", "时间切分验证与策略回测"),
            "future": ("未来未标注集", "未来收益未知，仅做推理"),
        }
        for key, (label, purpose) in dataset_labels.items():
            path_text = dataset_paths.get(key)
            if path_text:
                lines.append(f"|{label}|{purpose}|`{path_text}`|")
    lines.extend(
        [
            "",
            "## 7. 购买策略与回测",
            "",
        ]
    )
    if backtest_metrics:
        lines.append(f"- 策略规则：{backtest_metrics.get('strategy_rule')}")
        lines.append(f"- 成本假设：单次完整买卖往返成本 {backtest_metrics.get('round_trip_cost_bps')} bps。")
        lines.append("- 说明：回测使用测试期的前瞻收益标签；由于每日滚动 horizon 收益存在重叠，结果用于研究排序信号。")
        lines.append("")
        lines.append("### 当前买入信号")
        lines.append("")
        lines.append("|动作 | 日期 | 代码 | 名称 | 指数族 | 仓位 | 预测未来收益 | 理由|")
        lines.append("|---|---|---|---|---|---:|---:|---|")
        for row in (buy_signals or [])[:12]:
            lines.append(
                f"|{row.get('action', '')}|{row.get('date', '')}|{row.get('code', '')}|{row.get('name', '')}|"
                f"{row.get('family_id', '')}|{to_float(row.get('allocation_pct'), 0.0):.1f}%|"
                f"{fmt_pct(row.get('predicted_forward_return'))}|{row.get('reason', '')}|"
            )
        lines.append("")
        lines.append("### 回测指标")
        lines.append("")
        lines.append("|指标 | 值|")
        lines.append("|---|---:|")
        for key in [
            "decision_count",
            "active_decision_count",
            "trade_count",
            "trade_win_rate",
            "decision_win_rate",
            "benchmark_outperform_rate",
            "avg_trade_return",
            "median_trade_return",
            "strategy_cumulative_return",
            "benchmark_cumulative_return",
            "excess_cumulative_return",
            "strategy_max_drawdown",
            "sharpe_like",
        ]:
            value = backtest_metrics.get(key)
            if isinstance(value, float) and any(token in key for token in ["rate", "return", "drawdown"]):
                text = fmt_pct(value)
            elif isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = str(value)
            lines.append(f"|{key}|{text}|")
    else:
        lines.append("- 本次未生成策略回测。")
    lines.extend(
        [
            "",
            "## 8. 图表结果",
            "",
        ]
    )
    if chart_paths:
        chart_items = [
            ("最新预测排序图", chart_paths.get("latest_prediction_rank")),
            ("候选指数族覆盖图", chart_paths.get("candidate_family_counts")),
            ("验证指标快照图", chart_paths.get("holdout_metric_snapshot")),
            ("Holdout 累计曲线", chart_paths.get("holdout_cumulative")),
            ("购买策略回测累计曲线", chart_paths.get("strategy_backtest_cumulative")),
            ("模型参数调优图", chart_paths.get("model_l2_tuning")),
            ("策略参数调优图", chart_paths.get("strategy_parameter_tuning")),
            ("相关股票暴露图", chart_paths.get("related_stock_top_exposure")),
            ("跨周期平均收益图", chart_paths.get("multi_scale_horizon_return")),
            ("跨周期胜率图", chart_paths.get("multi_scale_win_rate")),
            ("长期历史覆盖图", chart_paths.get("long_history_coverage")),
            ("跨年份周期收益图", chart_paths.get("multi_scale_period_return")),
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
            "## 9. 最新预测排序",
            "",
            "|排名 | 日期 | 代码 | 名称 | 指数族 | 收盘价 | 预测未来收益 | 近 5 日 | 近 20 日|20 日波动|20 日回撤|",
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
    p.add_argument("--auto-tune", dest="auto_tune", action="store_true", default=True, help="Automatically tune model and strategy parameters on an inner validation split.")
    p.add_argument("--no-auto-tune", dest="auto_tune", action="store_false", help="Disable automatic parameter tuning.")
    p.add_argument("--tune-days", type=int, default=90, help="Validation-window length used for automatic tuning.")
    p.add_argument("--tune-l2-grid", default="0.3,1,3,10,30", help="Comma-separated Ridge L2 grid.")
    p.add_argument("--tune-top-k-grid", default="1,2,3,4,5", help="Comma-separated TopK grid for purchase-strategy tuning.")
    p.add_argument("--tune-min-pred-grid", default="-0.005,0,0.005,0.01,0.02", help="Comma-separated minimum prediction grid for strategy tuning.")
    p.add_argument("--strategy-top-k", type=int, default=3, help="Purchase-strategy top K ETFs to buy on each decision date.")
    p.add_argument("--strategy-min-pred", type=float, default=0.0, help="Minimum predicted forward return required for a BUY signal.")
    p.add_argument("--round-trip-cost-bps", type=float, default=10.0, help="Round-trip trading cost/slippage in basis points.")
    p.add_argument("--related-stocks-per-index", type=int, default=30, help="Fetch top N related/component stocks per broad-index family.")
    p.add_argument("--scale-horizons", default="5,20,60,120,250", help="Comma-separated holding horizons for long-history multi-scale analysis.")
    p.add_argument("--model-out", default=str(MODELS_DIR / "csi_broad_etf_model_lite.pkl"))
    return p


def main(argv: list[str] | None = None) -> None:
    ensure_dirs()
    args = build_parser().parse_args(argv)
    scale_horizons = [h for h in parse_int_grid(args.scale_horizons) if h > 0]
    if args.horizon not in scale_horizons:
        scale_horizons.append(args.horizon)
    scale_horizons = sorted(set(scale_horizons)) or [args.horizon]
    universe = discover_universe(args.max_etfs_per_index, args.min_amount, args.include_enhanced, args.include_style, args.force)
    write_csv_rows(REPORTS_DIR / "latest_candidates_lite.csv", universe)
    related_stocks: list[dict[str, Any]] = []
    if args.related_stocks_per_index > 0:
        related_stocks = fetch_related_stocks(universe, args.related_stocks_per_index, force=args.force)
        write_csv_rows(REPORTS_DIR / "related_stocks_lite.csv", related_stocks)
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
    multi_scale = build_multi_scale_analysis(histories, universe, scale_horizons)
    multi_scale_paths = {
        "history_coverage_path": str(REPORTS_DIR / "long_history_coverage_lite.csv"),
        "multi_scale_etf_metrics_path": str(REPORTS_DIR / "multi_scale_etf_metrics_lite.csv"),
        "multi_scale_family_metrics_path": str(REPORTS_DIR / "multi_scale_family_metrics_lite.csv"),
        "multi_scale_period_metrics_path": str(REPORTS_DIR / "multi_scale_period_metrics_lite.csv"),
    }
    write_csv_rows(Path(multi_scale_paths["history_coverage_path"]), multi_scale["coverage"])
    write_csv_rows(Path(multi_scale_paths["multi_scale_etf_metrics_path"]), multi_scale["etf_metrics"])
    write_csv_rows(Path(multi_scale_paths["multi_scale_family_metrics_path"]), multi_scale["family_metrics"])
    write_csv_rows(Path(multi_scale_paths["multi_scale_period_metrics_path"]), multi_scale["period_metrics"])
    panel, feature_cols, target_col = build_panel(histories, universe, args.horizon)
    write_csv_rows(PROCESSED_DIR / "training_panel_lite.csv", panel)
    train, test, split_date = time_split(panel, target_col, args.test_days)
    effective_l2 = args.l2
    effective_strategy_top_k = args.strategy_top_k
    effective_strategy_min_pred = args.strategy_min_pred
    tuning_best: dict[str, Any] | None = None
    tuning_model_rows: list[dict[str, Any]] = []
    tuning_strategy_rows: list[dict[str, Any]] = []
    if args.auto_tune:
        effective_l2, effective_strategy_top_k, effective_strategy_min_pred, tuning_best, tuning_model_rows, tuning_strategy_rows = tune_model_and_strategy(
            train,
            feature_cols,
            target_col,
            l2_grid=parse_float_grid(args.tune_l2_grid),
            top_k_grid=parse_int_grid(args.tune_top_k_grid),
            min_pred_grid=parse_float_grid(args.tune_min_pred_grid),
            tune_days=args.tune_days,
            round_trip_cost_bps=args.round_trip_cost_bps,
            horizon=args.horizon,
        )
        write_csv_rows(REPORTS_DIR / "model_parameter_tuning_lite.csv", tuning_model_rows)
        write_csv_rows(REPORTS_DIR / "strategy_parameter_tuning_lite.csv", tuning_strategy_rows)
    model = train_ridge(train, feature_cols, target_col, l2=effective_l2)
    metrics, holdout = evaluate(test, model, target_col)
    datasets = split_datasets(panel, train, test, model)
    dataset_paths = {
        "historical": str(PROCESSED_DIR / "historical_dataset_lite.csv"),
        "train": str(PROCESSED_DIR / "train_dataset_lite.csv"),
        "test": str(PROCESSED_DIR / "test_dataset_lite.csv"),
        "future": str(PROCESSED_DIR / "future_dataset_lite.csv"),
    }
    write_csv_rows(Path(dataset_paths["historical"]), datasets["historical"])
    write_csv_rows(Path(dataset_paths["train"]), datasets["train"])
    write_csv_rows(Path(dataset_paths["test"]), datasets["test"])
    write_csv_rows(Path(dataset_paths["future"]), datasets["future"])
    metrics.update(
        {
            "model_type": "stdlib_ridge",
            "model_l2": effective_l2,
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
            "historical_rows": len(datasets["historical"]),
            "future_rows": len(datasets["future"]),
            "feature_cols": feature_cols,
            "latest_candidates_path": str(REPORTS_DIR / "latest_candidates_lite.csv"),
            "latest_predictions_path": str(REPORTS_DIR / "latest_predictions_lite.csv"),
            "training_panel_path": str(PROCESSED_DIR / "training_panel_lite.csv"),
            "dataset_paths": dataset_paths,
            "model_path": args.model_out,
            "auto_tune": bool(args.auto_tune),
            "tuning_best": tuning_best,
            "model_tuning_path": str(REPORTS_DIR / "model_parameter_tuning_lite.csv") if tuning_model_rows else "",
            "strategy_tuning_path": str(REPORTS_DIR / "strategy_parameter_tuning_lite.csv") if tuning_strategy_rows else "",
            "related_stocks_path": str(REPORTS_DIR / "related_stocks_lite.csv") if related_stocks else "",
            "related_stock_count": len(related_stocks),
            "requested_start": args.start,
            "requested_end": args.end,
            "scale_horizons": scale_horizons,
            "long_history_start": min((str(row.get("start_date")) for row in multi_scale["coverage"]), default=""),
            "long_history_end": max((str(row.get("end_date")) for row in multi_scale["coverage"]), default=""),
            "long_history_max_years": max((to_float(row.get("history_years"), 0.0) or 0.0 for row in multi_scale["coverage"]), default=0.0),
            "long_history_min_years": min((to_float(row.get("history_years"), 0.0) or 0.0 for row in multi_scale["coverage"]), default=0.0),
            "long_history_coverage_count": len(multi_scale["coverage"]),
            "multi_scale_etf_metric_rows": len(multi_scale["etf_metrics"]),
            "multi_scale_family_metric_rows": len(multi_scale["family_metrics"]),
            "multi_scale_period_metric_rows": len(multi_scale["period_metrics"]),
            **multi_scale_paths,
        }
    )
    preds = latest_predictions(panel, model, args.horizon)
    write_csv_rows(REPORTS_DIR / "latest_predictions_lite.csv", preds)
    write_csv_rows(REPORTS_DIR / "holdout_predictions_lite.csv", holdout)
    buy_signals = build_buy_signals(preds, args.horizon, effective_strategy_top_k, effective_strategy_min_pred)
    write_csv_rows(REPORTS_DIR / "buy_signals_lite.csv", buy_signals)
    backtest_metrics, backtest_daily, backtest_trades = backtest_purchase_strategy(
        holdout,
        target_col,
        top_k=effective_strategy_top_k,
        min_pred=effective_strategy_min_pred,
        round_trip_cost_bps=args.round_trip_cost_bps,
        horizon=args.horizon,
    )
    write_csv_rows(REPORTS_DIR / "strategy_backtest_daily_lite.csv", backtest_daily)
    write_csv_rows(REPORTS_DIR / "strategy_backtest_trades_lite.csv", backtest_trades)
    metrics["buy_signals_path"] = str(REPORTS_DIR / "buy_signals_lite.csv")
    metrics["backtest_daily_path"] = str(REPORTS_DIR / "strategy_backtest_daily_lite.csv")
    metrics["backtest_trades_path"] = str(REPORTS_DIR / "strategy_backtest_trades_lite.csv")
    metrics["strategy_backtest"] = backtest_metrics
    chart_paths = generate_charts(
        universe,
        metrics,
        preds,
        holdout,
        target_col,
        args.horizon,
        backtest_metrics=backtest_metrics,
        backtest_rows=backtest_daily,
        tuning_model_rows=tuning_model_rows,
        tuning_strategy_rows=tuning_strategy_rows,
        related_stocks=related_stocks,
        multi_scale_family_rows=multi_scale["family_metrics"],
        multi_scale_period_rows=multi_scale["period_metrics"],
        history_coverage_rows=multi_scale["coverage"],
    )
    metrics["chart_paths"] = chart_paths
    (REPORTS_DIR / "training_metrics_lite.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_summary(
        universe,
        metrics,
        preds,
        feature_cols,
        args.horizon,
        chart_paths,
        backtest_metrics,
        buy_signals,
        dataset_paths,
        tuning_best,
        tuning_model_rows,
        tuning_strategy_rows,
        related_stocks,
        multi_scale["family_metrics"],
        multi_scale["coverage"],
    )
    artifact = {
        "model": model,
        "metrics": metrics,
        "universe": universe,
        "related_stocks": related_stocks,
        "buy_signals": buy_signals,
        "backtest_metrics": backtest_metrics,
        "multi_scale_paths": multi_scale_paths,
        "history_coverage": multi_scale["coverage"],
        "multi_scale_family_metrics": multi_scale["family_metrics"],
        "tuning_best": tuning_best,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as fh:
        pickle.dump(artifact, fh)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))
    print(f"\n轻量模型已保存：{model_path}")
    print(f"候选池：{REPORTS_DIR / 'latest_candidates_lite.csv'}")
    print(f"最新预测：{REPORTS_DIR / 'latest_predictions_lite.csv'}")
    print(f"买入信号：{REPORTS_DIR / 'buy_signals_lite.csv'}")
    print(f"策略回测：{REPORTS_DIR / 'strategy_backtest_daily_lite.csv'}")
    print(f"长历史覆盖：{REPORTS_DIR / 'long_history_coverage_lite.csv'}")
    print(f"跨时间尺度：{REPORTS_DIR / 'multi_scale_family_metrics_lite.csv'}")
    if related_stocks:
        print(f"相关股票：{REPORTS_DIR / 'related_stocks_lite.csv'}")
    if tuning_best:
        print(f"自动调参：L2={effective_l2}, TopK={effective_strategy_top_k}, min_pred={effective_strategy_min_pred:.4f}")
    print(f"报告：{REPORTS_DIR / 'training_summary_lite.md'}")
    print(f"图表目录：{REPORTS_DIR / 'charts'}")


if __name__ == "__main__":
    main()
