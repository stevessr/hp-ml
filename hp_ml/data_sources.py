"""Live data access for Chinese exchange-traded funds.

The implementation uses Eastmoney quote endpoints because they expose both the
full ETF spot list and daily K-line history without needing a vendor terminal.
Responses are cached to CSV so modelling can be repeated offline after a fetch.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import HISTORY_DIR, RAW_DIR

EASTMONEY_ETF_LIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_FUND_CODE_SEARCH_URL = "https://fund.eastmoney.com/js/fundcode_search.js"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
LISTED_ETF_PREFIXES = ("159", "510", "512", "515", "516", "517", "520", "560", "561", "562", "563", "588", "589")

SPOT_FIELDS = {
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


def today_yyyymmdd() -> str:
    return dt.date.today().strftime("%Y%m%d")


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session


def _get_json(session: requests.Session, url: str, params: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    text = response.text.strip()
    # Some finance endpoints may be wrapped as callback(...); keep the parser
    # tolerant even though the current endpoints return pure JSON.
    if text and not text.startswith("{"):
        left = text.find("{")
        right = text.rfind("}")
        if left >= 0 and right >= left:
            text = text[left : right + 1]
    return json.loads(text)


def fetch_etf_spot(force: bool = False, cache_path: Path | None = None) -> pd.DataFrame:
    """Fetch the full exchange ETF quote list from Eastmoney.

    Parameters
    ----------
    force:
        Ignore the same-day cache and hit the live endpoint.
    cache_path:
        Optional explicit CSV path.  Defaults to data/raw/etf_spot_YYYYMMDD.csv.
    """

    cache_path = cache_path or RAW_DIR / f"etf_spot_{today_yyyymmdd()}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, dtype={"code": str})

    session = make_session()
    params = {
        "pn": 1,
        "pz": 10000,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f6",  # sort by amount to make top liquidity visible
        # Eastmoney ETF boards. This fs expression is also used by common public
        # ETF quote examples and currently returns all listed ETFs.
        "fs": "b:MK0021,b:MK0022,b:MK0023,b:MK0024",
        "fields": ",".join(SPOT_FIELDS.keys()),
    }
    payload = _get_json(session, EASTMONEY_ETF_LIST_URL, params=params)
    rows = (payload.get("data") or {}).get("diff") or []
    if not rows:
        raise RuntimeError("Eastmoney ETF spot endpoint returned no rows")

    df = pd.DataFrame(rows).rename(columns=SPOT_FIELDS)
    keep_cols = list(SPOT_FIELDS.values())
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    for col in [c for c in df.columns if c not in {"code", "name"}]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fetched_at"] = dt.datetime.now().isoformat(timespec="seconds")
    df.to_csv(cache_path, index=False)
    return df


def fetch_fundcode_search(force: bool = False, cache_path: Path | None = None) -> pd.DataFrame:
    """Fetch Eastmoney fund-code search data as a quote-list fallback.

    It has no live liquidity fields but exposes the current fund-code/name
    catalogue, which is still useful for automatic ETF universe discovery when
    the quote endpoint returns 5xx.
    """

    cache_path = cache_path or RAW_DIR / f"fundcode_search_{today_yyyymmdd()}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, dtype={"code": str})

    session = make_session()
    response = session.get(EASTMONEY_FUND_CODE_SEARCH_URL, timeout=30)
    response.raise_for_status()
    text = response.content.decode("utf-8-sig", "ignore")
    left = text.find("[[")
    right = text.rfind("]]")
    if left < 0 or right < left:
        raise RuntimeError("fundcode_search.js did not contain a JSON array")
    data = json.loads(text[left : right + 2])
    rows = []
    for item in data:
        if len(item) < 4:
            continue
        code = str(item[0]).zfill(6)
        name = str(item[2])
        fund_type = str(item[3])
        normalized = name.upper()
        if not code.startswith(LISTED_ETF_PREFIXES):
            continue
        if "ETF" not in normalized or "联接" in normalized or "LOF" in normalized:
            continue
        rows.append({"code": code, "name": name, "fund_type": fund_type, "source": "eastmoney_fundcode_search"})
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df


def infer_eastmoney_market(code: str) -> int:
    """Return Eastmoney market id: 1 for Shanghai, 0 for Shenzhen."""

    code = str(code).strip().zfill(6)
    if code.startswith(("5", "6", "9")):
        return 1
    return 0


def secid_for_code(code: str) -> str:
    code = str(code).strip().zfill(6)
    return f"{infer_eastmoney_market(code)}.{code}"


def _market_prefix(code: str) -> str:
    return "sh" if str(code).zfill(6).startswith(("5", "6", "9")) else "sz"


def _yyyymmdd_to_iso(value: str) -> str:
    value = str(value)
    if "-" in value:
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def fetch_etf_history_tencent(code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """Fetch ETF daily history from Tencent as a fallback data source."""

    code = str(code).strip().zfill(6)
    symbol = _market_prefix(code) + code
    fq = "qfq" if adjust.lower() == "qfq" else ""
    param = f"{symbol},day,{_yyyymmdd_to_iso(start)},{_yyyymmdd_to_iso(end)},2000,{fq}".rstrip(",")
    session = make_session()
    payload = _get_json(session, TENCENT_KLINE_URL, params={"param": param}, timeout=12)
    node = (payload.get("data") or {}).get(symbol) or {}
    lines = node.get("qfqday") or node.get("hfqday") or node.get("day") or []
    if not lines:
        raise RuntimeError(f"No Tencent K-line rows for ETF {code}")

    rows = []
    prev_close: float | None = None
    for item in lines:
        if len(item) < 6:
            continue
        date_s, open_v, close_v, high_v, low_v, volume_v = item[:6]
        row = {
            "code": code,
            "date": date_s,
            "open": pd.to_numeric(open_v, errors="coerce"),
            "close": pd.to_numeric(close_v, errors="coerce"),
            "high": pd.to_numeric(high_v, errors="coerce"),
            "low": pd.to_numeric(low_v, errors="coerce"),
            "volume": pd.to_numeric(volume_v, errors="coerce"),
            "amount": None,
            "amplitude": None,
            "pct_chg": None,
            "price_change": None,
            "turnover_rate": None,
        }
        if prev_close not in (None, 0) and pd.notna(row["close"]):
            row["price_change"] = row["close"] - prev_close
            row["pct_chg"] = row["price_change"] / prev_close * 100
            row["amplitude"] = (row["high"] - row["low"]) / prev_close * 100
        if pd.notna(row["close"]):
            prev_close = float(row["close"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for col in HISTORY_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").drop_duplicates(["code", "date"], keep="last")


def fetch_etf_history(
    code: str,
    start: str = "20180101",
    end: str | None = None,
    adjust: str = "qfq",
    force: bool = False,
    sleep_seconds: float = 0.12,
) -> pd.DataFrame:
    """Fetch daily ETF history from Eastmoney and cache it to CSV.

    Parameters
    ----------
    code:
        6-digit ETF code.
    start, end:
        Date strings in YYYYMMDD.
    adjust:
        ``none``/``""`` for raw, ``qfq`` for front-adjusted, ``hfq`` for back-adjusted.
    force:
        Ignore the code/date cache.
    """

    code = str(code).strip().zfill(6)
    end = end or today_yyyymmdd()
    adjust_key = {"": 0, "none": 0, "raw": 0, "qfq": 1, "hfq": 2}.get(adjust.lower(), 1)
    cache_path = HISTORY_DIR / f"{code}_{start}_{end}_{adjust or 'raw'}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, dtype={"code": str}, parse_dates=["date"])

    try:
        session = make_session()
        params = {
            "secid": secid_for_code(code),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": 101,
            "fqt": adjust_key,
            "beg": start,
            "end": end,
        }
        if sleep_seconds:
            time.sleep(sleep_seconds)
        payload = _get_json(session, EASTMONEY_KLINE_URL, params=params, timeout=12)
        klines = ((payload.get("data") or {}).get("klines")) or []
        if not klines:
            raise RuntimeError(f"No K-line rows returned for ETF {code} ({params['secid']})")

        records: list[list[str]] = [line.split(",") for line in klines]
        df = pd.DataFrame(records, columns=HISTORY_COLUMNS)
        df.insert(0, "code", code)
        df["date"] = pd.to_datetime(df["date"])
        for col in HISTORY_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("date").drop_duplicates(["code", "date"], keep="last")
    except Exception:
        df = fetch_etf_history_tencent(code, start=start, end=end, adjust=adjust)
    df.to_csv(cache_path, index=False)
    return df


def fetch_many_histories(
    codes: list[str],
    start: str,
    end: str | None = None,
    adjust: str = "qfq",
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for code in codes:
        try:
            histories[code] = fetch_etf_history(code, start=start, end=end, adjust=adjust, force=force)
        except Exception as exc:  # noqa: BLE001 - keep pipeline resilient to one bad fund
            errors[code] = str(exc)
    if errors:
        (RAW_DIR / "history_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    return histories


def human_amount(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = float(value)
    if abs(value) >= 1e8:
        return f"{value / 1e8:.2f}亿"
    if abs(value) >= 1e4:
        return f"{value / 1e4:.2f}万"
    return f"{value:.0f}"
