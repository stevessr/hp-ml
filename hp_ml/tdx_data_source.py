"""通达信数据源 - 用于训练流水线

这个模块提供了与 data_sources.py 兼容的接口，用于从通达信服务器拉取 ETF K 线数据。
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import HISTORY_DIR
from .tdx_client import TdxClient, fetch_kline_day


def fetch_etf_history_tdx(
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    cache: bool = True,
    host: str | None = None,
) -> pd.DataFrame:
    """从通达信拉取 ETF 历史 K 线数据

    Args:
        code: ETF 代码（如 "sz159300", "sh510300"）
        start_date: 开始日期（格式：YYYYMMDD），None=尽可能早
        end_date: 结束日期（格式：YYYYMMDD），None=今天
        cache: 是否使用缓存
        host: 通达信服务器地址，None=自动选择

    Returns:
        DataFrame，包含列：date, open, high, low, close, volume, amount
    """
    # 检查缓存
    if cache:
        cache_path = HISTORY_DIR / f"{code}_tdx.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            df["date"] = pd.to_datetime(df["date"])
            return df

    # 从通达信拉取数据
    with TdxClient(host=host) as client:
        klines = client.get_kline_day_all(code, max_count=10000)

    if not klines:
        raise ValueError(f"未能获取 {code} 的 K 线数据")

    # 转换为 DataFrame
    df = pd.DataFrame([
        {
            "date": k.time,
            "open": k.open,
            "high": k.high,
            "low": k.low,
            "close": k.close,
            "volume": k.volume,
            "amount": k.amount,
        }
        for k in klines
    ])

    # 过滤日期范围
    if start_date:
        start_dt = pd.to_datetime(start_date, format="%Y%m%d")
        df = df[df["date"] >= start_dt]

    if end_date:
        end_dt = pd.to_datetime(end_date, format="%Y%m%d")
        df = df[df["date"] <= end_dt]

    # 保存缓存
    if cache:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def fetch_etf_batch_tdx(
    codes: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    cache: bool = True,
    host: str | None = None,
    delay: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """批量从通达信拉取 ETF 历史 K 线数据

    Args:
        codes: ETF 代码列表
        start_date: 开始日期（格式：YYYYMMDD）
        end_date: 结束日期（格式：YYYYMMDD）
        cache: 是否使用缓存
        host: 通达信服务器地址
        delay: 每次请求之间的延迟（秒），避免请求过快

    Returns:
        字典，键为 ETF 代码，值为 DataFrame
    """
    result = {}

    for i, code in enumerate(codes):
        try:
            df = fetch_etf_history_tdx(
                code,
                start_date=start_date,
                end_date=end_date,
                cache=cache,
                host=host,
            )
            result[code] = df
            print(f"✓ [{i+1}/{len(codes)}] {code}: {len(df)} 条记录")

            # 延迟以避免请求过快
            if i < len(codes) - 1 and delay > 0:
                time.sleep(delay)

        except Exception as e:
            print(f"✗ [{i+1}/{len(codes)}] {code}: {e}")
            continue

    return result


if __name__ == "__main__":
    # 测试代码
    print("测试通达信数据源...\n")

    test_codes = ["sz159300", "sh510300", "sz000001"]

    for code in test_codes:
        print(f"拉取 {code} 数据...")
        try:
            df = fetch_etf_history_tdx(code, cache=False)
            print(f"✓ 成功获取 {len(df)} 条记录")
            print(f"  时间范围：{df['date'].min()} 至 {df['date'].max()}")
            print(f"  最新数据:\n{df.tail(3)}\n")
        except Exception as e:
            print(f"✗ 失败：{e}\n")
