from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hp_ml.data_sources import fetch_etf_history  # noqa: E402
from hp_ml.universe import discover_broad_etfs  # noqa: E402

DEFAULT_START_DATE = "20050101"
DEFAULT_END_DATE = "20260603"
DEFAULT_ADJUST = "qfq"
CHINESE_COLUMNS = {
    "code": "代码",
    "date": "日期",
    "open": "开盘价",
    "close": "收盘价",
    "high": "最高价",
    "low": "最低价",
    "volume": "成交量",
    "amount": "成交额",
    "amplitude": "振幅",
    "pct_chg": "涨跌幅",
    "price_change": "涨跌额",
    "turnover_rate": "换手率",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载所有中证/沪深宽基指数相关 ETF 的历史日线数据")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="开始日期，格式 YYYYMMDD")
    parser.add_argument("--end", default=DEFAULT_END_DATE, help="结束日期，格式 YYYYMMDD")
    parser.add_argument("--adjust", default=DEFAULT_ADJUST, choices=("qfq", "hfq", "raw", "none"), help="复权方式")
    parser.add_argument("--output-dir", default="data/raw/history", help="CSV 输出目录")
    parser.add_argument("--summary", default="reports/downloaded_csi_broad_etfs.csv", help="下载汇总 CSV 路径")
    parser.add_argument("--force", action="store_true", help="忽略缓存并重新拉取行情")
    parser.add_argument("--include-enhanced", action="store_true", help="包含指数增强 ETF")
    parser.add_argument("--include-style", action="store_true", help="包含红利、低波、成长等风格/主题 ETF")
    return parser.parse_args()


def to_chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    return result.rename(columns=CHINESE_COLUMNS)


def safe_filename(code: str, name: str, start: str, end: str, adjust: str) -> str:
    clean_name = "".join(ch for ch in name if ch not in r'\\/:*?"<>|').strip()
    adjust_key = "raw" if adjust in {"", "none", "raw"} else adjust
    return f"{code}_{clean_name}_{start}_{end}_{adjust_key}.csv"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)

    universe = discover_broad_etfs(
        max_per_family=None,
        include_enhanced=args.include_enhanced,
        include_style=args.include_style,
        force=args.force,
    )
    universe = universe.sort_values(["family_id", "selected_rank_in_family", "code"]).reset_index(drop=True)

    print(f"发现 {len(universe)} 只中证/沪深宽基相关 ETF，开始下载 {args.start} 至 {args.end} 的日线数据...")

    summaries: list[dict[str, object]] = []
    failed: list[tuple[str, str, str]] = []

    for position, (_, row) in enumerate(universe.iterrows(), start=1):
        code = str(row["code"]).zfill(6)
        name = str(row.get("name") or row.get("display_name") or "ETF")
        family_id = str(row.get("family_id") or "")
        display_name = str(row.get("display_name") or "")
        print(f"[{position}/{len(universe)}] 下载 {code} {name}（{display_name}）...")
        try:
            history = fetch_etf_history(
                code,
                start=args.start,
                end=args.end,
                adjust=args.adjust,
                force=args.force,
            )
            if history.empty:
                raise RuntimeError("历史行情为空")

            csv_path = output_dir / safe_filename(code, name, args.start, args.end, args.adjust)
            to_chinese_columns(history).to_csv(csv_path, index=False, encoding="utf-8-sig")
            summaries.append(
                {
                    "code": code,
                    "name": name,
                    "family_id": family_id,
                    "display_name": display_name,
                    "rows": len(history),
                    "start_date": history["date"].min().strftime("%Y-%m-%d"),
                    "end_date": history["date"].max().strftime("%Y-%m-%d"),
                    "csv_path": str(csv_path),
                }
            )
        except Exception as exc:  # noqa: BLE001 - 单只 ETF 失败不应中断全量下载
            failed.append((code, name, str(exc)))
            print(f"  失败：{exc}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.summary, index=False, encoding="utf-8-sig")

    print("\n下载完成！")
    print(f"成功：{len(summaries)} 只；失败：{len(failed)} 只。")
    print(f"汇总文件：{args.summary}")
    if not summary_df.empty:
        print("数据样本：")
        print(summary_df.head())
    if failed:
        print("\n失败清单：")
        for code, name, message in failed:
            print(f"- {code} {name}: {message}")


if __name__ == "__main__":
    main()
