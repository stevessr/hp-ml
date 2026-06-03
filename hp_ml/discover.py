"""CLI for ETF universe discovery."""
from __future__ import annotations

import argparse

from .universe import discover_broad_etfs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover CSI broad-based ETFs from live ETF quotes.")
    parser.add_argument("--max-per-family", type=int, default=5, help="Keep top N ETFs per broad-index family.")
    parser.add_argument("--min-amount", type=float, default=0.0, help="Minimum same-day turnover amount in CNY.")
    parser.add_argument("--include-enhanced", action="store_true", help="Keep index-enhanced ETFs.")
    parser.add_argument("--include-style", action="store_true", help="Keep style/theme products that contain broad-index names.")
    parser.add_argument("--force", action="store_true", help="Refresh live spot cache.")
    parser.add_argument("--out", default="reports/latest_candidates.csv", help="CSV output path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    df = discover_broad_etfs(
        max_per_family=args.max_per_family,
        min_amount=args.min_amount,
        include_enhanced=args.include_enhanced,
        include_style=args.include_style,
        force=args.force,
    )
    df.to_csv(args.out, index=False)
    print(f"发现 {len(df)} 只中证宽基 ETF 候选，已写入 {args.out}")
    display_cols = [c for c in ["code", "name", "family_id", "latest_price", "amount", "selected_rank_in_family"] if c in df]
    print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
