"""Walk-forward model/strategy auto-tuning until it beats a baseline.

This module is intentionally research-oriented: it searches model and portfolio
control parameters on the available historical data, keeps the best data-driven
configuration, and writes an auditable report that compares the strategy with an
equal-weight ETF baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, REPORTS_DIR

FEATURE_COLUMNS = ["F_Tech", "F_Sent", "Return_Lag1", "Vol_Lag1"]


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters for the rolling directional classifier."""

    tech_window: int = 10
    sent_window: int = 20
    train_lookback_days: int = 365
    max_iter: int = 100
    max_depth: int = 3
    learning_rate: float = 0.03

    @property
    def label(self) -> str:
        return (
            f"tech{self.tech_window}_sent{self.sent_window}_lookback{self.train_lookback_days}"
            f"_hgb{self.max_iter}x{self.max_depth}_lr{self.learning_rate:g}"
        )


@dataclass(frozen=True)
class StrategyConfig:
    """Portfolio-control parameters searched after model probabilities exist."""

    mode: str = "ema"  # "ema" or "discrete"
    top_k: int = 2
    prob_threshold: float = 0.49
    sent_break_threshold: float = 1.2
    trend_filter: bool = True
    alpha_buy: float = 0.6
    alpha_sell: float = 0.05
    fee_rate: float = 0.001

    @property
    def label(self) -> str:
        filter_label = "trend" if self.trend_filter else "no_trend"
        if self.mode == "ema":
            mode_label = f"ema_ab{self.alpha_buy:g}_as{self.alpha_sell:g}"
        else:
            mode_label = "discrete"
        return (
            f"{mode_label}_top{self.top_k}_p{self.prob_threshold:g}"
            f"_sent{self.sent_break_threshold:g}_{filter_label}_fee{self.fee_rate:g}"
        )


@dataclass
class TuneResult:
    """Best tuning result and all evidence needed for reporting."""

    success: bool
    model_config: ModelConfig
    strategy_config: StrategyConfig
    metrics: dict[str, Any]
    daily: pd.DataFrame
    trials: pd.DataFrame
    signal_count: int


def cumulative_return(returns: Iterable[float]) -> float:
    values = [float(v) for v in returns if v is not None and math.isfinite(float(v))]
    if not values:
        return 0.0
    return float(np.prod(1.0 + np.asarray(values, dtype=float)) - 1.0)


def max_drawdown_from_returns(returns: Iterable[float]) -> float:
    values = [float(v) for v in returns if v is not None and math.isfinite(float(v))]
    if not values:
        return 0.0
    curve = np.cumprod(1.0 + np.asarray(values, dtype=float))
    peaks = np.maximum.accumulate(curve)
    drawdowns = curve / peaks - 1.0
    return float(drawdowns.min())


def annualized_return(total_return: float, observations: int) -> float | None:
    if observations <= 0 or total_return <= -1:
        return None
    return float((1.0 + total_return) ** (252.0 / observations) - 1.0)


def sharpe_like(returns: Iterable[float]) -> float | None:
    values = np.asarray([float(v) for v in returns if v is not None and math.isfinite(float(v))], dtype=float)
    if len(values) < 2:
        return None
    vol = float(values.std(ddof=1) * math.sqrt(252))
    if vol <= 0:
        return None
    return float(values.mean() * 252 / vol)


def strategy_metrics(daily: pd.DataFrame, *, min_excess: float = 0.0) -> dict[str, Any]:
    strategy_total = cumulative_return(daily["Strategy"])
    benchmark_total = cumulative_return(daily["Benchmark"])
    observations = int(len(daily))
    active_days = int((daily["selected_count"] > 0).sum()) if "selected_count" in daily else 0
    return {
        "success": bool(strategy_total > benchmark_total + min_excess),
        "observations": observations,
        "active_days": active_days,
        "active_day_rate": active_days / observations if observations else None,
        "strategy_cumulative_return": strategy_total,
        "benchmark_cumulative_return": benchmark_total,
        "excess_cumulative_return": strategy_total - benchmark_total,
        "strategy_annualized_return": annualized_return(strategy_total, observations),
        "benchmark_annualized_return": annualized_return(benchmark_total, observations),
        "strategy_max_drawdown": max_drawdown_from_returns(daily["Strategy"]),
        "benchmark_max_drawdown": max_drawdown_from_returns(daily["Benchmark"]),
        "strategy_sharpe_like": sharpe_like(daily["Strategy"]),
        "benchmark_sharpe_like": sharpe_like(daily["Benchmark"]),
        "avg_strategy_daily_return": float(pd.Series(daily["Strategy"]).mean()) if observations else None,
        "avg_benchmark_daily_return": float(pd.Series(daily["Benchmark"]).mean()) if observations else None,
        "outperform_day_rate": float((daily["Strategy"] > daily["Benchmark"]).mean()) if observations else None,
        "avg_turnover": float(daily["turnover"].mean()) if "turnover" in daily and observations else None,
        "avg_selected_count": float(daily["selected_count"].mean()) if "selected_count" in daily and observations else None,
    }


def load_price_series(data_dir: Path) -> dict[str, pd.Series]:
    """Load ``*_daily.csv`` close-price files from a directory."""

    if not data_dir.exists():
        raise FileNotFoundError(f"ETF data directory does not exist: {data_dir}")
    prices: dict[str, pd.Series] = {}
    for path in sorted(data_dir.glob("*_daily.csv")):
        code = path.name.split("_", 1)[0]
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
        if "Close" in df.columns:
            close = df["Close"]
        else:
            close = df.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna().astype(float)
        if not close.empty:
            prices[code] = close
    if not prices:
        raise RuntimeError(f"No *_daily.csv price files found in {data_dir}")
    return prices


def load_sentiment_series(path: Path) -> tuple[pd.Series, str]:
    """Load the most relevant Google Trends sentiment column."""

    if not path.exists():
        raise FileNotFoundError(f"Sentiment file does not exist: {path}")
    trends = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if trends.empty:
        raise RuntimeError(f"Sentiment file is empty: {path}")
    candidates = [col for col in trends.columns if "A" in col or "股" in col or "开户" in col]
    column = candidates[0] if candidates else str(trends.columns[0])
    return pd.to_numeric(trends[column], errors="coerce").fillna(0.0), column


def build_feature_panel(
    prices: dict[str, pd.Series],
    sentiment: pd.Series,
    config: ModelConfig,
) -> pd.DataFrame:
    """Build a long panel with model features and next-day direction labels."""

    sent_std = sentiment.rolling(config.sent_window).std().replace(0, np.nan)
    sent_z = ((sentiment - sentiment.rolling(config.sent_window).mean()) / sent_std).fillna(0.0)
    frames: list[pd.DataFrame] = []
    for code, close in prices.items():
        frame = pd.DataFrame(index=close.index)
        frame["Price"] = close
        frame["Daily_Return"] = close.pct_change().fillna(0.0)
        frame["SMA"] = close.rolling(config.tech_window).mean()
        raw_tech = (close / frame["SMA"] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        tech_std = raw_tech.rolling(config.tech_window).std().replace(0, np.nan)
        frame["F_Tech"] = ((raw_tech - raw_tech.rolling(config.tech_window).mean()) / tech_std).fillna(0.0)
        frame["Vol_Lag1"] = frame["Daily_Return"].rolling(5).std().shift(1)
        frame["Return_Lag1"] = frame["Daily_Return"].shift(1)
        frame["F_Sent"] = sent_z.reindex(close.index, method="ffill").fillna(0.0)
        frame["Raw_Sentiment_Z"] = frame["F_Sent"]
        next_day_return = frame["Daily_Return"].shift(-1)
        frame["Target"] = np.where(next_day_return.notna(), (next_day_return > 0).astype(float), np.nan)
        frame["Asset_Code"] = code
        frame = frame[
            FEATURE_COLUMNS
            + ["Target", "Daily_Return", "Asset_Code", "Price", "SMA", "Raw_Sentiment_Z"]
        ].dropna(subset=FEATURE_COLUMNS + ["Daily_Return", "Asset_Code", "Price", "SMA", "Raw_Sentiment_Z"])
        frames.append(frame)
    if not frames:
        raise RuntimeError("No usable price frames after feature engineering")
    return pd.concat(frames).sort_index()


def walkforward_boundaries(start: str, end: str) -> list[pd.Timestamp]:
    """Return month-end walk-forward boundaries including exact start/end."""

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts <= start_ts:
        raise ValueError(f"end must be after start: start={start}, end={end}")
    month_ends = list(pd.date_range(start=start_ts, end=end_ts, freq="ME"))
    boundaries = [start_ts] + [d for d in month_ends if start_ts < d < end_ts] + [end_ts]
    deduped = sorted(set(pd.Timestamp(d) for d in boundaries))
    if len(deduped) < 2:
        raise ValueError(f"Not enough walk-forward boundaries between {start} and {end}")
    return deduped


def generate_walkforward_signals(
    panel: pd.DataFrame,
    config: ModelConfig,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Train a rolling classifier and emit per-asset daily probabilities."""

    signal_rows: list[dict[str, Any]] = []
    boundaries = walkforward_boundaries(start, end)
    for i in range(len(boundaries) - 1):
        window_start = boundaries[i] if i == 0 else boundaries[i] + pd.Timedelta(days=1)
        window_end = boundaries[i + 1]
        train_start = window_start - pd.Timedelta(days=config.train_lookback_days)
        train_end = window_start - pd.Timedelta(days=1)
        train = panel.loc[train_start:train_end].dropna(subset=["Target"])
        test = panel.loc[window_start:window_end]
        if train.empty or test.empty:
            continue
        if train["Target"].nunique(dropna=True) < 2:
            continue

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train[FEATURE_COLUMNS])
        model = HistGradientBoostingClassifier(
            max_iter=config.max_iter,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            random_state=42,
        )
        model.fit(x_train, train["Target"].astype(int))
        positive_index = int(np.where(model.classes_ == 1)[0][0])

        for day in test.index.unique().sort_values():
            day_data = test.loc[[day]]
            probabilities = model.predict_proba(scaler.transform(day_data[FEATURE_COLUMNS]))[:, positive_index]
            for row_index, (_, row) in enumerate(day_data.iterrows()):
                signal_rows.append(
                    {
                        "date": pd.Timestamp(day).date().isoformat(),
                        "asset_code": str(row["Asset_Code"]),
                        "daily_return": float(row["Daily_Return"]),
                        "price": float(row["Price"]),
                        "sma": float(row["SMA"]),
                        "sentiment_z": float(row["Raw_Sentiment_Z"]),
                        "prob_up": float(probabilities[row_index]),
                        "model_config": config.label,
                    }
                )

    if not signal_rows:
        raise RuntimeError(f"No walk-forward signals generated for {config}")
    return pd.DataFrame(signal_rows).sort_values(["date", "asset_code"]).reset_index(drop=True)


def simulate_strategy(signals: pd.DataFrame, codes: list[str], config: StrategyConfig) -> pd.DataFrame:
    """Simulate an equal-weight selected portfolio from probability signals."""

    if config.mode not in {"ema", "discrete"}:
        raise ValueError(f"Unsupported strategy mode: {config.mode}")

    weights = {code: 0.0 for code in codes}
    daily_rows: list[dict[str, Any]] = []
    for date_s, group in signals.groupby("date", sort=True):
        group = group.copy()
        benchmark_return = float(group["daily_return"].mean())
        sentiment_z = float(group["sentiment_z"].iloc[0]) if not group.empty else 0.0
        selected = pd.DataFrame(columns=group.columns)
        if sentiment_z <= config.sent_break_threshold:
            candidates = group[group["prob_up"] > config.prob_threshold].copy()
            if config.trend_filter:
                candidates = candidates[candidates["price"] > candidates["sma"]]
            if not candidates.empty:
                selected = candidates.nlargest(max(1, int(config.top_k)), "prob_up")

        target_weights = {code: 0.0 for code in codes}
        if not selected.empty:
            allocation = 1.0 / len(selected)
            for code in selected["asset_code"].astype(str):
                target_weights[code] = allocation

        gross_return = 0.0
        turnover = 0.0
        day_returns = dict(zip(group["asset_code"].astype(str), group["daily_return"].astype(float)))
        for code in codes:
            previous = weights[code]
            target = target_weights[code]
            if config.mode == "ema":
                alpha = config.alpha_buy if target > previous else config.alpha_sell
                executed = float(np.clip(alpha * target + (1.0 - alpha) * previous, 0.0, 1.0))
            else:
                executed = target
            gross_return += executed * day_returns.get(code, 0.0)
            turnover += abs(executed - previous)
            weights[code] = executed
        fee = turnover * config.fee_rate
        raw_strategy_return = gross_return - fee
        daily_rows.append(
            {
                "date": date_s,
                "Benchmark": benchmark_return,
                "strategy_return_raw": raw_strategy_return,
                "Strategy": raw_strategy_return,  # replaced by one-day-delayed returns below
                "gross_strategy_return": gross_return,
                "fee": fee,
                "turnover": turnover,
                "selected_count": int(len(selected)),
                "selected_codes": ",".join(selected["asset_code"].astype(str).tolist()) if not selected.empty else "",
                "avg_selected_prob": float(selected["prob_up"].mean()) if not selected.empty else None,
                "strategy_config": config.label,
            }
        )

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise RuntimeError("Strategy simulation produced no daily rows")
    # Signals are formed after the current close, so strategy PnL is shifted one
    # trading day to avoid using same-day returns as immediately tradable PnL.
    daily["Strategy"] = daily["strategy_return_raw"].shift(1).fillna(0.0)
    daily["strategy_cumulative"] = (1.0 + daily["Strategy"]).cumprod() - 1.0
    daily["benchmark_cumulative"] = (1.0 + daily["Benchmark"]).cumprod() - 1.0
    daily["excess_cumulative"] = daily["strategy_cumulative"] - daily["benchmark_cumulative"]
    return daily


def default_model_configs() -> list[ModelConfig]:
    """Small, ordered grid. First config is fast; later configs widen capacity."""

    return [
        ModelConfig(tech_window=10, sent_window=20, train_lookback_days=365, max_iter=100, max_depth=3, learning_rate=0.03),
        ModelConfig(tech_window=20, sent_window=20, train_lookback_days=730, max_iter=150, max_depth=5, learning_rate=0.03),
        ModelConfig(tech_window=30, sent_window=20, train_lookback_days=365, max_iter=100, max_depth=3, learning_rate=0.05),
        ModelConfig(tech_window=20, sent_window=30, train_lookback_days=730, max_iter=120, max_depth=4, learning_rate=0.03),
    ]


def default_strategy_configs(fee_rate: float) -> list[StrategyConfig]:
    """Generate strategy-control candidates for the auto-tuner."""

    configs: list[StrategyConfig] = []
    for top_k in (1, 2, 3, 5):
        for prob_threshold in (0.49, 0.498, 0.5):
            for sent_break in (1.2, 1.6):
                for trend_filter in (True, False):
                    for alpha_buy, alpha_sell in ((0.6, 0.05), (0.85, 0.12)):
                        configs.append(
                            StrategyConfig(
                                mode="ema",
                                top_k=top_k,
                                prob_threshold=prob_threshold,
                                sent_break_threshold=sent_break,
                                trend_filter=trend_filter,
                                alpha_buy=alpha_buy,
                                alpha_sell=alpha_sell,
                                fee_rate=fee_rate,
                            )
                        )
                    configs.append(
                        StrategyConfig(
                            mode="discrete",
                            top_k=top_k,
                            prob_threshold=prob_threshold,
                            sent_break_threshold=sent_break,
                            trend_filter=trend_filter,
                            alpha_buy=1.0,
                            alpha_sell=1.0,
                            fee_rate=fee_rate,
                        )
                    )
    return configs


def auto_tune_until_baseline(
    *,
    data_dir: Path,
    sentiment_path: Path,
    start: str,
    end: str,
    min_excess: float = 0.0,
    max_model_configs: int = 1,
    fee_rate: float = 0.001,
) -> TuneResult:
    """Search configurations and return the best result.

    The search stops after the first model-configuration block whose best
    strategy beats the baseline by ``min_excess``.  Within a model block all
    strategy candidates are evaluated so the selected row is the best observed
    strategy for that model's probabilities.
    """

    prices = load_price_series(data_dir)
    codes = sorted(prices)
    sentiment, sentiment_column = load_sentiment_series(sentiment_path)
    model_configs = default_model_configs()[: max(1, int(max_model_configs))]
    strategy_configs = default_strategy_configs(fee_rate)

    best_result: tuple[dict[str, Any], pd.DataFrame, ModelConfig, StrategyConfig, int] | None = None
    trial_rows: list[dict[str, Any]] = []
    for model_index, model_config in enumerate(model_configs, 1):
        panel = build_feature_panel(prices, sentiment, model_config)
        signals = generate_walkforward_signals(panel, model_config, start=start, end=end)
        for strategy_index, strategy_config in enumerate(strategy_configs, 1):
            daily = simulate_strategy(signals, codes, strategy_config)
            metrics = strategy_metrics(daily, min_excess=min_excess)
            row = {
                "model_trial": model_index,
                "strategy_trial": strategy_index,
                "model_config": model_config.label,
                "strategy_config": strategy_config.label,
                "sentiment_column": sentiment_column,
                **asdict(model_config),
                **asdict(strategy_config),
                **metrics,
            }
            trial_rows.append(row)
            if best_result is None or row["excess_cumulative_return"] > best_result[0]["excess_cumulative_return"]:
                best_result = (row, daily, model_config, strategy_config, len(signals))

        assert best_result is not None
        if best_result[0]["success"]:
            break

    if best_result is None:
        raise RuntimeError("No tuning trials were evaluated")
    trial_df = pd.DataFrame(trial_rows).sort_values("excess_cumulative_return", ascending=False).reset_index(drop=True)
    best_metrics, best_daily, best_model, best_strategy, signal_count = best_result
    return TuneResult(
        success=bool(best_metrics["success"]),
        model_config=best_model,
        strategy_config=best_strategy,
        metrics=best_metrics,
        daily=best_daily,
        trials=trial_df,
        signal_count=signal_count,
    )


def write_outputs(
    result: TuneResult,
    *,
    daily_out: Path,
    trials_out: Path,
    summary_out: Path,
    start: str,
    end: str,
    data_dir: Path,
    sentiment_path: Path,
) -> None:
    """Persist the selected daily report, all trials and a JSON summary."""

    daily_out.parent.mkdir(parents=True, exist_ok=True)
    trials_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    result.daily.to_csv(daily_out, index=False)
    result.trials.to_csv(trials_out, index=False)
    summary = {
        "success": result.success,
        "start": start,
        "end": end,
        "data_dir": str(data_dir),
        "sentiment_path": str(sentiment_path),
        "model_config": asdict(result.model_config),
        "strategy_config": asdict(result.strategy_config),
        "model_config_label": result.model_config.label,
        "strategy_config_label": result.strategy_config.label,
        "signal_count": result.signal_count,
        "trial_count": int(len(result.trials)),
        "daily_report_path": str(daily_out),
        "trial_report_path": str(trials_out),
        "metrics": result.metrics,
        "note": "Research backtest. The strategy is selected by auto-tuning on available historical data and compared with an equal-weight ETF baseline.",
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-tune a rolling ETF model until it beats an equal-weight baseline.")
    parser.add_argument("--data-dir", default=str(DATA_DIR / "topic2_broad_base"), help="Directory containing *_daily.csv ETF price files.")
    parser.add_argument("--sentiment-path", default=str(DATA_DIR / "alternative_data" / "google_trends_sentiment.csv"))
    parser.add_argument("--start", default="2024-11-01", help="Walk-forward evaluation start date.")
    parser.add_argument("--end", default="2026-06-03", help="Walk-forward evaluation end date.")
    parser.add_argument("--min-excess", type=float, default=0.0, help="Required cumulative excess return over baseline.")
    parser.add_argument("--max-model-configs", type=int, default=1, help="Maximum model hyper-parameter blocks to evaluate.")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Turnover-proportional trading cost.")
    parser.add_argument("--daily-out", default=str(REPORTS_DIR / "ml_auto_tune_until_baseline.csv"))
    parser.add_argument("--trials-out", default=str(REPORTS_DIR / "ml_auto_tune_trials.csv"))
    parser.add_argument("--summary-out", default=str(REPORTS_DIR / "ml_auto_tune_summary.json"))
    parser.add_argument("--allow-fail", action="store_true", help="Do not exit non-zero if no configuration beats baseline.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = auto_tune_until_baseline(
        data_dir=Path(args.data_dir),
        sentiment_path=Path(args.sentiment_path),
        start=args.start,
        end=args.end,
        min_excess=args.min_excess,
        max_model_configs=args.max_model_configs,
        fee_rate=args.fee_rate,
    )
    write_outputs(
        result,
        daily_out=Path(args.daily_out),
        trials_out=Path(args.trials_out),
        summary_out=Path(args.summary_out),
        start=args.start,
        end=args.end,
        data_dir=Path(args.data_dir),
        sentiment_path=Path(args.sentiment_path),
    )
    metrics = result.metrics
    print(
        json.dumps(
            {
                "success": result.success,
                "strategy_cumulative_return": metrics["strategy_cumulative_return"],
                "benchmark_cumulative_return": metrics["benchmark_cumulative_return"],
                "excess_cumulative_return": metrics["excess_cumulative_return"],
                "model_config": result.model_config.label,
                "strategy_config": result.strategy_config.label,
                "trial_count": int(len(result.trials)),
                "daily_out": args.daily_out,
                "trials_out": args.trials_out,
                "summary_out": args.summary_out,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    if not result.success and not args.allow_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
