"""End-to-end ETF discovery, data pull, feature engineering and model training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .data_sources import fetch_many_histories, today_yyyymmdd
from .features import build_feature_panel
from .model import fit_and_evaluate, predict_latest
from .reporting import make_training_summary, write_json
from .universe import discover_broad_etfs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine CSI broad ETFs and train a forward-return model.")
    parser.add_argument("--start", default="20180101", help="History start date YYYYMMDD.")
    parser.add_argument("--end", default=today_yyyymmdd(), help="History end date YYYYMMDD.")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in trading days.")
    parser.add_argument("--test-days", type=int, default=252, help="Holdout window length in trading days.")
    parser.add_argument("--max-etfs-per-index", type=int, default=3, help="Top ETFs per index family by turnover amount.")
    parser.add_argument("--min-amount", type=float, default=0.0, help="Minimum same-day turnover amount in CNY.")
    parser.add_argument("--model", default="hgb", choices=["hgb", "rf", "ridge"], help="Model type.")
    parser.add_argument("--include-enhanced", action="store_true", help="Include index-enhanced ETFs.")
    parser.add_argument("--include-style", action="store_true", help="Include style/theme ETFs that contain broad-index names.")
    parser.add_argument("--force", action="store_true", help="Refresh spot and history caches.")
    parser.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", "none", "raw"], help="Eastmoney K-line adjustment.")
    parser.add_argument("--model-out", default=str(MODELS_DIR / "csi_broad_etf_model.joblib"), help="Model artifact path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    universe = discover_broad_etfs(
        max_per_family=args.max_etfs_per_index,
        min_amount=args.min_amount,
        include_enhanced=args.include_enhanced,
        include_style=args.include_style,
        force=args.force,
    )
    universe_path = REPORTS_DIR / "latest_candidates.csv"
    universe.to_csv(universe_path, index=False)

    codes = universe["code"].astype(str).str.zfill(6).tolist()
    histories = fetch_many_histories(codes, start=args.start, end=args.end, adjust=args.adjust, force=args.force)
    if not histories:
        raise RuntimeError("No ETF histories fetched; cannot train model")

    panel, feature_cols, target_col = build_feature_panel(histories, universe=universe, horizon=args.horizon)
    panel_path = PROCESSED_DIR / "training_panel.csv"
    panel.to_csv(panel_path, index=False)

    model, metrics, test_predictions, split = fit_and_evaluate(
        panel,
        feature_cols=feature_cols,
        target_col=target_col,
        model_type=args.model,
        test_days=args.test_days,
    )

    latest_predictions = predict_latest(panel, feature_cols=feature_cols, model=model, horizon=args.horizon)
    predictions_path = REPORTS_DIR / "latest_predictions.csv"
    latest_predictions.to_csv(predictions_path, index=False)
    test_predictions.to_csv(REPORTS_DIR / "holdout_predictions.csv", index=False)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "horizon": args.horizon,
        "universe": universe,
        "metrics": metrics,
        "split_date": split.split_date,
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)

    metrics_path = REPORTS_DIR / "training_metrics.json"
    metrics_payload = {
        **metrics,
        "codes": codes,
        "candidate_count": int(len(universe)),
        "history_count": int(len(histories)),
        "panel_rows": int(len(panel)),
        "trainable_rows": int(panel["is_trainable"].sum()),
        "feature_cols": feature_cols,
        "model_path": str(model_out),
        "latest_candidates_path": str(universe_path),
        "latest_predictions_path": str(predictions_path),
        "training_panel_path": str(panel_path),
    }
    write_json(metrics_path, metrics_payload)
    make_training_summary(
        universe=universe,
        metrics=metrics_payload,
        predictions=latest_predictions,
        feature_cols=feature_cols,
        horizon=args.horizon,
        output_path=REPORTS_DIR / "training_summary.md",
    )

    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=str))
    print(f"\n模型已保存: {model_out}")
    print(f"候选池: {universe_path}")
    print(f"最新预测: {predictions_path}")
    print(f"报告: {REPORTS_DIR / 'training_summary.md'}")


if __name__ == "__main__":
    main()
