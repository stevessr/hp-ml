"""Reporting helpers for the ETF modelling pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data_sources import human_amount


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def format_pct(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
    except Exception:  # noqa: BLE001
        return ""
    return f"{value * 100:.{digits}f}%"


def make_training_summary(
    *,
    universe: pd.DataFrame,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    output_path: Path,
    chart_paths: dict[str, str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    families = universe.groupby("family_id")["code"].count().sort_index()
    top_candidates = universe.sort_values("amount", ascending=False).head(10).copy() if "amount" in universe else universe.head(10).copy()
    top_predictions = predictions.head(10).copy()

    lines: list[str] = []
    lines.append("# 中证宽基 ETF 自动挖掘与模型训练报告")
    lines.append("")
    lines.append("本报告由 `python -m hp_ml.train` 自动生成。模型输出用于量化研究和候选排序，不构成投资建议。")
    lines.append("")
    lines.append("## 1. 自动发现的宽基 ETF 池")
    lines.append("")
    lines.append(f"- 候选 ETF 数量：{len(universe)}")
    lines.append("- 覆盖指数族：" + ", ".join(f"{idx}({count})" for idx, count in families.items()))
    lines.append("- 发现方式：扫描东方财富 ETF 行情/基金代码列表，用名称模式识别中证/沪深宽基指数，并默认剔除行业、主题、风格、跨境、债券、货币和指数增强产品。")
    lines.append("")
    lines.append("### 成交额靠前候选")
    lines.append("")
    lines.append("|代码|名称|指数族|最新价|成交额|族内排名|")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, row in top_candidates.iterrows():
        lines.append(
            f"|{row.get('code', '')}|{row.get('name', '')}|{row.get('family_id', '')}|"
            f"{row.get('latest_price', '')}|{human_amount(row.get('amount'))}|{row.get('selected_rank_in_family', '')}|"
        )

    lines.append("")
    lines.append("## 2. 特征与监督学习目标")
    lines.append("")
    lines.append(f"- 预测目标：未来 {horizon} 个交易日 ETF 前复权收盘收益率。")
    lines.append("- 特征主题：短/中期动量、波动率、均线乖离、回撤、成交额冲击、换手、振幅、月份周期和指数族哑变量。")
    lines.append(f"- 特征数量：{len(feature_cols)}")
    lines.append("")
    lines.append("## 3. 时间切分验证指标")
    lines.append("")
    lines.append("|指标|值|")
    lines.append("|---|---:|")
    for key in [
        "model_type",
        "train_rows",
        "test_rows",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "mae",
        "rmse",
        "r2",
        "directional_accuracy",
        "spearman_ic_by_date",
        "top1_mean_fwd_ret",
        "top_vs_all",
        "top_bottom_spread",
    ]:
        value = metrics.get(key)
        if isinstance(value, float):
            value_text = f"{value:.6f}"
        else:
            value_text = str(value)
        lines.append(f"|{key}|{value_text}|")

    lines.append("")
    lines.append("## 4. 图表结果")
    lines.append("")
    if chart_paths:
        for title, key in [
            ("最新预测排序图", "latest_prediction_rank"),
            ("候选指数族覆盖图", "candidate_family_counts"),
            ("验证指标快照图", "holdout_metric_snapshot"),
            ("Holdout累计曲线", "holdout_cumulative"),
        ]:
            path_text = chart_paths.get(key)
            if not path_text:
                continue
            try:
                md_path = str(Path(path_text).relative_to(output_path.parent))
            except ValueError:
                md_path = str(path_text)
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}]({md_path})")
            lines.append("")
    else:
        lines.append("- 本次未生成图表。")
        lines.append("")

    lines.append("## 5. 最新候选排序")
    lines.append("")
    lines.append("|排名|日期|代码|名称|指数族|收盘价|预测未来收益|近5日|近20日|20日波动|20日回撤|")
    lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    pred_col = f"pred_fwd_ret_{horizon}"
    for _, row in top_predictions.iterrows():
        lines.append(
            f"|{int(row.get('pred_score_rank', 0))}|{pd.to_datetime(row.get('date')).date()}|"
            f"{row.get('code', '')}|{row.get('name', '')}|{row.get('family_id', '')}|{row.get('close', '')}|"
            f"{format_pct(row.get(pred_col))}|{format_pct(row.get('ret_5'))}|"
            f"{format_pct(row.get('ret_20'))}|{format_pct(row.get('vol_20'))}|{format_pct(row.get('drawdown_20'))}|"
        )

    lines.append("")
    lines.append("## 5. 下一步可扩展")
    lines.append("")
    lines.append("- 加入基金规模、费率、跟踪误差、申赎清单等 ETF 微观指标。")
    lines.append("- 将未来收益目标改成相对指数族均值的超额收益，减少市场 Beta 对排序的干扰。")
    lines.append("- 增加组合约束、交易成本、调仓频率和 walk-forward 回测。")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
