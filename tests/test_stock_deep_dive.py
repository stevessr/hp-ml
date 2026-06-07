import pandas as pd

from hp_ml.stock_deep_dive import (
    aggregate_holder_rows,
    compute_bullish_score,
    mechanism_tags,
    select_etf_signals,
    write_company_shareholder_reports,
)


def test_select_etf_signals_keeps_positive_and_top_ranked_rows():
    predictions = pd.DataFrame(
        [
            {"code": "510500", "name": "中证500ETF", "family_id": "CSI_500", "pred_fwd_ret_5": 0.01, "pred_score_rank": 2},
            {"code": "510300", "name": "沪深300ETF", "family_id": "CSI_300", "pred_fwd_ret_5": -0.02, "pred_score_rank": 1},
            {"code": "512100", "name": "中证1000ETF", "family_id": "CSI_1000", "pred_fwd_ret_5": -0.03, "pred_score_rank": 3},
        ]
    )

    selected = select_etf_signals(predictions, top_etfs=2, min_etf_pred=0.0)

    assert set(selected["code"]) == {"510500", "510300"}
    assert selected.loc[selected["code"] == "510500", "signal_reason"].iloc[0].startswith("pred>=")
    assert "top2" in selected.loc[selected["code"] == "510300", "signal_reason"].iloc[0]


def test_aggregate_holder_rows_sums_latest_free_and_total_ratios():
    free_rows = [
        {"END_DATE": "2026-03-31 00:00:00", "HOLDER_RANK": 1, "HOLDER_NAME": "个人甲", "HOLDER_NEWTYPE": "个人", "FREE_HOLDNUM_RATIO": 5.0, "HOLD_RATIO": 3.0, "HOLD_RATIO_CHANGE": 0.1},
        {"END_DATE": "2026-03-31 00:00:00", "HOLDER_RANK": 2, "HOLDER_NAME": "某某基金", "HOLDER_NEWTYPE": "基金", "FREE_HOLDNUM_RATIO": 4.0, "HOLD_RATIO": 2.5, "HOLD_RATIO_CHANGE": -0.2},
        {"END_DATE": "2025-12-31 00:00:00", "HOLDER_RANK": 1, "HOLDER_NAME": "旧股东", "HOLDER_NEWTYPE": "基金", "FREE_HOLDNUM_RATIO": 99.0},
    ]
    total_rows = [
        {"END_DATE": "2026-03-31 00:00:00", "RANK": 1, "HOLDER_NAME": "控股集团", "HOLDER_NATURE": "一般企业", "HOLD_RATIO": 25.0, "HOLD_RATIO_CHANGE": 1.0},
        {"END_DATE": "2026-03-31 00:00:00", "RANK": 2, "HOLDER_NAME": "个人乙", "HOLDER_NATURE": "个人", "HOLD_RATIO": 10.0, "HOLD_RATIO_CHANGE": 0.0},
    ]

    summary = aggregate_holder_rows(free_rows, total_rows)

    assert summary["holder_report_date_free"] == "2026-03-31"
    assert summary["top_free_holder_ratio_sum_pct"] == 9.0
    assert summary["top_free_institution_ratio_sum_pct"] == 4.0
    assert summary["top_free_fund_like_ratio_sum_pct"] == 4.0
    assert summary["top_total_holder_ratio_sum_pct"] == 35.0
    assert summary["top_total_institution_ratio_sum_pct"] == 25.0


def test_bullish_score_and_tags_reward_momentum_amount_and_holders():
    base = {
        "source_etf_pred_fwd_ret_5_best": 0.005,
        "source_etf_pred_rank_best": 1,
        "stock_ret_5": 0.02,
        "stock_ret_20": 0.05,
        "history_amount_ratio_20": 2.0,
        "history_amount_z20": 1.5,
        "quote_turnover_rate": 8.0,
        "top_free_holder_ratio_sum_pct": 35.0,
        "top_free_institution_ratio_sum_pct": 12.0,
    }
    weaker = dict(base, stock_ret_5=-0.03, stock_ret_20=-0.01, history_amount_ratio_20=0.7, top_free_holder_ratio_sum_pct=5.0)

    assert compute_bullish_score(base) > compute_bullish_score(weaker)
    tags = mechanism_tags(base)
    assert "价格动量" in tags
    assert "成交额放大" in tags
    assert "高换手交易" in tags
    assert "流通股东集中" in tags


def test_writes_per_company_shareholder_composition_report(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "bullish_rank": 1,
                "bullish_score": 8.5,
                "stock_code": "300059",
                "stock_name": "东方财富",
                "family_ids": "CSI_300",
                "source_etf_codes": "510300",
                "source_etf_names": "沪深300ETF",
                "mechanism_tags": "成交额放大;机构股东参与",
                "industry": "证券",
                "region": "上海市",
                "quote_amount": 1_000_000_000,
                "quote_turnover_rate": 3.2,
                "stock_ret_5": 0.01,
                "stock_ret_20": 0.02,
                "history_amount_ratio_20": 1.8,
                "top_free_holder_ratio_sum_pct": 9.0,
                "top_free_institution_ratio_sum_pct": 4.0,
                "top_free_fund_like_ratio_sum_pct": 4.0,
                "top_free_hold_ratio_change_sum_pct": -0.1,
                "top_free_holder_name": "其实",
                "top_free_holder_type": "个人",
                "top_free_holder_ratio_pct": 5.0,
                "top_total_holder_ratio_sum_pct": 35.0,
                "top_total_institution_ratio_sum_pct": 25.0,
                "top_total_hold_ratio_change_sum_pct": 1.0,
                "top_total_holder_name": "控股集团",
                "top_total_holder_type": "一般企业",
                "top_total_holder_ratio_pct": 25.0,
                "holder_report_date_free": "2026-03-31",
                "holder_report_date_total": "2026-03-31",
            }
        ]
    )
    details = pd.DataFrame(
        [
            {
                "stock_code": "300059",
                "stock_name": "东方财富",
                "holder_kind": "free",
                "rank": 1,
                "holder_name": "其实",
                "holder_type": "个人",
                "hold_ratio_pct": 3.0,
                "free_hold_ratio_pct": 5.0,
            },
            {
                "stock_code": "300059",
                "stock_name": "东方财富",
                "holder_kind": "total",
                "rank": 1,
                "holder_name": "控股集团",
                "holder_type": "一般企业",
                "hold_ratio_pct": 25.0,
            },
        ]
    )

    reports = write_company_shareholder_reports(
        rows=rows,
        details=details,
        out_dir=tmp_path / "reports",
        charts_dir=tmp_path / "charts",
    )

    assert reports[0]["stock_code"] == "INDEX"
    report_path = tmp_path / "reports" / "01_300059_东方财富_shareholder_composition.md"
    assert report_path.exists()
    assert (tmp_path / "charts" / "01_300059_东方财富_free_holders_pie.svg").exists()
    assert (tmp_path / "charts" / "01_300059_东方财富_total_holders_pie.svg").exists()
    text = report_path.read_text(encoding="utf-8")
    assert "股东成分报告" in text
    assert "饼图" in text
    assert "十大流通股东" in text
    assert "控股集团" in text
