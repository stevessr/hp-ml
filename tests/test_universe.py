from hp_ml.lite_train import build_multi_scale_analysis, classify_name, excluded


def test_classifies_core_broad_names():
    assert classify_name("沪深 300ETF 华泰柏瑞")[0] == "CSI_300"
    assert classify_name("中证 500ETF 南方")[0] == "CSI_500"
    assert classify_name("中证 1000ETF 华夏")[0] == "CSI_1000"
    assert classify_name("中证 A500ETF 易方达")[0] == "CSI_A500"


def test_avoids_non_csi_generic_names():
    assert classify_name("国证 2000ETF 平安")[0] is None
    assert classify_name("深证 100ETF 富国")[0] is None


def test_excludes_style_or_theme_by_default():
    assert "红利" in (excluded("中证 A500 红利低波 ETF", include_enhanced=False, include_style=False) or "")
    assert "科创" in (excluded("科创 50ETF", include_enhanced=False, include_style=False) or "")
    assert "现金流" in (excluded("自由现金流 800ETF", include_enhanced=False, include_style=False) or "")


def test_excludes_enhanced_by_default():
    assert excluded("中证 1000 增强 ETF", include_enhanced=False, include_style=False) == "enhanced"


def test_build_multi_scale_analysis_summarizes_horizons():
    histories = {
        "510300": [
            {"date": "2024-01-01", "close": 1.0},
            {"date": "2024-01-02", "close": 1.1},
            {"date": "2024-01-03", "close": 1.2},
            {"date": "2024-01-04", "close": 1.0},
        ]
    }
    universe = [{"code": "510300", "name": "沪深300ETF", "family_id": "CSI_300"}]

    result = build_multi_scale_analysis(histories, universe, [1, 2])

    assert result["coverage"][0]["rows"] == 4
    assert len(result["etf_metrics"]) == 2
    all_rows = {row["horizon_days"]: row for row in result["family_metrics"] if row["family_id"] == "ALL"}
    assert all_rows[1]["observations"] == 3
    assert all_rows[2]["observations"] == 2
