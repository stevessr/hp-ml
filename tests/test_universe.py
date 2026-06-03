from hp_ml.lite_train import classify_name, excluded


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


def test_excludes_enhanced_by_default():
    assert excluded("中证 1000 增强 ETF", include_enhanced=False, include_style=False) == "enhanced"
