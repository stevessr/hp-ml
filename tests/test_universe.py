from hp_ml.lite_train import classify_name, excluded


def test_classifies_core_broad_names():
    assert classify_name("沪深300ETF华泰柏瑞")[0] == "CSI_300"
    assert classify_name("中证500ETF南方")[0] == "CSI_500"
    assert classify_name("中证1000ETF华夏")[0] == "CSI_1000"
    assert classify_name("中证A500ETF易方达")[0] == "CSI_A500"


def test_avoids_non_csi_generic_names():
    assert classify_name("国证2000ETF平安")[0] is None
    assert classify_name("深证100ETF富国")[0] is None


def test_excludes_style_or_theme_by_default():
    assert "红利" in (excluded("中证A500红利低波ETF", include_enhanced=False, include_style=False) or "")
    assert "科创" in (excluded("科创50ETF", include_enhanced=False, include_style=False) or "")


def test_excludes_enhanced_by_default():
    assert excluded("中证1000增强ETF", include_enhanced=False, include_style=False) == "enhanced"
