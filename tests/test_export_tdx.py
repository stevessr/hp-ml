import pickle

import pytest

from hp_ml.export_tdx import linear_spec_from_artifact, main, render_tdx_formula


def sample_lite_artifact():
    return {
        "created_at": "2026-06-05T12:00:00",
        "model": {
            "feature_cols": ["ret_1", "amount_z20", "family_CSI_300", "family_CSI_500"],
            "means": {"ret_1": 0.01, "amount_z20": 0.2, "family_CSI_300": 0.1, "family_CSI_500": 0.2},
            "stds": {"ret_1": 0.03, "amount_z20": 1.5, "family_CSI_300": 0.3, "family_CSI_500": 0.4},
            "weights": [0.001, 0.2, -0.3, 0.4, -0.5],
            "target_col": "fwd_ret_5",
            "l2": 3.0,
        },
    }


def test_renders_lite_ridge_formula_with_selected_family():
    spec = linear_spec_from_artifact(sample_lite_artifact())

    formula = render_tdx_formula(spec, formula_name="HPML_TEST", family_id="CSI_300", signal_threshold=0.01)

    assert "HPML_TEST - hp-ml 通达信公式导出" in formula
    assert "R1:=CLOSE/REF(CLOSE,1)-1;" in formula
    assert "AMTZ20:=(AMOUNT-MA(AMOUNT,20))/MAX(STD(AMOUNT,20),0.000001);" in formula
    assert "FAM1:=1;" in formula
    assert "FAM2:=0;" in formula
    assert "HPMLSCORE:S4,COLORWHITE;" in formula
    assert "HPMLBUY:IF(S4>0.01,1,0),COLORRED;" in formula


def test_rejects_unknown_family():
    spec = linear_spec_from_artifact(sample_lite_artifact())

    with pytest.raises(ValueError, match="Unknown family_id"):
        render_tdx_formula(spec, formula_name="HPML_TEST", family_id="CSI_1000")


def test_cli_exports_one_file_per_family(tmp_path):
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(sample_lite_artifact(), fh)

    out_dir = tmp_path / "tdx"
    main(["--model", str(model_path), "--out", str(out_dir), "--all-families", "--formula-name", "HPML_TEST"])

    files = sorted(path.name for path in out_dir.glob("*.tdx"))
    assert files == ["HPML_TEST_CSI_300.tdx", "HPML_TEST_CSI_500.tdx"]
    csi300 = (out_dir / "HPML_TEST_CSI_300.tdx").read_text(encoding="utf-8")
    csi500 = (out_dir / "HPML_TEST_CSI_500.tdx").read_text(encoding="utf-8")
    assert "FAM1:=1;" in csi300
    assert "FAM2:=1;" in csi500
