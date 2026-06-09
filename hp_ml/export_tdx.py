"""Export trained linear hp-ml models as TongDaXin formula text.

TongDaXin formulas can only express deterministic indicator arithmetic, so this
exporter supports the stdlib Ridge model saved by ``hp_ml.lite_train`` and
linear scikit-learn pipelines such as ``hp_ml.train --model ridge``.  Tree and
gradient-boosting models are intentionally rejected instead of silently
exporting a misleading approximation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import MODELS_DIR, REPORTS_DIR


@dataclass(frozen=True)
class FeatureFormula:
    """TongDaXin expression for a model feature."""

    var_name: str
    expression: str
    note: str


@dataclass(frozen=True)
class LinearModelSpec:
    """Model coefficients in the same standardized feature space used by TDX."""

    feature_cols: list[str]
    coefficients: list[float]
    intercept: float
    means: dict[str, float]
    scales: dict[str, float]
    target_col: str
    horizon: int | None
    source_kind: str
    created_at: str | None = None

    @property
    def family_ids(self) -> list[str]:
        return sorted(col.removeprefix("family_") for col in self.feature_cols if col.startswith("family_"))


FEATURE_FORMULAS: dict[str, FeatureFormula] = {
    "ret_1": FeatureFormula("R1", "CLOSE/REF(CLOSE,1)-1", "1 日收益"),
    "ret_3": FeatureFormula("R3", "CLOSE/REF(CLOSE,3)-1", "3 日收益"),
    "ret_5": FeatureFormula("R5", "CLOSE/REF(CLOSE,5)-1", "5 日收益"),
    "ret_10": FeatureFormula("R10", "CLOSE/REF(CLOSE,10)-1", "10 日收益"),
    "ret_20": FeatureFormula("R20", "CLOSE/REF(CLOSE,20)-1", "20 日收益"),
    "ret_60": FeatureFormula("R60", "CLOSE/REF(CLOSE,60)-1", "60 日收益"),
    "vol_5": FeatureFormula("SD5", "STD(CLOSE/REF(CLOSE,1)-1,5)", "5 日收益波动率"),
    "vol_20": FeatureFormula("SD20", "STD(CLOSE/REF(CLOSE,1)-1,20)", "20 日收益波动率"),
    "vol_60": FeatureFormula("SD60", "STD(CLOSE/REF(CLOSE,1)-1,60)", "60 日收益波动率"),
    "ma_gap_5_20": FeatureFormula("MA520", "MA(CLOSE,5)/MA(CLOSE,20)-1", "5/20 日均线乖离"),
    "ma_gap_20_60": FeatureFormula("MA2060", "MA(CLOSE,20)/MA(CLOSE,60)-1", "20/60 日均线乖离"),
    "drawdown_20": FeatureFormula("DD20", "CLOSE/HHV(CLOSE,20)-1", "20 日回撤"),
    "drawdown_60": FeatureFormula("DD60", "CLOSE/HHV(CLOSE,60)-1", "60 日回撤"),
    "amount_log": FeatureFormula("AMTLOG", "LN(MAX(AMOUNT,0)+1)", "成交额 log1p"),
    "amount_z20": FeatureFormula("AMTZ20", "(AMOUNT-MA(AMOUNT,20))/MAX(STD(AMOUNT,20),0.000001)", "20 日成交额 z 分数"),
    "turnover_rate": FeatureFormula("TURN", "VOL/MAX(CAPITAL,1)*100", "换手率近似"),
    "amplitude": FeatureFormula("AMPL", "(HIGH-LOW)/MAX(REF(CLOSE,1),0.000001)*100", "振幅百分比"),
    "intraday_range": FeatureFormula("IRANGE", "(HIGH-LOW)/MAX(CLOSE,0.000001)", "日内振幅"),
    "volume_chg_5": FeatureFormula("VCHG5", "VOL/REF(VOL,5)-1", "5 日成交量变化"),
    "liquidity_shock_20": FeatureFormula("LS20", "AMOUNT/MAX(MA(AMOUNT,20),0.000001)-1", "20 日成交额冲击"),
    "month_sin": FeatureFormula("MSIN", "SIN(2*3.1415926*MONTH/12)", "月份 sin 周期"),
    "month_cos": FeatureFormula("MCOS", "COS(2*3.1415926*MONTH/12)", "月份 cos 周期"),
    "days_since_start": FeatureFormula("DAYS", "BARSCOUNT(CLOSE)-1", "上市以来交易日数近似"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained linear hp-ml model to TongDaXin formula text.")
    parser.add_argument(
        "--model",
        default=str(MODELS_DIR / "csi_broad_etf_model_lite.pkl"),
        help="Path to a hp-ml model artifact. Supports lite Ridge .pkl and sklearn Ridge .joblib.",
    )
    parser.add_argument(
        "--out",
        default=str(REPORTS_DIR / "hp_ml_tdx_formula.tdx"),
        help="Output .tdx/.txt path, or a directory when --all-families is used.",
    )
    parser.add_argument("--formula-name", default="HPML 宽基 ETF", help="Formula display name used in comments and file stems.")
    family = parser.add_mutually_exclusive_group()
    family.add_argument(
        "--family-id",
        help="Set one family_* dummy to 1, e.g. CSI_300. Without this all family dummies are exported as 0.",
    )
    family.add_argument(
        "--all-families",
        action="store_true",
        help="Write one formula file per family_* dummy found in the model.",
    )
    parser.add_argument("--signal-threshold", type=float, default=0.0, help="HPMLBUY threshold for HPMLSCORE.")
    parser.add_argument(
        "--min-abs-coef",
        type=float,
        default=0.0,
        help="Drop terms with absolute coefficient below this value to simplify the formula.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Output text encoding. Use gbk if an older TongDaXin import dialog shows garbled Chinese.",
    )
    return parser


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return default
    return out if math.isfinite(out) else default


def _as_float_list(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [_as_float(value) for value in values]


def _format_float(value: float) -> str:
    value = _as_float(value)
    if value == 0:
        return "0"
    text = f"{value:.12f}" if abs(value) < 0.0001 else f"{value:.10f}"
    text = text.rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def _safe_filename(text: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", text).strip("_")
    return safe or "tdx_formula"


def _extract_horizon(target_col: str, artifact: dict[str, Any]) -> int | None:
    horizon = artifact.get("horizon")
    if horizon is not None:
        try:
            return int(horizon)
        except Exception:  # noqa: BLE001
            pass
    match = re.search(r"fwd_ret_(\d+)", str(target_col))
    return int(match.group(1)) if match else None


def load_artifact(path: Path) -> dict[str, Any]:
    """Load a hp-ml model artifact with pickle first and joblib as fallback."""

    errors: list[str] = []
    try:
        with path.open("rb") as fh:
            artifact = pickle.load(fh)
        if isinstance(artifact, dict):
            return artifact
        errors.append(f"pickle returned {type(artifact).__name__}, not dict")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pickle: {exc}")

    try:
        import joblib  # type: ignore

        artifact = joblib.load(path)
        if isinstance(artifact, dict):
            return artifact
        errors.append(f"joblib returned {type(artifact).__name__}, not dict")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"joblib: {exc}")
    raise RuntimeError(f"Cannot load model artifact {path}: {'; '.join(errors)}")


def _linear_spec_from_lite(artifact: dict[str, Any], model: dict[str, Any]) -> LinearModelSpec:
    weights = _as_float_list(model.get("weights", []))
    feature_cols = [str(col) for col in model.get("feature_cols", [])]
    if not weights or len(weights) != len(feature_cols) + 1:
        raise ValueError("Lite model artifact has inconsistent weights and feature_cols.")
    target_col = str(model.get("target_col") or artifact.get("target_col") or "")
    if not target_col:
        raise ValueError("Lite model artifact is missing target_col.")
    return LinearModelSpec(
        feature_cols=feature_cols,
        coefficients=weights[1:],
        intercept=weights[0],
        means={str(k): _as_float(v) for k, v in dict(model.get("means", {})).items()},
        scales={str(k): _as_float(v, 1.0) or 1.0 for k, v in dict(model.get("stds", {})).items()},
        target_col=target_col,
        horizon=_extract_horizon(target_col, artifact),
        source_kind="lite_stdlib_ridge",
        created_at=artifact.get("created_at"),
    )


def _linear_spec_from_sklearn(artifact: dict[str, Any], pipeline: Any) -> LinearModelSpec:
    feature_cols = [str(col) for col in artifact.get("feature_cols", [])]
    if not feature_cols:
        raise ValueError("Sklearn model artifact is missing feature_cols.")

    estimator = pipeline
    scaler = None
    if hasattr(pipeline, "named_steps"):
        steps = pipeline.named_steps
        estimator = steps.get("model") or steps.get("estimator") or list(steps.values())[-1]
        scaler = steps.get("scaler")
    if not hasattr(estimator, "coef_"):
        raise ValueError(
            "Only linear/Ridge sklearn models can be exported to TongDaXin. "
            "Retrain with `python -m hp_ml.train --model ridge` or use the lite Ridge model."
        )

    coefficients = _as_float_list(estimator.coef_)
    if len(coefficients) != len(feature_cols):
        raise ValueError("Sklearn estimator coef_ length does not match feature_cols.")
    intercept = _as_float(getattr(estimator, "intercept_", 0.0))
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean_values = _as_float_list(scaler.mean_)
        scale_values = [_as_float(value, 1.0) or 1.0 for value in _as_float_list(scaler.scale_)]
    else:
        mean_values = [0.0] * len(feature_cols)
        scale_values = [1.0] * len(feature_cols)

    target_col = str(artifact.get("target_col") or "")
    if not target_col:
        raise ValueError("Sklearn model artifact is missing target_col.")
    return LinearModelSpec(
        feature_cols=feature_cols,
        coefficients=coefficients,
        intercept=intercept,
        means=dict(zip(feature_cols, mean_values, strict=True)),
        scales=dict(zip(feature_cols, scale_values, strict=True)),
        target_col=target_col,
        horizon=_extract_horizon(target_col, artifact),
        source_kind=f"sklearn_{estimator.__class__.__name__}",
        created_at=artifact.get("created_at"),
    )


def linear_spec_from_artifact(artifact: dict[str, Any]) -> LinearModelSpec:
    """Convert a supported hp-ml artifact to a linear model spec."""

    model = artifact.get("model")
    if isinstance(model, dict) and {"weights", "feature_cols", "means", "stds"}.issubset(model):
        return _linear_spec_from_lite(artifact, model)
    if model is not None:
        return _linear_spec_from_sklearn(artifact, model)
    raise ValueError("Model artifact does not contain a supported `model` object.")


def _feature_to_tdx(feature: str, family_id: str | None, family_var_names: dict[str, str]) -> FeatureFormula:
    if feature in FEATURE_FORMULAS:
        return FEATURE_FORMULAS[feature]
    if feature.startswith("family_"):
        fam = feature.removeprefix("family_")
        value = "1" if family_id and fam == family_id else "0"
        return FeatureFormula(family_var_names[feature], value, f"{feature} 常量")
    raise ValueError(f"No TongDaXin expression is defined for feature {feature!r}.")


def render_tdx_formula(
    spec: LinearModelSpec,
    *,
    formula_name: str,
    family_id: str | None = None,
    signal_threshold: float = 0.0,
    min_abs_coef: float = 0.0,
) -> str:
    """Render a TongDaXin formula text for one optional index family."""

    if family_id is not None and family_id not in spec.family_ids:
        raise ValueError(f"Unknown family_id={family_id!r}; available families: {', '.join(spec.family_ids) or '(none)'}")

    family_var_names = {
        feature: f"FAM{idx}"
        for idx, feature in enumerate([col for col in spec.feature_cols if col.startswith("family_")], start=1)
    }

    used_features: list[tuple[str, FeatureFormula, float]] = []
    unsupported: list[str] = []
    for feature, coef in zip(spec.feature_cols, spec.coefficients, strict=True):
        if abs(coef) < min_abs_coef:
            continue
        try:
            used_features.append((feature, _feature_to_tdx(feature, family_id, family_var_names), coef))
        except ValueError:
            unsupported.append(feature)
    if unsupported:
        raise ValueError("Unsupported features for TongDaXin export: " + ", ".join(unsupported))

    horizon_text = f"{spec.horizon}日" if spec.horizon is not None else spec.target_col
    lines = [
        f"{{{formula_name} - hp-ml 通达信公式导出}}",
        f"{{生成时间：{dt.datetime.now().isoformat(timespec='seconds')}}}",
        f"{{模型：{spec.source_kind}; 目标：预测未来{horizon_text}收益; 源模型时间：{spec.created_at or 'unknown'}}}",
        "{说明：仅线性/Ridge 模型可精确展开; 输出仅作量化研究，不构成投资建议。}",
    ]
    if spec.family_ids:
        selected = family_id or "未指定，全部 family 常量为 0"
        lines.append(f"{{指数族：{selected}; 可用：{', '.join(spec.family_ids)}}}")
    lines.extend(
        [
            "{近似：TURN 使用 VOL/CAPITAL*100; DAYS 使用 BARSCOUNT(CLOSE)-1 近似原始自然日特征。}",
            "",
        ]
    )

    emitted_vars: set[str] = set()
    for feature, formula, _ in used_features:
        if formula.var_name in emitted_vars:
            continue
        emitted_vars.add(formula.var_name)
        if feature.startswith("family_"):
            lines.append(f"{formula.var_name}:={formula.expression}; {{{formula.note}, family_id={family_id or 'none'}}}")
        else:
            lines.append(f"{formula.var_name}:={formula.expression}; {{{formula.note}}}")

    lines.append("")
    lines.append(f"S0:={_format_float(spec.intercept)}; {{截距}}")
    score_var = "S0"
    score_idx = 0
    for feature, formula, coef in used_features:
        mean = spec.means.get(feature, 0.0)
        scale = spec.scales.get(feature, 1.0) or 1.0
        score_idx += 1
        next_var = f"S{score_idx}"
        lines.append(
            f"{next_var}:={score_var}+({_format_float(coef)})*"
            f"(({formula.var_name})-({_format_float(mean)}))/({_format_float(scale)}); {{{feature}}}"
        )
        score_var = next_var

    lines.extend(
        [
            "",
            f"HPMLSCORE:{score_var},COLORWHITE;",
            f"HPMLBUY:IF({score_var}>{_format_float(signal_threshold)},1,0),COLORRED;",
            "",
        ]
    )
    return "\n".join(lines)


def write_formula(path: Path, text: str, encoding: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)
    return path


def export_formula_files(
    spec: LinearModelSpec,
    *,
    out: Path,
    formula_name: str,
    family_id: str | None,
    all_families: bool,
    signal_threshold: float,
    min_abs_coef: float,
    encoding: str,
) -> list[Path]:
    if all_families:
        if not spec.family_ids:
            raise ValueError("--all-families was requested, but the model has no family_* features.")
        paths: list[Path] = []
        for fam in spec.family_ids:
            text = render_tdx_formula(
                spec,
                formula_name=f"{formula_name}_{fam}",
                family_id=fam,
                signal_threshold=signal_threshold,
                min_abs_coef=min_abs_coef,
            )
            if out.suffix:
                target = out.with_name(f"{out.stem}_{_safe_filename(fam)}{out.suffix}")
            else:
                target = out / f"{_safe_filename(formula_name)}_{_safe_filename(fam)}.tdx"
            paths.append(write_formula(target, text, encoding))
        return paths

    text = render_tdx_formula(
        spec,
        formula_name=formula_name,
        family_id=family_id,
        signal_threshold=signal_threshold,
        min_abs_coef=min_abs_coef,
    )
    return [write_formula(out, text, encoding)]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifact = load_artifact(Path(args.model))

    # 尝试使用统一导出器
    try:
        from .tdx_exporters import UnifiedTDXExporter

        unified_exporter = UnifiedTDXExporter()

        # 检测模型类型
        model = artifact.get("model")
        if model is not None:
            model_type_name = type(model).__name__
            if hasattr(model, "named_steps"):
                inner_model = model.named_steps.get("model") or model.named_steps.get("estimator")
                if inner_model:
                    model_type_name = type(inner_model).__name__

            print(f"检测到模型类型：{model_type_name}")

        # 如果需要导出所有家族
        if args.all_families:
            # 先用线性导出获取 family 列表
            try:
                spec = linear_spec_from_artifact(artifact)
                family_ids = spec.family_ids
            except Exception:
                # 如果不是线性模型，尝试从 feature_cols 提取
                feature_cols = artifact.get("feature_cols", [])
                family_ids = sorted(col.removeprefix("family_") for col in feature_cols if col.startswith("family_"))

            if not family_ids:
                print("错误：模型没有 family_* 特征，无法使用 --all-families")
                return

            paths: list[Path] = []
            for fam in family_ids:
                result = unified_exporter.export(
                    artifact,
                    formula_name=f"{args.formula_name}_{fam}",
                    family_id=fam,
                    signal_threshold=args.signal_threshold,
                    min_abs_coef=max(args.min_abs_coef, 0.0),
                )

                if result.export_method == "rejected":
                    print(f"\n{fam} - 导出失败：")
                    for warning in result.warnings:
                        print(f"  {warning}")
                    continue

                # 写入文件
                out_path = Path(args.out)
                if out_path.suffix:
                    target = out_path.with_name(f"{out_path.stem}_{_safe_filename(fam)}{out_path.suffix}")
                else:
                    target = out_path / f"{_safe_filename(args.formula_name)}_{_safe_filename(fam)}.tdx"

                target = write_formula(target, result.formula_text, args.encoding)
                paths.append(target)

                print(f"\n{fam} - 导出成功 ({result.export_method}, 预估精度：{result.accuracy_estimate or 'N/A'})")
                if result.warnings:
                    for warning in result.warnings:
                        print(f"  {warning}")

            if paths:
                print(f"\n通达信公式已导出到：")
                for path in paths:
                    print(f"  {path}")
            return

        # 单个 family 或无 family 导出
        result = unified_exporter.export(
            artifact,
            formula_name=args.formula_name,
            family_id=args.family_id,
            signal_threshold=args.signal_threshold,
            min_abs_coef=max(args.min_abs_coef, 0.0),
        )

        if result.export_method == "rejected":
            print("导出失败：")
            for warning in result.warnings:
                print(f"  {warning}")
            return

        # 写入文件
        out_path = Path(args.out)
        out_path = write_formula(out_path, result.formula_text, args.encoding)

        print(f"通达信公式已导出：{out_path}")
        print(f"导出方式：{result.export_method}")
        print(f"模型类型：{result.model_type}")
        if result.accuracy_estimate is not None:
            print(f"预估精度：{result.accuracy_estimate:.1%}")

        if result.warnings:
            print("\n注意事项：")
            for warning in result.warnings:
                print(f"  {warning}")

        # 提示 family 选项
        feature_cols = artifact.get("feature_cols", [])
        family_features = [col for col in feature_cols if col.startswith("family_")]
        if family_features and not args.family_id:
            family_ids = sorted(col.removeprefix("family_") for col in family_features)
            print("\n提示：模型包含 family_* 特征，本次未指定 --family-id，公式中这些特征均为 0。")
            print(f"      可用 --family-id {family_ids[0]} 生成单指数族公式，或 --all-families 批量导出。")

    except ImportError:
        # 回退到原有的线性模型导出逻辑
        print("使用传统线性模型导出...")
        spec = linear_spec_from_artifact(artifact)
        paths = export_formula_files(
            spec,
            out=Path(args.out),
            formula_name=args.formula_name,
            family_id=args.family_id,
            all_families=args.all_families,
            signal_threshold=args.signal_threshold,
            min_abs_coef=max(args.min_abs_coef, 0.0),
            encoding=args.encoding,
        )

        for path in paths:
            print(f"通达信公式已导出：{path}")
        if spec.family_ids and not args.family_id and not args.all_families:
            print("提示：模型包含 family_* 特征，本次未指定 --family-id，公式中这些特征均为 0。")
            print("      可用 --family-id CSI_300 生成单指数族公式，或 --all-families 批量导出。")


if __name__ == "__main__":
    main()
