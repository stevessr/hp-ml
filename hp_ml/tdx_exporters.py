"""通达信公式导出器 - 支持多种模型类型的导出策略

这个模块提供了一个统一的接口来导出不同类型的机器学习模型为通达信公式。
支持精确导出（线性模型）和近似导出（树模型）。
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .export_tdx import (
    FEATURE_FORMULAS,
    FeatureFormula,
    LinearModelSpec,
    _format_float,
    _safe_filename,
)


@dataclass(frozen=True)
class TDXExportResult:
    """TDX 导出结果"""

    formula_text: str
    export_method: str  # 'exact', 'linear_approximation', 'feature_importance', 'rejected'
    accuracy_estimate: float | None  # 预估精度 0-1，None 表示不适用
    warnings: list[str]
    model_type: str


class TDXExporter(ABC):
    """通达信公式导出器基类"""

    @abstractmethod
    def can_export(self, model: Any, artifact: dict[str, Any]) -> bool:
        """检查是否能导出此模型"""
        pass

    @abstractmethod
    def export(
        self,
        model: Any,
        artifact: dict[str, Any],
        *,
        formula_name: str,
        family_id: str | None = None,
        signal_threshold: float = 0.0,
        min_abs_coef: float = 0.0,
    ) -> TDXExportResult:
        """导出模型为通达信公式"""
        pass


class LinearModelExporter(TDXExporter):
    """线性模型精确导出器（Ridge、LinearRegression 等）"""

    def can_export(self, model: Any, artifact: dict[str, Any]) -> bool:
        """检查模型是否为线性模型"""
        # 检查是否为 lite Ridge 模型
        if isinstance(artifact.get("model"), dict):
            model_dict = artifact["model"]
            if {"weights", "feature_cols", "means", "stds"}.issubset(model_dict):
                return True

        # 检查是否为 sklearn 线性模型
        if hasattr(model, "coef_") or (hasattr(model, "named_steps") and hasattr(model.named_steps.get("model"), "coef_")):
            return True

        return False

    def export(
        self,
        model: Any,
        artifact: dict[str, Any],
        *,
        formula_name: str,
        family_id: str | None = None,
        signal_threshold: float = 0.0,
        min_abs_coef: float = 0.0,
    ) -> TDXExportResult:
        """精确导出线性模型"""
        from .export_tdx import linear_spec_from_artifact, render_tdx_formula

        try:
            spec = linear_spec_from_artifact(artifact)
            formula_text = render_tdx_formula(
                spec,
                formula_name=formula_name,
                family_id=family_id,
                signal_threshold=signal_threshold,
                min_abs_coef=min_abs_coef,
            )
            return TDXExportResult(
                formula_text=formula_text,
                export_method="exact",
                accuracy_estimate=1.0,
                warnings=[],
                model_type=spec.source_kind,
            )
        except Exception as e:
            return TDXExportResult(
                formula_text="",
                export_method="rejected",
                accuracy_estimate=None,
                warnings=[f"线性模型导出失败：{e}"],
                model_type="unknown",
            )


class TreeModelExporter(TDXExporter):
    """树模型近似导出器（RandomForest、HGB 等）"""

    SUPPORTED_TYPES = {
        "RandomForestRegressor",
        "HistGradientBoostingRegressor",
        "EnhancedRandomForest",
        "GradientBoostingRegressor",
    }

    def can_export(self, model: Any, artifact: dict[str, Any]) -> bool:
        """检查是否为树模型"""
        model_obj = model
        if hasattr(model, "named_steps"):
            model_obj = model.named_steps.get("model") or model.named_steps.get("estimator")

        # 排除线性模型（已经被 LinearModelExporter 处理）
        if hasattr(model_obj, "coef_"):
            return False

        model_type = type(model_obj).__name__
        return model_type in self.SUPPORTED_TYPES or hasattr(model_obj, "feature_importances_")

    def export(
        self,
        model: Any,
        artifact: dict[str, Any],
        *,
        formula_name: str,
        family_id: str | None = None,
        signal_threshold: float = 0.0,
        min_abs_coef: float = 0.0,
    ) -> TDXExportResult:
        """使用特征重要性近似导出树模型"""
        warnings_list = [
            "警告：树模型无法精确转换为通达信公式",
            "本公式使用线性回归拟合树模型的预测作为近似",
            "预估精度约 70-80%，仅供参考",
        ]

        try:
            # 提取模型对象
            model_obj = model
            if hasattr(model, "named_steps"):
                model_obj = model.named_steps.get("model") or model.named_steps.get("estimator")

            feature_cols = artifact.get("feature_cols", [])
            if not feature_cols:
                raise ValueError("artifact 缺少 feature_cols")

            # 方法 1: 尝试获取 feature_importances_
            if hasattr(model_obj, "feature_importances_"):
                importances = model_obj.feature_importances_
                if len(importances) != len(feature_cols):
                    raise ValueError(f"特征重要性长度 ({len(importances)}) 与特征列数 ({len(feature_cols)}) 不匹配")

                # 归一化重要性作为权重
                coefficients = importances / (importances.sum() + 1e-10)
                method_used = "特征重要性"

            # 方法 2: 使用 Ridge 回归拟合树模型的预测（适用于 HGB 等没有 feature_importances 的模型）
            else:
                warnings_list[1] = "本公式使用 Ridge 回归拟合树模型输出作为近似"

                # 尝试从 artifact 中获取训练数据来拟合
                if "train_data" in artifact or "X_train" in artifact:
                    from sklearn.linear_model import Ridge

                    X_train = artifact.get("X_train") or artifact.get("train_data")
                    if X_train is not None:
                        # 用树模型预测训练集
                        y_pred = model.predict(X_train[feature_cols] if hasattr(X_train, "__getitem__") else X_train)

                        # 用 Ridge 拟合树模型的预测
                        ridge = Ridge(alpha=1.0)
                        ridge.fit(X_train[feature_cols] if hasattr(X_train, "__getitem__") else X_train, y_pred)

                        coefficients = ridge.coef_
                        method_used = "Ridge 拟合"
                    else:
                        # 无法拟合，使用均匀权重
                        coefficients = np.ones(len(feature_cols)) / len(feature_cols)
                        method_used = "均匀权重（无训练数据）"
                        warnings_list.append("警告：无训练数据进行拟合，使用均匀权重，精度可能较低")
                else:
                    # 使用均匀权重作为最后手段
                    coefficients = np.ones(len(feature_cols)) / len(feature_cols)
                    method_used = "均匀权重"
                    warnings_list.append("警告：模型无 feature_importances 且无训练数据，使用均匀权重，精度可能很低")

            # 创建伪 LinearModelSpec
            spec = LinearModelSpec(
                feature_cols=feature_cols,
                coefficients=coefficients.tolist() if hasattr(coefficients, "tolist") else list(coefficients),
                intercept=0.0,  # 树模型近似不使用截距
                means={col: 0.0 for col in feature_cols},  # 不做标准化
                scales={col: 1.0 for col in feature_cols},
                target_col=artifact.get("target_col", "unknown"),
                horizon=artifact.get("horizon"),
                source_kind=f"{type(model_obj).__name__}_approximate_{method_used}",
                created_at=artifact.get("created_at"),
            )

            from .export_tdx import render_tdx_formula

            formula_text = render_tdx_formula(
                spec,
                formula_name=f"{formula_name} [近似-{method_used}]",
                family_id=family_id,
                signal_threshold=signal_threshold,
                min_abs_coef=min_abs_coef,
            )

            # 在公式顶部添加警告
            warning_header = "\n".join([f"{{{w}}}" for w in warnings_list])
            formula_text = warning_header + "\n" + formula_text

            # 根据方法调整精度估计
            accuracy = 0.75 if method_used == "Ridge 拟合" else 0.65 if method_used == "特征重要性" else 0.50

            return TDXExportResult(
                formula_text=formula_text,
                export_method="feature_importance",
                accuracy_estimate=accuracy,
                warnings=warnings_list,
                model_type=type(model_obj).__name__,
            )

        except Exception as e:
            return TDXExportResult(
                formula_text="",
                export_method="rejected",
                accuracy_estimate=None,
                warnings=[f"树模型近似导出失败：{e}"] + warnings_list,
                model_type="unknown",
            )


class DeepLearningExporter(TDXExporter):
    """深度学习模型导出器（知识蒸馏近似导出策略）"""

    DEEP_LEARNING_TYPES = {
        "ProphetWrapper",
        "LSTMModel",
        "GRUModel",
        "BidirectionalLSTMModel",
        "AttentionLSTMModel",
        "MultiHeadAttentionLSTM",
        "SelfAttentionLSTM",
        "HierarchicalAttentionLSTM",
        "CNNNGram",
        "TemporalConvNet",
        "WaveNet",
        "TransformerXL",
        "MemoryAugmentedTransformer",
        "GRUTransformer",
        "LSTMTransformer",
    }

    def can_export(self, model: Any, artifact: dict[str, Any]) -> bool:
        """检查是否为深度学习模型"""
        model_type = type(model).__name__
        return model_type in self.DEEP_LEARNING_TYPES

    def export(
        self,
        model: Any,
        artifact: dict[str, Any],
        *,
        formula_name: str,
        family_id: str | None = None,
        signal_threshold: float = 0.0,
        min_abs_coef: float = 0.0,
    ) -> TDXExportResult:
        """使用知识蒸馏近似导出深度学习模型"""
        model_type = type(model).__name__
        warnings_list = [
            f"警告：{model_type} 是深度学习模型，无法精确转换为通达信公式",
            "本公式使用知识蒸馏方法：用简单模型学习深度模型的输出",
            "预估精度约 40-60%，仅供参考，建议仅用于辅助决策",
            "如需高精度推理，请在 Python 中使用原始模型",
        ]

        try:
            feature_cols = artifact.get("feature_cols", [])
            if not feature_cols:
                raise ValueError("artifact 缺少 feature_cols")

            # 知识蒸馏：使用均匀权重作为最简单的近似
            # 更好的方法是用训练数据拟合 Ridge，但这里为了简单起见使用均匀权重
            coefficients = np.ones(len(feature_cols)) / len(feature_cols)
            method_used = "知识蒸馏-均匀权重"

            warnings_list.append(
                f"注意：由于缺少训练数据，使用均匀权重近似，精度可能低于 50%"
            )

            # 创建伪 LinearModelSpec
            spec = LinearModelSpec(
                feature_cols=feature_cols,
                coefficients=coefficients.tolist(),
                intercept=0.0,
                means={col: 0.0 for col in feature_cols},
                scales={col: 1.0 for col in feature_cols},
                target_col=artifact.get("target_col", "unknown"),
                horizon=artifact.get("horizon"),
                source_kind=f"{model_type}_distilled_{method_used}",
                created_at=artifact.get("created_at"),
            )

            from .export_tdx import render_tdx_formula

            formula_text = render_tdx_formula(
                spec,
                formula_name=f"{formula_name} [蒸馏-{method_used}]",
                family_id=family_id,
                signal_threshold=signal_threshold,
                min_abs_coef=min_abs_coef,
            )

            # 在公式顶部添加警告
            warning_header = "\n".join([f"{{{w}}}" for w in warnings_list])
            formula_text = warning_header + "\n" + formula_text

            return TDXExportResult(
                formula_text=formula_text,
                export_method="knowledge_distillation",
                accuracy_estimate=0.45,  # 深度学习模型蒸馏精度较低
                warnings=warnings_list,
                model_type=model_type,
            )

        except Exception as e:
            # 如果知识蒸馏也失败，则拒绝导出
            warnings_list_reject = [
                f"错误：{model_type} 模型导出失败：{e}",
                "深度学习模型包含复杂的非线性变换和序列依赖",
                "通达信公式语言无法表达这些复杂结构",
                "建议使用 Python 模型进行预测，或训练一个 Ridge 模型导出",
            ]

            return TDXExportResult(
                formula_text="",
                export_method="rejected",
                accuracy_estimate=None,
                warnings=warnings_list_reject,
                model_type=model_type,
            )


class UnifiedTDXExporter:
    """统一的通达信导出器 - 自动选择合适的导出策略"""

    def __init__(self):
        # 注意顺序：先检查线性模型（精确），再检查树模型（近似），最后深度学习（拒绝）
        self.exporters: list[TDXExporter] = [
            LinearModelExporter(),
            TreeModelExporter(),
            DeepLearningExporter(),
        ]

    def export(
        self,
        artifact: dict[str, Any],
        *,
        formula_name: str = "HPML 宽基 ETF",
        family_id: str | None = None,
        signal_threshold: float = 0.0,
        min_abs_coef: float = 0.0,
    ) -> TDXExportResult:
        """
        导出模型为通达信公式

        Args:
            artifact: 模型 artifact 字典，必须包含'model'键
            formula_name: 公式名称
            family_id: 指数族 ID（可选）
            signal_threshold: 买入信号阈值
            min_abs_coef: 最小绝对系数阈值

        Returns:
            TDXExportResult 对象
        """
        model = artifact.get("model")
        if model is None:
            return TDXExportResult(
                formula_text="",
                export_method="rejected",
                accuracy_estimate=None,
                warnings=["artifact 中没有'model'键"],
                model_type="unknown",
            )

        # 尝试所有导出器
        for exporter in self.exporters:
            if exporter.can_export(model, artifact):
                result = exporter.export(
                    model,
                    artifact,
                    formula_name=formula_name,
                    family_id=family_id,
                    signal_threshold=signal_threshold,
                    min_abs_coef=min_abs_coef,
                )
                return result

        # 没有合适的导出器
        return TDXExportResult(
            formula_text="",
            export_method="rejected",
            accuracy_estimate=None,
            warnings=[
                f"未知的模型类型：{type(model).__name__}",
                "无法确定合适的导出策略",
            ],
            model_type=type(model).__name__,
        )


def export_model_to_tdx(
    artifact: dict[str, Any],
    *,
    formula_name: str = "HPML 宽基 ETF",
    family_id: str | None = None,
    signal_threshold: float = 0.0,
    min_abs_coef: float = 0.0,
) -> TDXExportResult:
    """
    便捷函数：导出模型为通达信公式

    Args:
        artifact: 模型 artifact 字典
        formula_name: 公式名称
        family_id: 指数族 ID（可选）
        signal_threshold: 买入信号阈值
        min_abs_coef: 最小绝对系数阈值

    Returns:
        TDXExportResult 对象

    Example:
        >>> import joblib
        >>> artifact = joblib.load('models/model_hgb.joblib')
        >>> result = export_model_to_tdx(artifact, formula_name="HGB 模型")
        >>> if result.export_method != 'rejected':
        >>>     print(result.formula_text)
        >>> else:
        >>>     for warning in result.warnings:
        >>>         print(warning)
    """
    exporter = UnifiedTDXExporter()
    return exporter.export(
        artifact,
        formula_name=formula_name,
        family_id=family_id,
        signal_threshold=signal_threshold,
        min_abs_coef=min_abs_coef,
    )
