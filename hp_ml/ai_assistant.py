"""AI 辅助模型进化系统：集成 Claude API"""
from __future__ import annotations

import json
import os
import warnings
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AIAssistedEvolution:
    """AI 辅助模型进化系统"""

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Args:
            api_key: Anthropic API key (如果为 None，从环境变量读取)
            model: Claude 模型版本
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic SDK 未安装，请运行：pip install anthropic")

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("请提供 API key 或设置 ANTHROPIC_API_KEY 环境变量")

        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        self.conversation_history: list[dict[str, str]] = []

    def suggest_features(
        self,
        existing_features: list[str],
        data_description: str,
        target_description: str,
        performance_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        让 AI 建议新特征

        Args:
            existing_features: 现有特征列表
            data_description: 数据描述
            target_description: 目标变量描述
            performance_metrics: 当前模型性能指标

        Returns:
            AI 建议的特征字典
        """
        prompt = f"""你是一个量化金融和机器学习专家。请分析以下信息并建议新的特征工程方案。

## 当前特征
{', '.join(existing_features)}

## 数据描述
{data_description}

## 预测目标
{target_description}

## 当前性能指标
{json.dumps(performance_metrics, indent=2) if performance_metrics else "尚未训练"}

请提供：
1. 建议新增的 5-10 个特征及其计算方法
2. 可能改进预测的交互特征
3. 特征工程的优先级排序

以 JSON 格式返回：
{{
    "new_features": [
        {{
            "name": "特征名称",
            "description": "特征描述",
            "formula": "计算公式或方法",
            "priority": "high/medium/low"
        }}
    ],
    "interaction_features": ["特征 1 * 特征 2", ...],
    "rationale": "推荐理由"
}}
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text

        # 尝试提取 JSON
        try:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
        except Exception:
            pass

        return {"raw_response": content}

    def analyze_model_performance(
        self,
        model_type: str,
        metrics: dict[str, float],
        feature_importance: pd.DataFrame | None = None,
        error_analysis: str | None = None,
    ) -> dict[str, Any]:
        """
        让 AI 分析模型性能并提供改进建议

        Args:
            model_type: 模型类型
            metrics: 性能指标
            feature_importance: 特征重要性数据
            error_analysis: 错误分析描述

        Returns:
            AI 分析结果
        """
        feature_info = ""
        if feature_importance is not None:
            top_features = feature_importance.head(10).to_string()
            feature_info = f"\n## 特征重要性 Top 10\n{top_features}"

        error_info = f"\n## 错误分析\n{error_analysis}" if error_analysis else ""

        prompt = f"""作为量化金融专家，请分析以下模型性能并提供改进建议。

## 模型类型
{model_type}

## 性能指标
{json.dumps(metrics, indent=2)}
{feature_info}
{error_info}

请提供：
1. 性能诊断：模型表现如何？存在什么问题？
2. 改进建议：具体可行的 3-5 个改进方案
3. 超参数调整方向
4. 特征工程建议
5. 模型选择建议

以 JSON 格式返回：
{{
    "diagnosis": "性能诊断",
    "improvements": [
        {{
            "action": "改进措施",
            "expected_impact": "预期效果",
            "implementation_steps": ["步骤 1", "步骤 2"]
        }}
    ],
    "hyperparameter_suggestions": {{}},
    "feature_suggestions": [],
    "model_recommendations": []
}}
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text

        try:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str)
        except Exception:
            pass

        return {"raw_response": content}

    def generate_strategy_code(
        self,
        strategy_description: str,
        available_data: list[str],
        constraints: dict[str, Any] | None = None,
    ) -> str:
        """
        让 AI 生成交易策略代码

        Args:
            strategy_description: 策略描述
            available_data: 可用数据字段
            constraints: 约束条件

        Returns:
            生成的 Python 代码
        """
        constraints_str = json.dumps(constraints, indent=2) if constraints else "无特殊约束"

        prompt = f"""请根据以下需求生成 Python 交易策略代码。

## 策略描述
{strategy_description}

## 可用数据字段
{', '.join(available_data)}

## 约束条件
{constraints_str}

要求：
1. 使用 pandas 和 numpy
2. 代码清晰，有注释
3. 包含完整的策略逻辑
4. 返回买卖信号

请生成完整可运行的 Python 函数。
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def explain_prediction(
        self,
        model_type: str,
        features: dict[str, float],
        prediction: float,
        feature_importance: dict[str, float] | None = None,
    ) -> str:
        """
        让 AI 解释单个预测结果

        Args:
            model_type: 模型类型
            features: 特征值
            prediction: 预测值
            feature_importance: 特征重要性

        Returns:
            解释文本
        """
        importance_str = ""
        if feature_importance:
            importance_str = f"\n## 特征重要性\n{json.dumps(feature_importance, indent=2)}"

        prompt = f"""请解释这个模型预测结果。

## 模型类型
{model_type}

## 输入特征
{json.dumps(features, indent=2)}

## 预测结果
{prediction}
{importance_str}

请用简洁的语言解释：
1. 为什么模型会给出这个预测？
2. 哪些特征起到了关键作用？
3. 这个预测的可信度如何？
4. 有哪些风险因素？
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def optimize_ensemble_weights(
        self,
        model_performances: dict[str, dict[str, float]],
        correlation_matrix: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        让 AI 建议集成模型的权重

        Args:
            model_performances: 各模型性能指标
            correlation_matrix: 模型预测相关性矩阵

        Returns:
            建议的权重字典
        """
        corr_str = ""
        if correlation_matrix is not None:
            corr_str = f"\n## 模型预测相关性\n{correlation_matrix.to_string()}"

        prompt = f"""请分析以下模型性能并建议集成权重。

## 各模型性能
{json.dumps(model_performances, indent=2)}
{corr_str}

考虑因素：
1. 模型性能指标（准确率、夏普比率等）
2. 模型多样性（相关性低更好）
3. 稳定性和鲁棒性

以 JSON 格式返回权重：
{{
    "weights": {{
        "model1": 0.3,
        "model2": 0.5,
        "model3": 0.2
    }},
    "rationale": "权重分配理由"
}}
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text

        try:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                return result.get("weights", {})
        except Exception:
            pass

        # 默认返回等权
        models = list(model_performances.keys())
        return {model: 1.0 / len(models) for model in models}

    def iterative_improvement(
        self,
        current_state: dict[str, Any],
        improvement_goal: str,
        iterations: int = 3,
    ) -> list[dict[str, Any]]:
        """
        迭代改进循环

        Args:
            current_state: 当前状态（模型、特征、性能等）
            improvement_goal: 改进目标
            iterations: 迭代次数

        Returns:
            改进建议列表
        """
        improvements = []

        for i in range(iterations):
            prompt = f"""这是第 {i+1}/{iterations} 轮改进。

## 当前状态
{json.dumps(current_state, indent=2)}

## 改进目标
{improvement_goal}

## 之前的改进尝试
{json.dumps(improvements, indent=2) if improvements else "这是第一轮"}

请提供下一步具体的改进方案，要与之前不同。

以 JSON 格式返回：
{{
    "iteration": {i+1},
    "approach": "改进方法",
    "actions": ["具体步骤"],
    "expected_outcome": "预期结果",
    "rationale": "选择此方案的理由"
}}
"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text

            try:
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    improvement = json.loads(json_str)
                    improvements.append(improvement)
            except Exception:
                improvements.append({"raw_response": content})

        return improvements


def create_ai_assistant(api_key: str | None = None) -> AIAssistedEvolution:
    """
    创建 AI 辅助系统（便捷函数）

    Args:
        api_key: API key

    Returns:
        AI 辅助系统实例
    """
    return AIAssistedEvolution(api_key=api_key)
