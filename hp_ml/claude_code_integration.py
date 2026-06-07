"""Claude Code 集成：通过命令行调用 Claude Code 实现自动进化"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


class ClaudeCodeEvolution:
    """通过 Claude Code 实现模型自动进化"""

    def __init__(self, claude_code_path: str = "claude-code"):
        """
        Args:
            claude_code_path: claude-code 命令路径
        """
        self.claude_code_path = claude_code_path
        self._verify_claude_code()

    def _verify_claude_code(self) -> None:
        """验证 Claude Code 是否可用"""
        try:
            result = subprocess.run(
                [self.claude_code_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError("Claude Code 不可用")
        except Exception as e:
            raise RuntimeError(f"无法找到 Claude Code: {e}")

    def generate_improved_features(
        self,
        current_features: list[str],
        performance_metrics: dict[str, float],
        data_sample: pd.DataFrame,
    ) -> str:
        """
        让 Claude Code 生成改进的特征工程代码

        Args:
            current_features: 当前特征列表
            performance_metrics: 当前性能指标
            data_sample: 数据样本

        Returns:
            生成的 Python 代码
        """
        # 准备上下文文件
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 写入当前状态
            context_file = tmpdir_path / "context.json"
            context = {
                "current_features": current_features,
                "performance_metrics": performance_metrics,
                "data_columns": data_sample.columns.tolist(),
                "data_sample": data_sample.head(10).to_dict(),
            }

            with open(context_file, "w", encoding="utf-8") as f:
                json.dump(context, f, indent=2, ensure_ascii=False)

            # 构建提示
            prompt = f"""请分析 {context_file} 中的模型性能数据，生成改进的特征工程代码。

要求：
1. 分析当前特征和性能指标
2. 设计 3-5 个新的高价值特征
3. 生成完整的 Python 函数代码
4. 代码应该可以直接运行，接收 DataFrame 返回新特征

将代码保存到 {tmpdir_path / 'improved_features.py'}
"""

            # 调用 Claude Code
            result = subprocess.run(
                [self.claude_code_path, "chat", prompt],
                cwd=tmpdir_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            # 读取生成的代码
            output_file = tmpdir_path / "improved_features.py"
            if output_file.exists():
                return output_file.read_text(encoding="utf-8")
            else:
                return result.stdout

    def optimize_model_hyperparameters(
        self,
        model_type: str,
        current_params: dict[str, Any],
        performance_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        让 Claude Code 优化模型超参数

        Args:
            model_type: 模型类型
            current_params: 当前参数
            performance_history: 历史性能记录

        Returns:
            建议的新参数
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 写入历史数据
            history_file = tmpdir_path / "history.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_type": model_type,
                        "current_params": current_params,
                        "performance_history": performance_history,
                    },
                    f,
                    indent=2,
                )

            prompt = f"""分析 {history_file} 中的超参数搜索历史，建议下一组参数。

要求：
1. 分析性能趋势
2. 识别最有影响的参数
3. 建议新的参数组合
4. 将结果保存为 JSON 到 {tmpdir_path / 'suggested_params.json'}
"""

            subprocess.run(
                [self.claude_code_path, "chat", prompt],
                cwd=tmpdir_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            result_file = tmpdir_path / "suggested_params.json"
            if result_file.exists():
                with open(result_file, "r", encoding="utf-8") as f:
                    return json.load(f)

            return current_params

    def design_ensemble_strategy(
        self,
        model_performances: dict[str, dict[str, float]],
        prediction_correlations: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        让 Claude Code 设计集成策略

        Args:
            model_performances: 各模型性能
            prediction_correlations: 预测相关性矩阵

        Returns:
            集成策略配置
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 保存数据
            perf_file = tmpdir_path / "performances.json"
            with open(perf_file, "w", encoding="utf-8") as f:
                json.dump(model_performances, f, indent=2)

            corr_file = tmpdir_path / "correlations.csv"
            prediction_correlations.to_csv(corr_file)

            prompt = f"""基于 {perf_file} 和 {corr_file}，设计最优集成策略。

分析：
1. 模型性能差异
2. 预测相关性
3. 最优权重分配
4. 集成方法选择（Stacking/Blending/Weighted）

将策略保存到 {tmpdir_path / 'ensemble_strategy.json'}，包含：
- weights: 各模型权重
- method: 集成方法
- rationale: 选择理由
"""

            subprocess.run(
                [self.claude_code_path, "chat", prompt],
                cwd=tmpdir_path,
                capture_output=True,
                text=True,
                timeout=300,
            )

            strategy_file = tmpdir_path / "ensemble_strategy.json"
            if strategy_file.exists():
                with open(strategy_file, "r", encoding="utf-8") as f:
                    return json.load(f)

            # 默认策略
            models = list(model_performances.keys())
            return {
                "weights": {m: 1.0 / len(models) for m in models},
                "method": "weighted_average",
                "rationale": "等权平均（默认）",
            }

    def iterative_model_evolution(
        self,
        project_dir: Path,
        evolution_goal: str,
        max_iterations: int = 5,
    ) -> list[dict[str, Any]]:
        """
        迭代模型进化循环

        Args:
            project_dir: 项目目录
            evolution_goal: 进化目标
            max_iterations: 最大迭代次数

        Returns:
            进化历史记录
        """
        evolution_history = []

        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"进化迭代 {iteration + 1}/{max_iterations}")
            print(f"{'='*60}")

            # 构建提示
            prompt = f"""这是模型进化的第 {iteration + 1} 轮迭代。

## 目标
{evolution_goal}

## 之前的尝试
{json.dumps(evolution_history, indent=2, ensure_ascii=False) if evolution_history else '这是第一轮'}

## 任务
1. 分析当前模型性能（查看 reports/ 目录）
2. 识别改进机会
3. 实施一个具体的改进（特征/模型/策略）
4. 运行训练并记录结果
5. 将结果总结保存到 reports/evolution_iter_{iteration + 1}.json

请自主完成以上任务。
"""

            # 调用 Claude Code
            result = subprocess.run(
                [self.claude_code_path, "chat", prompt],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )

            # 读取迭代结果
            result_file = project_dir / "reports" / f"evolution_iter_{iteration + 1}.json"
            if result_file.exists():
                with open(result_file, "r", encoding="utf-8") as f:
                    iter_result = json.load(f)
                    evolution_history.append(iter_result)
                    print(f"✓ 迭代 {iteration + 1} 完成")
                    print(f"  改进: {iter_result.get('improvement', 'N/A')}")
                    print(f"  性能: {iter_result.get('performance', 'N/A')}")
            else:
                print(f"⚠ 迭代 {iteration + 1} 未生成结果文件")
                evolution_history.append({
                    "iteration": iteration + 1,
                    "status": "incomplete",
                    "output": result.stdout[:500],
                })

        return evolution_history

    def autonomous_feature_discovery(
        self,
        project_dir: Path,
        data_description: str,
        target_description: str,
    ) -> list[str]:
        """
        自主特征发现

        Args:
            project_dir: 项目目录
            data_description: 数据描述
            target_description: 目标描述

        Returns:
            发现的新特征列表
        """
        prompt = f"""自主探索和发现新特征。

## 数据
{data_description}

## 目标
{target_description}

## 任务
1. 探索 data/processed/ 中的数据
2. 分析现有特征（查看 hp_ml/features.py）
3. 设计并实现 5-10 个创新特征
4. 将新特征代码添加到 hp_ml/features.py
5. 将特征列表保存到 reports/discovered_features.json

请自主完成特征发现过程。
"""

        subprocess.run(
            [self.claude_code_path, "chat", prompt],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )

        # 读取发现的特征
        features_file = project_dir / "reports" / "discovered_features.json"
        if features_file.exists():
            with open(features_file, "r", encoding="utf-8") as f:
                result = json.load(f)
                return result.get("features", [])

        return []


def create_claude_code_assistant(claude_code_path: str = "claude-code") -> ClaudeCodeEvolution:
    """
    创建 Claude Code 助手（便捷函数）

    Args:
        claude_code_path: claude-code 命令路径

    Returns:
        Claude Code 助手实例
    """
    return ClaudeCodeEvolution(claude_code_path=claude_code_path)
