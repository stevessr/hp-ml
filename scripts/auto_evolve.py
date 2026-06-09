#!/usr/bin/env python3
"""
自动模型进化脚本：通过 Claude Code 实现持续优化

这个脚本会启动一个自主进化循环，让 Claude Code 自动：
1. 分析当前模型性能
2. 识别改进机会
3. 设计并实施改进方案
4. 评估改进效果
5. 迭代优化直到达到目标
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hp_ml.claude_code_integration import ClaudeCodeEvolution


def main():
    parser = argparse.ArgumentParser(description="AI 驱动的模型自动进化")

    parser.add_argument(
        "--goal",
        default="将夏普比率提升到 1.5 以上",
        help="进化目标描述"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="最大迭代次数"
    )

    parser.add_argument(
        "--claude-code-path",
        default="claude-code",
        help="claude-code 命令路径"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🧬 AI 驱动的模型自动进化系统")
    print("=" * 80)
    print(f"\n目标：{args.goal}")
    print(f"最大迭代：{args.max_iterations}")
    print()

    # 获取项目目录
    project_dir = Path(__file__).parent.parent

    # 创建 Claude Code 助手
    try:
        assistant = ClaudeCodeEvolution(claude_code_path=args.claude_code_path)
        print("✓ Claude Code 助手已就绪\n")
    except Exception as e:
        print(f"✗ 无法初始化 Claude Code 助手：{e}")
        print("\n请确保：")
        print("1. 已安装 Claude Code: https://claude.ai/download")
        print("2. Claude Code 在 PATH 中，或通过 --claude-code-path 指定路径")
        sys.exit(1)

    # 启动自主进化循环
    print("启动自主进化循环...\n")

    evolution_history = assistant.iterative_model_evolution(
        project_dir=project_dir,
        evolution_goal=args.goal,
        max_iterations=args.max_iterations,
    )

    # 输出总结
    print("\n" + "=" * 80)
    print("🎯 进化完成！")
    print("=" * 80)

    print(f"\n完成 {len(evolution_history)} 轮迭代\n")

    for i, record in enumerate(evolution_history, 1):
        status = record.get("status", "completed")
        improvement = record.get("improvement", "N/A")
        print(f"  迭代 {i}: {status} - {improvement}")

    print(f"\n详细记录已保存到：{project_dir / 'reports' / 'evolution_*.json'}")

    # 检查是否达到目标
    final_performance = evolution_history[-1].get("performance", {}) if evolution_history else {}
    sharpe = final_performance.get("sharpe_ratio", 0)

    if sharpe >= 1.5:
        print("\n✓ 目标达成！")
    else:
        print(f"\n⚠ 当前夏普比率：{sharpe:.2f}，继续优化可能需要更多迭代")


if __name__ == "__main__":
    main()
