#!/usr/bin/env python3
"""工作流进度监控脚本"""
import json
import time
from pathlib import Path

# 工作流输出目录
WORKFLOW_DIR = Path.home() / ".claude/projects/-home-steve----vibe-coding-hp-ml/2eb5b02a-da75-41a0-bcb4-6712237774d5/subagents/workflows"

def find_latest_workflow():
    """查找最新的工作流"""
    if not WORKFLOW_DIR.exists():
        return None

    workflows = list(WORKFLOW_DIR.glob("wf_*"))
    if not workflows:
        return None

    return max(workflows, key=lambda p: p.stat().st_mtime)

def monitor_workflow():
    """监控工作流执行状态"""
    print("🔍 查找最新工作流...")

    latest = find_latest_workflow()
    if not latest:
        print("❌ 未找到活动工作流")
        return

    print(f"✓ 工作流目录：{latest.name}")

    # 查找输出文件
    output_files = list(latest.glob("*.jsonl")) + list(latest.glob("*.json"))

    if not output_files:
        print("⏳ 工作流尚未产生输出...")
        return

    print(f"\n📊 输出文件 ({len(output_files)} 个):")
    for f in sorted(output_files, key=lambda p: p.stat().st_mtime, reverse=True):
        size = f.stat().st_size / 1024
        print(f"  - {f.name} ({size:.1f} KB)")

if __name__ == "__main__":
    monitor_workflow()
