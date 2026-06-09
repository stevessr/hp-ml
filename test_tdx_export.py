#!/usr/bin/env python3
"""快速测试模型导出到通达信公式

使用方法：
    python test_tdx_export.py prophet
    python test_tdx_export.py lstm
    python test_tdx_export.py ridge
"""
import sys
from pathlib import Path

def test_export(model_type: str):
    """测试模型导出"""
    print(f"\n{'='*60}")
    print(f"测试 {model_type.upper()} 模型导出到通达信")
    print(f"{'='*60}\n")

    # 检查模型文件是否存在
    model_path = Path(f"models/model_{model_type}.joblib")
    if not model_path.exists():
        print(f"❌ 模型文件不存在：{model_path}")
        print(f"\n请先训练模型：")
        print(f"    python -m hp_ml.multi_model_train --models {model_type}")
        return False

    print(f"✓ 找到模型文件：{model_path}")
    print(f"  文件大小：{model_path.stat().st_size / 1024:.1f} KB\n")

    # 执行导出
    print("开始导出到通达信公式...")
    import subprocess

    output_dir = Path("reports/tdx_formulas_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "hp_ml.export_tdx",
        "--model", str(model_path),
        "--out", str(output_dir),
        "--all-families",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ 导出失败：")
        print(result.stderr)
        return False

    # 检查生成的文件
    tdx_files = list(output_dir.glob("*.tdx"))
    if not tdx_files:
        print(f"❌ 未生成通达信公式文件")
        return False

    print(f"\n✅ 导出成功！生成了 {len(tdx_files)} 个公式文件：\n")

    for f in sorted(tdx_files):
        size = f.stat().st_size / 1024
        print(f"  - {f.name} ({size:.1f} KB)")

    # 显示第一个文件的预览
    first_file = sorted(tdx_files)[0]
    print(f"\n{'='*60}")
    print(f"公式预览：{first_file.name}")
    print(f"{'='*60}\n")

    with open(first_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:30]  # 显示前30行
        print(''.join(lines))
        if len(lines) == 30:
            print("\n... (省略后续内容)")

    # 使用说明
    print(f"\n{'='*60}")
    print("下一步操作")
    print(f"{'='*60}\n")

    print(f"1. 打开通达信软件")
    print(f"2. 功能 → 公式管理器 → 技术指标公式 → 新建")
    print(f"3. 导入公式文件：{output_dir}")

    if "ridge" in model_type:
        print(f"4. 单层公式，直接使用即可")
    else:
        print(f"4. 多层公式，按顺序创建 4 个公式：")
        print(f"   - 特征层")
        print(f"   - 隐藏层1")
        print(f"   - 隐藏层2")
        print(f"   - 输出层")

    print(f"\n详细使用说明：TDX_EXPORT_GUIDE.md")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法：")
        print("    python test_tdx_export.py prophet")
        print("    python test_tdx_export.py lstm")
        print("    python test_tdx_export.py ridge")
        print("    python test_tdx_export.py gru")
        sys.exit(1)

    model_type = sys.argv[1].lower()
    success = test_export(model_type)

    if success:
        print("\n✅ 测试通过！")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)
