#!/usr/bin/env python3
"""验证图表生成功能的测试脚本"""

from pathlib import Path
import sys

from hp_ml.charts import (
    write_horizontal_bar_chart,
    write_column_chart,
    write_pie_chart,
    write_line_chart,
)


def test_all_chart_types():
    """测试所有图表类型"""
    test_dir = Path("reports/charts/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    tests_passed = 0
    tests_failed = 0

    # Test 1: 横向柱状图
    try:
        write_horizontal_bar_chart(
            test_dir / "test_horizontal_bar.svg",
            title="测试横向柱状图",
            subtitle="正负值混合测试",
            rows=[
                ("项目 A", 0.15),
                ("项目 B", -0.08),
                ("项目 C", 0.22),
                ("项目 D", -0.05),
                ("项目 E", 0.18),
            ],
            value_kind="pct",
        )
        print("✓ 横向柱状图测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 横向柱状图测试失败：{e}")
        tests_failed += 1

    # Test 2: 纵向柱状图
    try:
        write_column_chart(
            test_dir / "test_column.svg",
            title="测试纵向柱状图",
            subtitle="行业分布测试",
            rows=[
                ("电子", 25),
                ("通信", 18),
                ("计算机", 15),
                ("机械", 12),
                ("医药", 10),
            ],
            value_kind="number",
        )
        print("✓ 纵向柱状图测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 纵向柱状图测试失败：{e}")
        tests_failed += 1

    # Test 3: 饼图
    try:
        write_pie_chart(
            test_dir / "test_pie.svg",
            title="测试饼图",
            subtitle="股东结构测试",
            rows=[
                ("个人", 0.35),
                ("基金", 0.28),
                ("一般法人", 0.15),
                ("QFII", 0.08),
            ],
            add_remainder=True,
        )
        print("✓ 饼图测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 饼图测试失败：{e}")
        tests_failed += 1

    # Test 4: 折线图
    try:
        write_line_chart(
            test_dir / "test_line.svg",
            title="测试折线图",
            subtitle="多系列曲线测试",
            series={
                "系列 A": [
                    ("2020-Q1", 0.05),
                    ("2020-Q2", 0.08),
                    ("2020-Q3", 0.12),
                    ("2020-Q4", 0.10),
                ],
                "系列 B": [
                    ("2020-Q1", 0.03),
                    ("2020-Q2", 0.06),
                    ("2020-Q3", 0.09),
                    ("2020-Q4", 0.11),
                ],
            },
            value_kind="pct",
        )
        print("✓ 折线图测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 折线图测试失败：{e}")
        tests_failed += 1

    # Test 5: 边界情况 - 空数据
    try:
        write_horizontal_bar_chart(
            test_dir / "test_empty.svg",
            title="空数据测试",
            subtitle="应该显示默认消息",
            rows=[],
            value_kind="pct",
        )
        print("✓ 空数据处理测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 空数据处理测试失败：{e}")
        tests_failed += 1

    # Test 6: 边界情况 - 单个数据点
    try:
        write_line_chart(
            test_dir / "test_single_point.svg",
            title="单点数据测试",
            subtitle="单个数据点",
            series={"单点": [("2024", 1.0)]},
            value_kind="number",
        )
        print("✓ 单点数据测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 单点数据测试失败：{e}")
        tests_failed += 1

    # 输出总结
    print("\n" + "=" * 50)
    print(f"测试完成：{tests_passed} 通过，{tests_failed} 失败")
    print(f"测试文件保存在：{test_dir}")
    print("=" * 50)

    return tests_failed == 0


if __name__ == "__main__":
    success = test_all_chart_types()
    sys.exit(0 if success else 1)
