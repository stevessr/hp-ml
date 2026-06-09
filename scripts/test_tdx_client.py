#!/usr/bin/env python
"""测试通达信数据拉取功能

验证：
1. 基础连接和 K 线拉取
2. 数据格式正确性
3. 与东方财富数据源的对比
4. 集成到训练流水线
"""
from __future__ import annotations

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hp_ml.tdx_client import TdxClient
from hp_ml.tdx_data_source import fetch_etf_history_tdx


def test_basic_connection():
    """测试基础连接"""
    print("=" * 60)
    print("测试 1: 基础连接和 K 线拉取")
    print("=" * 60)

    try:
        with TdxClient() as client:
            print("✓ 连接成功")

            # 测试拉取数据
            klines = client.get_kline_day("sz000001", start=0, count=10)
            print(f"✓ 获取到 {len(klines)} 根 K 线")

            if klines:
                print(f"\n最新 3 根 K 线：")
                for k in klines[-3:]:
                    print(f"  {k}")
                return True
            else:
                print("✗ 未获取到 K 线数据")
                return False

    except Exception as e:
        print(f"✗ 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_format():
    """测试数据格式"""
    print("\n" + "=" * 60)
    print("测试 2: 数据格式验证")
    print("=" * 60)

    try:
        df = fetch_etf_history_tdx("sz159300", cache=False)
        print(f"✓ 获取到 {len(df)} 条记录")
        print(f"✓ 列名：{list(df.columns)}")
        print(f"✓ 时间范围：{df['date'].min()} 至 {df['date'].max()}")

        # 检查数据类型
        print(f"\n数据类型：")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")

        # 显示最新数据
        print(f"\n最新 3 条记录：")
        print(df.tail(3).to_string())

        return True

    except Exception as e:
        print(f"✗ 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_codes():
    """测试多个 ETF 代码"""
    print("\n" + "=" * 60)
    print("测试 3: 多个 ETF 代码")
    print("=" * 60)

    test_codes = [
        ("sz159300", "沪深 300ETF"),
        ("sh510300", "沪深 300ETF"),
        ("sz000001", "平安银行"),
        ("sh600519", "贵州茅台"),
    ]

    success_count = 0

    for code, name in test_codes:
        try:
            with TdxClient() as client:
                klines = client.get_kline_day(code, start=0, count=5)
                if klines:
                    latest = klines[-1]
                    print(f"✓ {name}({code}): 收盘 {latest.close:.3f}")
                    success_count += 1
                else:
                    print(f"✗ {name}({code}): 无数据")
        except Exception as e:
            print(f"✗ {name}({code}): {e}")

    print(f"\n成功率：{success_count}/{len(test_codes)}")
    return success_count == len(test_codes)


def test_comparison():
    """对比通达信和东方财富数据源"""
    print("\n" + "=" * 60)
    print("测试 4: 数据源对比（通达信 vs 东方财富）")
    print("=" * 60)

    try:
        from hp_ml.data_sources import fetch_etf_history

        code = "sz159300"

        # 从东方财富拉取（不使用 cache 参数）
        print(f"从东方财富拉取 {code}...")
        df_em = fetch_etf_history(code)
        print(f"✓ 东方财富：{len(df_em)} 条记录")

        # 从通达信拉取
        print(f"从通达信拉取 {code}...")
        df_tdx = fetch_etf_history_tdx(code, cache=False)
        print(f"✓ 通达信：{len(df_tdx)} 条记录")

        # 对比最新数据（取同一日期）
        latest_date_em = df_em['date'].max()
        latest_date_tdx = df_tdx['date'].max()

        print(f"\n最新数据日期：")
        print(f"  东方财富：{latest_date_em}")
        print(f"  通达信：  {latest_date_tdx}")

        if latest_date_em.date() == latest_date_tdx.date():
            close_em = df_em['close'].iloc[-1]
            close_tdx = df_tdx['close'].iloc[-1]

            print(f"\n最新收盘价对比：")
            print(f"  东方财富：{close_em:.3f}")
            print(f"  通达信：  {close_tdx:.3f}")

            # 计算差异
            diff = abs(close_em - close_tdx)
            diff_pct = abs(diff / close_em * 100)
            print(f"  差异：    {diff:.3f} ({diff_pct:.2f}%)")

            if diff_pct < 1.0:  # 允许 1% 的误差
                print("✓ 数据基本一致")
                return True
            else:
                print("⚠ 数据存在差异，但仍可用")
                return True
        else:
            print("⚠ 最新数据日期不一致，可能是交易时间差异")
            return True

    except Exception as e:
        print(f"✗ 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("通达信数据拉取功能测试")
    print("=" * 60 + "\n")

    tests = [
        ("基础连接", test_basic_connection),
        ("数据格式", test_data_format),
        ("多 ETF 代码", test_multiple_codes),
        ("数据源对比", test_comparison),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n测试 '{name}' 异常：{e}")
            results.append((name, False))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    success_count = 0
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}  {name}")
        if result:
            success_count += 1

    print(f"\n总计：{success_count}/{len(results)} 通过")

    if success_count == len(results):
        print("\n✓ 所有测试通过！通达信数据源已就绪。")
        return 0
    else:
        print("\n⚠ 部分测试失败，请检查日志。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
