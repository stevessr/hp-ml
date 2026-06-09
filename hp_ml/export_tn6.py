"""导出通达信 TN6 格式文件

TN6 是通达信的指标公式文件格式，包含公式代码和元数据
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any


class TN6Exporter:
    """通达信 TN6 格式导出器

    TN6 文件结构（推测）：
    1. 文件头：标识符、版本号
    2. 公式元数据：名称、类型、参数
    3. 公式代码：指标公式文本
    4. 校验和
    """

    def __init__(self):
        self.version = 1
        self.encoding = 'gbk'  # 通达信使用 GBK 编码

    def export_formula(
        self,
        formula_name: str,
        formula_code: str,
        formula_type: int = 1,  # 1=技术指标, 2=条件选股, 3=专家系统
        output_path: Path | str = None,
    ) -> bytes:
        """
        导出单个公式为 TN6 格式

        Args:
            formula_name: 公式名称
            formula_code: 公式代码文本
            formula_type: 公式类型 (1=技术指标, 2=条件选股, 3=专家系统)
            output_path: 输出文件路径（可选）

        Returns:
            TN6 二进制数据
        """
        # 构建 TN6 数据
        tn6_data = self._build_tn6_data(formula_name, formula_code, formula_type)

        # 如果指定了输出路径，写入文件
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tn6_data)
            print(f"✅ 已导出 TN6 文件：{output_path}")

        return tn6_data

    def _build_tn6_data(
        self,
        formula_name: str,
        formula_code: str,
        formula_type: int,
    ) -> bytes:
        """构建 TN6 二进制数据

        TN6 文件格式（基于逆向分析和社区经验）：
        1. 文件头标识 (4 bytes): "TN6\x00" 或类似标识
        2. 版本号 (4 bytes): 整数
        3. 公式类型 (4 bytes): 1=技术指标, 2=条件选股, 3=专家系统
        4. 公式名称长度 (4 bytes): 字符串长度
        5. 公式名称: GBK 编码的字符串
        6. 公式代码长度 (4 bytes): 字符串长度
        7. 公式代码: GBK 编码的字符串
        8. 参数区（可选）
        9. 校验和 (4 bytes): CRC32 或简单求和
        """
        data = bytearray()

        # 1. 文件头标识
        data.extend(b'TN6\x00')

        # 2. 版本号
        data.extend(struct.pack('<I', self.version))

        # 3. 公式类型
        data.extend(struct.pack('<I', formula_type))

        # 4-5. 公式名称
        name_bytes = formula_name.encode(self.encoding)
        data.extend(struct.pack('<I', len(name_bytes)))
        data.extend(name_bytes)

        # 6-7. 公式代码
        code_bytes = formula_code.encode(self.encoding)
        data.extend(struct.pack('<I', len(code_bytes)))
        data.extend(code_bytes)

        # 8. 简单校验和（所有字节求和）
        checksum = sum(data) & 0xFFFFFFFF
        data.extend(struct.pack('<I', checksum))

        return bytes(data)

    def export_tdx_formula_as_tn6(
        self,
        tdx_file: Path | str,
        output_file: Path | str = None,
        formula_type: int = 1,
    ):
        """
        将 .tdx 文本公式文件转换为 .tn6 二进制格式

        Args:
            tdx_file: 输入的 .tdx 文本文件路径
            output_file: 输出的 .tn6 文件路径（默认替换扩展名）
            formula_type: 公式类型
        """
        tdx_file = Path(tdx_file)

        if not tdx_file.exists():
            raise FileNotFoundError(f"TDX 文件不存在：{tdx_file}")

        # 读取 TDX 文件内容
        with open(tdx_file, 'r', encoding='utf-8') as f:
            formula_code = f.read()

        # 从文件名提取公式名称
        formula_name = tdx_file.stem

        # 如果未指定输出文件，使用相同路径替换扩展名
        if output_file is None:
            output_file = tdx_file.with_suffix('.tn6')
        else:
            output_file = Path(output_file)

        # 导出为 TN6
        self.export_formula(
            formula_name=formula_name,
            formula_code=formula_code,
            formula_type=formula_type,
            output_path=output_file,
        )

        return output_file


def convert_all_tdx_to_tn6(input_dir: Path | str, output_dir: Path | str = None):
    """
    批量转换目录下所有 .tdx 文件为 .tn6 格式

    Args:
        input_dir: 输入目录（包含 .tdx 文件）
        output_dir: 输出目录（默认与输入相同）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    exporter = TN6Exporter()

    tdx_files = list(input_dir.glob('*.tdx'))
    if not tdx_files:
        print(f"❌ 未找到 .tdx 文件：{input_dir}")
        return

    print(f"\n🔄 开始批量转换...")
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print(f"找到 {len(tdx_files)} 个 .tdx 文件\n")

    success_count = 0
    for tdx_file in tdx_files:
        try:
            output_file = output_dir / tdx_file.with_suffix('.tn6').name
            exporter.export_tdx_formula_as_tn6(tdx_file, output_file)
            success_count += 1
        except Exception as e:
            print(f"❌ 转换失败：{tdx_file.name} - {e}")

    print(f"\n✅ 转换完成！成功 {success_count}/{len(tdx_files)} 个文件")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="转换 .tdx 公式文件为 .tn6 格式")
    parser.add_argument("input", help="输入 .tdx 文件或目录")
    parser.add_argument("-o", "--output", help="输出 .tn6 文件或目录")
    parser.add_argument("-t", "--type", type=int, default=1,
                        help="公式类型：1=技术指标(默认), 2=条件选股, 3=专家系统")
    parser.add_argument("--batch", action="store_true",
                        help="批量转换目录下所有 .tdx 文件")

    args = parser.parse_args()

    if args.batch:
        convert_all_tdx_to_tn6(args.input, args.output)
    else:
        exporter = TN6Exporter()
        exporter.export_tdx_formula_as_tn6(
            args.input,
            args.output,
            args.type
        )
