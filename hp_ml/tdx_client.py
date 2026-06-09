"""通达信行情服务器协议客户端 (TDX Protocol Client)

基于 tdx-go 实现，用于从通达信服务器拉取股票/ETF 历史 K 线数据。

协议说明：
- 帧结构：前缀 (1 字节) + 消息 ID(4 字节) + 控制码 (1 字节) + 长度 (2+2 字节) + 类型 (2 字节) + 数据
- 响应结构：前缀 (4 字节) + 控制码 (1 字节) + 消息 ID(4 字节) + 控制码 (1 字节) + 类型 (2 字节) + 长度 (2+2 字节) + 数据
- 数据压缩：使用 zlib 压缩
- 价格单位：厘（元×1000），需要除以 1000 得到元
"""
from __future__ import annotations

import io
import socket
import struct
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

# ============================================================================
# 协议常量
# ============================================================================

# 帧头和控制码
FRAME_PREFIX = 0x0C
RESPONSE_PREFIX = 0x0074CBB1  # 实际接收字节：B1 CB 74 00，小端序解析结果
CONTROL_REQUEST = 0x01

# 消息类型
MSG_TYPE_CONNECT = 0x000D      # 建立连接
MSG_TYPE_HEARTBEAT = 0x0004    # 心跳
MSG_TYPE_KLINE = 0x052D        # K 线数据

# K 线类型
KLINE_TYPE_5MIN = 0x00
KLINE_TYPE_15MIN = 0x01
KLINE_TYPE_30MIN = 0x02
KLINE_TYPE_1HOUR = 0x03
KLINE_TYPE_DAY = 0x04
KLINE_TYPE_WEEK = 0x05
KLINE_TYPE_MONTH = 0x06
KLINE_TYPE_1MIN = 0x07
KLINE_TYPE_1MIN_241 = 0x08     # 241 分钟线
KLINE_TYPE_QUARTER = 0x0A
KLINE_TYPE_YEAR = 0x0B

# 市场代码
EXCHANGE_SH = 1  # 上海
EXCHANGE_SZ = 0  # 深圳
EXCHANGE_BJ = 2  # 北京

# 通达信服务器列表（从 tdx-go README 中提取）
DEFAULT_HOSTS = [
    "124.71.187.122:7709",
    "122.51.120.217:7709",
    "111.229.247.189:7709",
    "124.223.163.242:7709",
    "150.158.160.2:7709",
]


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Kline:
    """K 线数据"""
    time: datetime          # 时间
    open: float            # 开盘价（元）
    high: float            # 最高价（元）
    low: float             # 最低价（元）
    close: float           # 收盘价（元）
    volume: int            # 成交量（股）
    amount: float          # 成交额（元）

    def __str__(self) -> str:
        return (f"{self.time.strftime('%Y-%m-%d')} "
                f"开:{self.open:.3f} 高:{self.high:.3f} "
                f"低:{self.low:.3f} 收:{self.close:.3f} "
                f"量:{self.volume} 额:{self.amount:.0f}")


# ============================================================================
# 工具函数
# ============================================================================

def _parse_code(code: str) -> tuple[int, str]:
    """解析股票代码，返回 (交易所代码，6 位数字)

    支持格式：
    - sz000001, sh600519 (带前缀)
    - 000001, 600519 (不带前缀，根据数字判断)
    """
    code = code.strip().lower()

    if code.startswith("sz"):
        return EXCHANGE_SZ, code[2:].zfill(6)
    elif code.startswith("sh"):
        return EXCHANGE_SH, code[2:].zfill(6)
    elif code.startswith("bj"):
        return EXCHANGE_BJ, code[2:].zfill(6)
    else:
        # 根据代码开头判断交易所
        code_num = code.zfill(6)
        if code_num.startswith(("0", "1", "3")):
            return EXCHANGE_SZ, code_num
        elif code_num.startswith(("6", "9")):
            return EXCHANGE_SH, code_num
        elif code_num.startswith(("4", "8")):
            return EXCHANGE_BJ, code_num
        else:
            # 默认深圳
            return EXCHANGE_SZ, code_num


def _u16(val: int) -> bytes:
    """uint16 小端序"""
    return struct.pack("<H", val)


def _u32(val: int) -> bytes:
    """uint32 小端序"""
    return struct.pack("<I", val)


def _parse_u16(data: bytes) -> int:
    """解析 uint16 小端序"""
    return struct.unpack("<H", data)[0]


def _parse_u32(data: bytes) -> int:
    """解析 uint32 小端序"""
    return struct.unpack("<I", data)[0]


def _parse_price_delta(data: bytes) -> tuple[int, bytes]:
    """解析价格增量（可变长度编码）

    通达信协议使用特殊的可变长度编码：
    - 检查每个字节的最高位（0x80）
    - 如果最高位是 1，说明还有后续字节
    - 如果最高位是 0，说明这是最后一个字节
    - 第一个字节取低 6 位，后续字节取低 7 位

    示例：
    - [0x3F] -> 63 (单字节)
    - [0x80, 0x01] -> 64 (双字节：低 6 位 + 高 7 位)
    """
    result = 0
    i = 0

    # 遍历字节，直到遇到最高位为 0 的字节
    while i < len(data):
        byte_val = data[i]

        if i == 0:
            # 第一个字节：取低 6 位
            result = byte_val & 0x3F
        else:
            # 后续字节：取低 7 位，左移相应位数
            result += (byte_val & 0x7F) << (6 + (i - 1) * 7)

        # 如果最高位是 0，说明这是最后一个字节
        if byte_val & 0x80 == 0:
            return result, data[i + 1:]

        i += 1

    # 不应该到这里
    raise ValueError("价格增量解析失败：未找到结束标志")


def _tdx_time_to_datetime(time_val: int, kline_type: int) -> datetime:
    """将通达信时间值转换为 datetime

    通达信时间编码：
    - 日 K/周 K/月 K/年 K：YYYYMMDD 格式（如 20240608）
    - 分钟 K：压缩格式（前 2 字节=年月日，后 2 字节=时分）
    """
    if kline_type in (KLINE_TYPE_DAY, KLINE_TYPE_WEEK, KLINE_TYPE_MONTH, KLINE_TYPE_YEAR, KLINE_TYPE_QUARTER):
        # 日 K 及以上：YYYYMMDD 格式
        year = time_val // 10000
        month = (time_val % 10000) // 100
        day = time_val % 100
        return datetime(year, month, day, 15, 0, 0)  # 统一设置为下午 3 点
    else:
        # 分钟 K：压缩格式
        # 前 2 字节：年月日压缩（年从 2004 开始，11 位；月 2 位；日 7 位）
        # 后 2 字节：时分（分钟数）
        year_month_day = time_val & 0xFFFF
        hour_minute = (time_val >> 16) & 0xFFFF

        year = (year_month_day >> 11) + 2004
        month = (year_month_day % 2048) // 100
        day = (year_month_day % 2048) % 100
        hour = hour_minute // 60
        minute = hour_minute % 60

        return datetime(year, month, day, hour, minute, 0)


# ============================================================================
# 通达信客户端
# ============================================================================

class TdxClient:
    """通达信行情客户端

    用法：
        with TdxClient() as client:
            klines = client.get_kline_day("sz000001", start=0, count=100)
            for k in klines:
                print(k)
    """

    def __init__(self, host: str | None = None, timeout: float = 10.0):
        """
        Args:
            host: 服务器地址（格式：ip:port），None 则自动选择
            timeout: 连接超时时间（秒）
        """
        self.host = host
        self.timeout = timeout
        self.socket: socket.socket | None = None
        self.msg_id = 0

    def connect(self) -> None:
        """连接到通达信服务器"""
        if self.host is None:
            # 尝试连接默认服务器列表
            for host in DEFAULT_HOSTS:
                try:
                    self._connect_host(host)
                    self.host = host
                    return
                except Exception:
                    continue
            raise ConnectionError("无法连接到任何通达信服务器")
        else:
            self._connect_host(self.host)

    def _connect_host(self, host: str) -> None:
        """连接到指定服务器"""
        ip, port_str = host.split(":")
        port = int(port_str)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)
        self.socket.connect((ip, port))

        # 发送连接请求
        self._send_connect_request()

    def close(self) -> None:
        """关闭连接"""
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

    def __enter__(self) -> TdxClient:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _next_msg_id(self) -> int:
        """获取下一个消息 ID"""
        self.msg_id = (self.msg_id + 1) % 0x100000000
        return self.msg_id

    def _build_frame(self, msg_type: int, data: bytes) -> bytes:
        """构建请求帧"""
        msg_id = self._next_msg_id()
        length = len(data) + 2

        frame = bytearray()
        frame.append(FRAME_PREFIX)
        frame.extend(_u32(msg_id))
        frame.append(CONTROL_REQUEST)
        frame.extend(_u16(length))
        frame.extend(_u16(length))
        frame.extend(_u16(msg_type))
        frame.extend(data)

        return bytes(frame)

    def _recv_response(self) -> tuple[int, bytes]:
        """接收响应帧

        Returns:
            (msg_type, data)
        """
        if not self.socket:
            raise RuntimeError("未连接到服务器")

        # 读取前缀（4 字节）
        prefix_data = self._recv_exact(4)
        prefix = _parse_u32(prefix_data)

        import os
        if os.getenv('TDX_DEBUG'):
            print(f"接收到前缀：hex={prefix_data.hex()}, u32={prefix:08X}")

        if prefix != RESPONSE_PREFIX:
            raise ValueError(f"无效的响应前缀：0x{prefix:08X}")

        # 读取头部（12 字节）
        header = self._recv_exact(12)
        control = header[0]
        msg_id = _parse_u32(header[1:5])
        unknown = header[5]
        msg_type = _parse_u16(header[6:8])
        zip_length = _parse_u16(header[8:10])
        raw_length = _parse_u16(header[10:12])

        # 读取数据
        data = self._recv_exact(zip_length)

        # 解压缩
        if zip_length != raw_length:
            data = zlib.decompress(data)
            if len(data) != raw_length:
                raise ValueError(f"解压后长度不匹配：期望 {raw_length}, 实际 {len(data)}")

        return msg_type, data

    def _recv_exact(self, n: int) -> bytes:
        """精确接收 n 字节数据"""
        if not self.socket:
            raise RuntimeError("未连接到服务器")

        buf = bytearray()
        while len(buf) < n:
            chunk = self.socket.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("连接已关闭")
            buf.extend(chunk)
        return bytes(buf)

    def _send_connect_request(self) -> None:
        """发送连接请求"""
        # 连接请求数据（从 tdx-go 中提取）
        data = bytes([0x01])
        frame = self._build_frame(MSG_TYPE_CONNECT, data)

        if not self.socket:
            raise RuntimeError("未连接到服务器")

        # DEBUG: 打印发送的帧
        import os
        if os.getenv('TDX_DEBUG'):
            print(f"发送连接帧：{frame.hex()}")

        self.socket.sendall(frame)

        # 接收响应
        try:
            msg_type, resp_data = self._recv_response()
            if os.getenv('TDX_DEBUG'):
                print(f"接收响应类型：0x{msg_type:04X}, 数据长度：{len(resp_data)}")
            if msg_type != MSG_TYPE_CONNECT:
                raise ValueError(f"连接响应类型错误：0x{msg_type:04X}")
        except Exception as e:
            if os.getenv('TDX_DEBUG'):
                print(f"接收响应失败：{e}")
            raise

    def get_kline_day(
        self,
        code: str,
        start: int = 0,
        count: int = 800
    ) -> list[Kline]:
        """获取日 K 线数据

        Args:
            code: 股票代码（如 "sz000001", "sh600519"）
            start: 起始位置（0=最新）
            count: 数量（最大 800）

        Returns:
            K 线列表（按时间升序）
        """
        return self._get_kline(code, KLINE_TYPE_DAY, start, count)

    def _get_kline(
        self,
        code: str,
        kline_type: int,
        start: int,
        count: int
    ) -> list[Kline]:
        """获取 K 线数据（内部方法）"""
        if count > 800:
            raise ValueError("单次请求数量不能超过 800")

        exchange, code_num = _parse_code(code)

        # 构建请求数据
        data = bytearray()
        data.append(exchange)
        data.append(0x00)
        data.extend(code_num.encode("ascii"))
        data.append(kline_type)
        data.append(0x00)
        data.extend(bytes([0x01, 0x00]))  # 未知字段
        data.extend(_u16(start))
        data.extend(_u16(count))
        data.extend(bytes(10))  # 填充

        # 发送请求
        frame = self._build_frame(MSG_TYPE_KLINE, bytes(data))
        if not self.socket:
            raise RuntimeError("未连接到服务器")

        self.socket.sendall(frame)

        # 接收响应
        msg_type, resp_data = self._recv_response()
        if msg_type != MSG_TYPE_KLINE:
            raise ValueError(f"K 线响应类型错误：0x{msg_type:04X}")

        # 解析 K 线数据
        return self._parse_kline_response(resp_data, kline_type)

    def _parse_kline_response(self, data: bytes, kline_type: int) -> list[Kline]:
        """解析 K 线响应数据"""
        if len(data) < 2:
            raise ValueError("响应数据长度不足")

        count = _parse_u16(data[:2])
        data = data[2:]

        import os
        if os.getenv('TDX_DEBUG'):
            print(f"K 线数量：{count}, 剩余数据：{len(data)} 字节")

        klines: list[Kline] = []
        last_close_price = 0  # 上一根 K 线的收盘价（厘）

        for i in range(count):
            if len(data) < 4:
                break

            # 时间（4 字节）
            time_val = _parse_u32(data[:4])
            if os.getenv('TDX_DEBUG'):
                if i < 3 or i == count - 1:  # 打印前 3 根和最后 1 根
                    print(f"第 {i+1} 根 K 线时间：hex={data[:4].hex()}, u32={time_val}")
            data = data[4:]

            # 价格增量（可变长度）
            # 注意：这些是增量值，需要累加
            # open_delta: 开盘价相对上一根 K 线收盘价的增量
            # close_delta: 收盘价相对本根 K 线开盘价的增量
            # high_delta: 最高价相对本根 K 线开盘价的增量
            # low_delta: 最低价相对本根 K 线开盘价的增量
            open_delta, data = _parse_price_delta(data)
            close_delta, data = _parse_price_delta(data)
            high_delta, data = _parse_price_delta(data)
            low_delta, data = _parse_price_delta(data)

            # 计算实际价格（厘）
            # 参考 tdx-go 的实现：
            # k.Open = open + last
            # k.Close = last + open + _close
            # k.High = open + last + high
            # k.Low = open + last + low
            open_price = last_close_price + open_delta
            close_price = last_close_price + open_delta + close_delta
            high_price = last_close_price + open_delta + high_delta
            low_price = last_close_price + open_delta + low_delta

            # 成交量（4 字节）
            if len(data) < 4:
                break
            volume = _parse_u32(data[:4])
            data = data[4:]

            # 成交额（4 字节，单位：手×价格，需要转换）
            if len(data) < 4:
                break
            amount_raw = _parse_u32(data[:4])
            data = data[4:]

            # 调整成交量和成交额（日 K 及以上需要除以 100）
            if kline_type in (KLINE_TYPE_DAY, KLINE_TYPE_WEEK, KLINE_TYPE_MONTH, KLINE_TYPE_YEAR):
                # 日 K 及以上不需要调整
                pass
            else:
                # 分钟 K 需要除以 100
                volume //= 100

            # 转换为元
            kline = Kline(
                time=_tdx_time_to_datetime(time_val, kline_type),
                open=open_price / 1000.0,
                high=high_price / 1000.0,
                low=low_price / 1000.0,
                close=close_price / 1000.0,
                volume=volume * 100,  # 转换为股（1 手=100 股）
                amount=amount_raw * 1000.0,  # 转换为元
            )

            klines.append(kline)
            last_close_price = close_price

        return klines

    def get_kline_day_all(self, code: str, max_count: int = 10000) -> list[Kline]:
        """获取全部日 K 线数据（分批拉取）

        Args:
            code: 股票代码
            max_count: 最大数量限制

        Returns:
            K 线列表（按时间升序）
        """
        all_klines: list[Kline] = []
        start = 0
        batch_size = 800

        while len(all_klines) < max_count:
            klines = self.get_kline_day(code, start=start, count=batch_size)
            if not klines:
                break

            all_klines.extend(klines)
            start += len(klines)

            # 如果返回数量少于请求数量，说明已经没有更多数据
            if len(klines) < batch_size:
                break

            # 避免请求过快
            time.sleep(0.1)

        return all_klines[:max_count]


# ============================================================================
# 便捷函数
# ============================================================================

def fetch_kline_day(code: str, host: str | None = None) -> list[Kline]:
    """便捷函数：获取日 K 线数据

    Args:
        code: 股票代码
        host: 服务器地址（None=自动选择）

    Returns:
        K 线列表
    """
    with TdxClient(host=host) as client:
        return client.get_kline_day_all(code)


if __name__ == "__main__":
    # 测试代码
    print("测试通达信客户端...")

    test_codes = ["sz000001", "sh600519", "sz159300"]

    for code in test_codes:
        print(f"\n{'='*60}")
        print(f"拉取 {code} 最近 10 个交易日 K 线数据")
        print(f"{'='*60}")

        try:
            with TdxClient() as client:
                klines = client.get_kline_day(code, start=0, count=10)
                print(f"成功获取 {len(klines)} 根 K 线\n")

                for kline in klines:
                    print(kline)
        except Exception as e:
            print(f"错误：{e}")

    print("\n测试完成！")
