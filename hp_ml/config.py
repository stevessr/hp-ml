"""Configuration for CSI broad-based ETF discovery.

The universe is intentionally discovered from exchange ETF quotes first.  These
patterns only classify the discovered names into broad-index families and remove
obvious sector/theme/style products.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HISTORY_DIR = RAW_DIR / "history"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

for _path in (RAW_DIR, HISTORY_DIR, PROCESSED_DIR, REPORTS_DIR, MODELS_DIR):
    _path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BroadIndexFamily:
    """A CSI broad-based index family used to label ETF products."""

    family_id: str
    display_name: str
    index_code: str | None
    patterns: tuple[str, ...]
    description: str


# Keep longer/more specific patterns before shorter ones to avoid e.g. A500
# being classified as CSI_500, or 中证 2000 being classified as CSI_200.
BROAD_INDEX_FAMILIES: tuple[BroadIndexFamily, ...] = (
    BroadIndexFamily(
        "CSI_A500",
        "中证 A500",
        "000510",
        (r"中证\s*A\s*500", r"(?<![A-Z0-9])A\s*500\s*ETF", r"(?<![A-Z0-9])A500(?![0-9])"),
        "行业均衡的大盘宽基，近年 ETF 扩容明显。",
    ),
    BroadIndexFamily(
        "CSI_A50",
        "中证 A50",
        "930050",
        (r"中证\s*A\s*50", r"(?<![A-Z0-9])A\s*50\s*ETF"),
        "中证 A 系列大盘龙头宽基。",
    ),
    BroadIndexFamily(
        "CSI_A100",
        "中证 A100",
        "000903",
        (r"中证\s*A?\s*100(?!0)", r"(?<![A-Z0-9])A\s*100\s*ETF", r"(?<![A-Z0-9])A100(?!0)"),
        "中证 100 更名后的 A 系列核心龙头宽基。",
    ),
    BroadIndexFamily(
        "CSI_300",
        "沪深 300",
        "000300",
        (r"沪深\s*300", r"(?<![A-Z0-9])HS\s*300"),
        "沪深两市大盘核心资产宽基。",
    ),
    BroadIndexFamily(
        "CSI_2000",
        "中证 2000",
        "932000",
        (r"中证\s*2000",),
        "更偏小微盘的宽基指数。",
    ),
    BroadIndexFamily(
        "CSI_1000",
        "中证 1000",
        "000852",
        (r"中证\s*1000",),
        "剔除沪深 300/中证 500 后的中小盘宽基。",
    ),
    BroadIndexFamily(
        "CSI_800",
        "中证 800",
        "000906",
        (r"中证\s*800",),
        "沪深 300 与中证 500 合成的大中盘宽基。",
    ),
    BroadIndexFamily(
        "CSI_700",
        "中证 700",
        "000907",
        (r"中证\s*700",),
        "中证 200 与中证 500 合成的中小盘宽基。",
    ),
    BroadIndexFamily(
        "CSI_500",
        "中证 500",
        "000905",
        (r"中证\s*500",),
        "剔除沪深 300 后的中盘宽基。",
    ),
    BroadIndexFamily(
        "CSI_200",
        "中证 200",
        "000904",
        (r"中证\s*200(?!0)",),
        "沪深 300 样本中排名 101 至 300 名的中盘宽基。",
    ),
    BroadIndexFamily(
        "CSI_ALL_SHARE",
        "中证全指",
        "000985",
        (r"中证\s*全指 (?=\s*ETF|交易型 | 指数\s*ETF|$)",),
        "覆盖沪深北市场符合条件股票和存托凭证的全市场宽基。",
    ),
    BroadIndexFamily(
        "CSI_FREE_FLOAT",
        "中证流通",
        "000902",
        (r"中证\s*流通",),
        "覆盖沪深北市场符合条件 A 股和存托凭证的可流通宽基。",
    ),
    BroadIndexFamily(
        "CSI_A_SHARE",
        "中证 A 股",
        "930903",
        (r"中证\s*A\s*股",),
        "覆盖沪深北市场符合条件 A 股的全市场宽基。",
    ),
)

# Terms that usually indicate the product is not a pure CSI broad-based ETF even
# if the name contains 300/500/1000/A500 etc.  The classifier can optionally keep
# enhanced products, but style/sector/cross-border products are excluded by
# default to stay focused on broad-index ETF investment tools.
STYLE_OR_THEME_TERMS = (
    "红利",
    "低波",
    "价值",
    "成长",
    "质量",
    "等权",
    "ESG",
    "央企",
    "国企",
    "高股息",
    "策略",
    "动量",
    "基本面",
    "自由现金流",
    "现金流",
    "消费",
    "医药",
    "医疗",
    "银行",
    "证券",
    "保险",
    "金融",
    "地产",
    "煤炭",
    "钢铁",
    "有色",
    "能源",
    "电池",
    "新能源",
    "汽车",
    "半导体",
    "芯片",
    "通信",
    "传媒",
    "游戏",
    "军工",
    "农业",
    "化工",
    "材料",
    "机器人",
    "人工智能",
    "AI",
    "科创",
    "创业板",
    "双创",
    "沪港深",
    "港股",
    "香港",
    "海外",
    "富时",
    "MSCI",
    "中国 A50",
    "纳斯达克",
    "标普",
    "日经",
    "德国",
    "法国",
    "沙特",
    "印度",
    "黄金",
    "商品",
    "豆粕",
    "REIT",
    "债",
    "货币",
)

ENHANCED_TERMS = ("增强", "指数增强", "增强策略", "指增")

# Fallback seeds are used only when the live ETF discovery endpoint is
# unavailable.  They are deliberately small and liquid representatives; normal
# runs still discover the universe from Eastmoney quotes.
FALLBACK_BROAD_ETF_SEEDS: tuple[dict[str, str], ...] = (
    {"code": "510300", "name": "沪深 300ETF 华泰柏瑞", "family_id": "CSI_300"},
    {"code": "510310", "name": "沪深 300ETF 易方达", "family_id": "CSI_300"},
    {"code": "510500", "name": "中证 500ETF 南方", "family_id": "CSI_500"},
    {"code": "512500", "name": "中证 500ETF 华夏", "family_id": "CSI_500"},
    {"code": "512100", "name": "中证 1000ETF 南方", "family_id": "CSI_1000"},
    {"code": "159845", "name": "中证 1000ETF 华夏", "family_id": "CSI_1000"},
    {"code": "563360", "name": "中证 A500ETF 华泰柏瑞", "family_id": "CSI_A500"},
    {"code": "159352", "name": "中证 A500ETF", "family_id": "CSI_A500"},
    {"code": "159136", "name": "中证 A50ETF 广发", "family_id": "CSI_A50"},
    {"code": "159601", "name": "中证 A50ETF 华夏", "family_id": "CSI_A50"},
    {"code": "159627", "name": "中证 A100ETF 华夏", "family_id": "CSI_A100"},
    {"code": "159800", "name": "中证 800ETF 鹏华", "family_id": "CSI_800"},
    {"code": "159531", "name": "中证 2000ETF 南方", "family_id": "CSI_2000"},
)
