"""ETF universe discovery and broad-index classification."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import (
    BROAD_INDEX_FAMILIES,
    ENHANCED_TERMS,
    FALLBACK_BROAD_ETF_SEEDS,
    STYLE_OR_THEME_TERMS,
)
from .data_sources import fetch_etf_spot, fetch_fundcode_search

LIQUIDITY_HINTS = {
    "510300": 100,
    "510310": 96,
    "510330": 92,
    "159919": 90,
    "510500": 100,
    "512500": 96,
    "159922": 88,
    "512100": 100,
    "159845": 96,
    "159629": 88,
    "159633": 86,
    "560010": 84,
    "159338": 90,
    "159353": 88,
    "159358": 86,
    "560510": 84,
    "560530": 82,
    "563860": 80,
}


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name or ""))
    text = text.upper().replace(" ", "")
    return text


def classify_etf_name(name: str) -> tuple[str | None, str | None, str | None]:
    """Classify a fund name into a broad CSI family.

    Returns ``(family_id, display_name, matched_pattern)``.  The caller decides
    whether to exclude style/enhanced variants.
    """

    normalized = normalize_name(name)
    if "ETF" not in normalized:
        return None, None, None
    for family in BROAD_INDEX_FAMILIES:
        for pattern in family.patterns:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                return family.family_id, family.display_name, pattern
    return None, None, None


def exclusion_reasons(name: str, include_enhanced: bool = False, include_style: bool = False) -> list[str]:
    normalized = normalize_name(name)
    reasons: list[str] = []
    if not include_enhanced and any(term.upper() in normalized for term in ENHANCED_TERMS):
        reasons.append("enhanced")
    if not include_style:
        matched_terms = [term for term in STYLE_OR_THEME_TERMS if term.upper() in normalized]
        if matched_terms:
            reasons.append("style_or_theme:" + "/".join(matched_terms[:4]))
    return reasons


def _family_metadata() -> pd.DataFrame:
    rows = []
    for family in BROAD_INDEX_FAMILIES:
        row = asdict(family)
        row.pop("patterns", None)
        rows.append(row)
    return pd.DataFrame(rows)


def fallback_universe() -> pd.DataFrame:
    meta = _family_metadata()
    df = pd.DataFrame(FALLBACK_BROAD_ETF_SEEDS).copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.merge(meta, on="family_id", how="left")
    df["source"] = "fallback_seed"
    df["amount"] = np.nan
    df["latest_price"] = np.nan
    df["rank_score"] = 0.0
    df["selected_rank_in_family"] = df.groupby("family_id").cumcount() + 1
    return df


def discover_broad_etfs(
    *,
    max_per_family: int | None = 3,
    min_amount: float = 0.0,
    include_enhanced: bool = False,
    include_style: bool = False,
    force: bool = False,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    """Discover listed CSI broad-based ETFs from the live ETF quote list.

    The function scans the entire ETF list, classifies candidates by broad index
    family, filters non-pure products, then ranks by current turnover amount.
    """

    try:
        spot = fetch_etf_spot(force=force)
        source = "eastmoney_spot"
    except Exception:
        try:
            spot = fetch_fundcode_search(force=force)
            source = "eastmoney_fundcode_search"
        except Exception:
            if not allow_fallback:
                raise
            return fallback_universe()

    rows = []
    for _, item in spot.iterrows():
        family_id, display_name, pattern = classify_etf_name(str(item.get("name", "")))
        if family_id is None:
            continue
        reasons = exclusion_reasons(
            str(item.get("name", "")),
            include_enhanced=include_enhanced,
            include_style=include_style,
        )
        if reasons:
            continue
        row = item.to_dict()
        row.update(
            {
                "family_id": family_id,
                "display_name": display_name,
                "matched_pattern": pattern,
                "source": source,
            }
        )
        rows.append(row)

    if not rows:
        if allow_fallback:
            return fallback_universe()
        raise RuntimeError("No CSI broad ETF candidates found after filtering")

    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype(str).str.zfill(6)
    if "amount" not in df.columns:
        df["amount"] = 0.0
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "latest_price" not in df.columns:
        df["latest_price"] = np.nan
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce").fillna(0.0)
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    df["latest_price"] = pd.to_numeric(df.get("latest_price"), errors="coerce")
    df = df[df["amount"] >= float(min_amount)].copy()
    if df.empty:
        if allow_fallback:
            return fallback_universe()
        raise RuntimeError("No CSI broad ETF candidates left after min_amount filter")

    df["rank_score"] = np.log1p(df["amount"].clip(lower=0)) + 0.15 * np.log1p(df["volume"].clip(lower=0))
    no_liquidity = (df["amount"] <= 0) & (df["volume"] <= 0)
    if no_liquidity.any():
        df.loc[no_liquidity, "rank_score"] = df.loc[no_liquidity, "code"].map(LIQUIDITY_HINTS).fillna(0.0)
    df = df.sort_values(["family_id", "rank_score", "amount"], ascending=[True, False, False])
    df["selected_rank_in_family"] = df.groupby("family_id").cumcount() + 1
    if max_per_family is not None and max_per_family > 0:
        df = df[df["selected_rank_in_family"] <= int(max_per_family)].copy()

    meta = _family_metadata()
    df = df.merge(meta, on="family_id", how="left")
    preferred_cols = [
        "code",
        "name",
        "family_id",
        "display_name_x",
        "display_name_y",
        "index_code",
        "description",
        "latest_price",
        "pct_chg",
        "amount",
        "volume",
        "turnover_rate",
        "rank_score",
        "selected_rank_in_family",
        "source",
        "matched_pattern",
        "fetched_at",
    ]
    df = df[[c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]]
    if "display_name_x" in df.columns:
        df["display_name"] = df["display_name_x"].fillna(df.get("display_name_y"))
        df = df.drop(columns=[c for c in ("display_name_x", "display_name_y") if c in df.columns])
    return df.sort_values(["family_id", "selected_rank_in_family"]).reset_index(drop=True)
