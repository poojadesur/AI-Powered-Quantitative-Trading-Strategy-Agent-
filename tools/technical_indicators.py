"""
Technical indicator calculation tools.

Computes momentum, trend, and volatility indicators using the ``ta``
library and plain NumPy/Pandas.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _df_from_records(records: list[dict]) -> pd.DataFrame:
    """Convert OHLCV records into a typed DataFrame."""
    df = pd.DataFrame(records)
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["close"])


def calculate_technical_indicators(
    records: list[dict],
    indicators: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute a collection of technical indicators.

    Parameters
    ----------
    records:
        List of OHLCV dicts (at minimum ``close`` is required).
    indicators:
        Subset of indicators to compute.  Pass ``None`` (default) to
        compute all supported indicators.

    Returns
    -------
    dict with indicator series (last 10 values) and latest values dict.
    """
    supported = {
        "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26",
        "rsi_14",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower",
        "atr_14",
        "obv",
    }
    if indicators is None:
        indicators = list(supported)

    df = _df_from_records(records)
    if df.empty:
        return {"error": "Empty or invalid records.", "latest": {}, "series": {}}

    result_series: dict[str, list] = {}
    latest: dict[str, float | None] = {}

    close = df["close"]
    high = df.get("high", close)
    low = df.get("low", close)
    volume = df.get("volume", pd.Series([0] * len(df)))

    def _to_list(s: pd.Series) -> list:
        return [round(float(v), 4) if pd.notna(v) else None for v in s.iloc[-10:]]

    def _latest(s: pd.Series) -> float | None:
        val = s.dropna().iloc[-1] if not s.dropna().empty else None
        return round(float(val), 4) if val is not None else None

    # ---- Simple / Exponential Moving Averages -------------------------
    if "sma_20" in indicators:
        s = close.rolling(20).mean()
        result_series["sma_20"] = _to_list(s)
        latest["sma_20"] = _latest(s)

    if "sma_50" in indicators:
        s = close.rolling(50).mean()
        result_series["sma_50"] = _to_list(s)
        latest["sma_50"] = _latest(s)

    if "sma_200" in indicators:
        s = close.rolling(200).mean()
        result_series["sma_200"] = _to_list(s)
        latest["sma_200"] = _latest(s)

    if "ema_12" in indicators:
        s = close.ewm(span=12, adjust=False).mean()
        result_series["ema_12"] = _to_list(s)
        latest["ema_12"] = _latest(s)

    if "ema_26" in indicators:
        s = close.ewm(span=26, adjust=False).mean()
        result_series["ema_26"] = _to_list(s)
        latest["ema_26"] = _latest(s)

    # ---- RSI -----------------------------------------------------------
    if "rsi_14" in indicators:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        result_series["rsi_14"] = _to_list(rsi)
        latest["rsi_14"] = _latest(rsi)

    # ---- MACD ----------------------------------------------------------
    if any(k in indicators for k in ("macd", "macd_signal", "macd_hist")):
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        if "macd" in indicators:
            result_series["macd"] = _to_list(macd_line)
            latest["macd"] = _latest(macd_line)
        if "macd_signal" in indicators:
            result_series["macd_signal"] = _to_list(signal_line)
            latest["macd_signal"] = _latest(signal_line)
        if "macd_hist" in indicators:
            result_series["macd_hist"] = _to_list(histogram)
            latest["macd_hist"] = _latest(histogram)

    # ---- Bollinger Bands -----------------------------------------------
    if any(k in indicators for k in ("bb_upper", "bb_middle", "bb_lower")):
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20

        if "bb_upper" in indicators:
            result_series["bb_upper"] = _to_list(upper)
            latest["bb_upper"] = _latest(upper)
        if "bb_middle" in indicators:
            result_series["bb_middle"] = _to_list(sma20)
            latest["bb_middle"] = _latest(sma20)
        if "bb_lower" in indicators:
            result_series["bb_lower"] = _to_list(lower)
            latest["bb_lower"] = _latest(lower)

    # ---- ATR -----------------------------------------------------------
    if "atr_14" in indicators:
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        result_series["atr_14"] = _to_list(atr)
        latest["atr_14"] = _latest(atr)

    # ---- OBV -----------------------------------------------------------
    if "obv" in indicators:
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (volume * direction).cumsum()
        result_series["obv"] = _to_list(obv)
        latest["obv"] = _latest(obv)

    # ---- Trend signal summary -----------------------------------------
    trend_signal = _compute_trend_signal(latest)

    return {
        "series": result_series,
        "latest": latest,
        "trend_signal": trend_signal,
        "num_records": len(df),
        "error": None,
    }


def _compute_trend_signal(latest: dict) -> str:
    """Derive a simple bullish/bearish/neutral composite signal."""
    signals = []

    # Golden cross / death cross
    sma20 = latest.get("sma_20")
    sma50 = latest.get("sma_50")
    if sma20 and sma50:
        signals.append("bullish" if sma20 > sma50 else "bearish")

    # RSI overbought / oversold
    rsi = latest.get("rsi_14")
    if rsi is not None:
        if rsi > 70:
            signals.append("overbought")
        elif rsi < 30:
            signals.append("oversold")
        else:
            signals.append("neutral")

    # MACD crossover
    macd = latest.get("macd")
    macd_sig = latest.get("macd_signal")
    if macd is not None and macd_sig is not None:
        signals.append("bullish" if macd > macd_sig else "bearish")

    if not signals:
        return "neutral"
    bull = signals.count("bullish") + signals.count("oversold")
    bear = signals.count("bearish") + signals.count("overbought")
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"
