"""
Financial data retrieval tools.

Uses yfinance to fetch OHLCV data and market summaries.
All functions return plain Python dicts so they are easily
serialisable and can be passed through LangGraph state.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def get_stock_price_data(
    ticker: str,
    lookback_days: int = 252,
    interval: str = "1d",
) -> dict[str, Any]:
    """
    Fetch historical OHLCV data for *ticker*.

    Parameters
    ----------
    ticker:
        Equity / ETF / crypto symbol recognised by Yahoo Finance.
    lookback_days:
        How many calendar days of history to retrieve.
    interval:
        yfinance interval string (``"1d"``, ``"1h"``, …).

    Returns
    -------
    dict with keys:
        ticker, interval, start, end, records (list of OHLCV dicts),
        latest_close, latest_volume, error (if any).
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        ticker_obj = yf.Ticker(ticker)
        df: pd.DataFrame = ticker_obj.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval=interval,
            auto_adjust=True,
        )

        if df.empty:
            return {
                "ticker": ticker,
                "interval": interval,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "records": [],
                "latest_close": None,
                "latest_volume": None,
                "error": "No data returned from Yahoo Finance.",
            }

        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "date"
        df = df.round(4)

        records = df.reset_index().assign(
            date=lambda x: x["date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records")

        return {
            "ticker": ticker,
            "interval": interval,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "records": records,
            "latest_close": float(df["close"].iloc[-1]),
            "latest_volume": int(df["volume"].iloc[-1]),
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error fetching data for %s", ticker)
        return {
            "ticker": ticker,
            "interval": interval,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "records": [],
            "latest_close": None,
            "latest_volume": None,
            "error": str(exc),
        }


def get_market_summary(tickers: list[str]) -> dict[str, Any]:
    """
    Retrieve a one-line snapshot for each ticker in *tickers*.

    Returns
    -------
    dict mapping each ticker to its latest close / change%.
    """
    summary: dict[str, Any] = {}
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.fast_info  # lightweight call
            summary[ticker] = {
                "last_price": round(float(info.last_price or 0), 4),
                "previous_close": round(float(info.previous_close or 0), 4),
                "change_pct": round(
                    (
                        (float(info.last_price or 0) - float(info.previous_close or 0))
                        / float(info.previous_close or 1)
                    )
                    * 100,
                    2,
                ),
                "market_cap": info.market_cap,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch summary for %s: %s", ticker, exc)
            summary[ticker] = {"error": str(exc)}
    return summary
