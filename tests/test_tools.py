"""
Tests for financial tools: financial_data, risk_metrics, technical_indicators.

These tests use synthetic / fixture data and do NOT make real network calls.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _generate_records(n: int = 300, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate synthetic OHLCV records for testing.

    Produces a simple trending price series so that indicator assertions
    are deterministic.
    """
    rng = random.Random(seed)
    price = 150.0
    records = []
    for i in range(n):
        change = rng.gauss(0.0003, 0.015)
        price = max(1.0, price * (1 + change))
        open_ = round(price * (1 + rng.gauss(0, 0.003)), 4)
        high = round(price * (1 + abs(rng.gauss(0, 0.005))), 4)
        low = round(price * (1 - abs(rng.gauss(0, 0.005))), 4)
        close = round(price, 4)
        volume = int(rng.uniform(1_000_000, 10_000_000))
        records.append(
            {
                "date": f"202{i // 365 + 1}-{(i % 365 // 30) + 1:02d}-{(i % 30) + 1:02d}",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return records


RECORDS = _generate_records(300)
SMALL_RECORDS = _generate_records(30)


# ---------------------------------------------------------------------------
# risk_metrics tests
# ---------------------------------------------------------------------------

class TestRiskMetrics:
    def test_basic_output_keys(self):
        from tools.risk_metrics import calculate_risk_metrics

        result = calculate_risk_metrics(RECORDS)
        assert result.get("error") is None
        for key in (
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "var_historical", "cvar", "annualised_volatility",
            "annualised_return", "cagr", "total_return",
        ):
            assert key in result, f"Missing key: {key}"

    def test_volatility_is_positive(self):
        from tools.risk_metrics import calculate_risk_metrics

        result = calculate_risk_metrics(RECORDS)
        assert result["annualised_volatility"] > 0

    def test_max_drawdown_is_non_positive(self):
        from tools.risk_metrics import calculate_risk_metrics

        result = calculate_risk_metrics(RECORDS)
        assert result["max_drawdown"] <= 0

    def test_var_is_negative(self):
        from tools.risk_metrics import calculate_risk_metrics

        result = calculate_risk_metrics(RECORDS)
        # VaR at 95% should be a loss (negative number)
        assert result["var_historical"] < 0

    def test_cvar_le_var(self):
        from tools.risk_metrics import calculate_risk_metrics

        result = calculate_risk_metrics(RECORDS)
        assert result["cvar"] <= result["var_historical"]

    def test_empty_records(self):
        from tools.risk_metrics import calculate_risk_metrics

        result = calculate_risk_metrics([])
        assert result.get("error") is not None

    def test_insufficient_records_graceful(self):
        from tools.risk_metrics import calculate_risk_metrics

        # Only 2 records — should not raise, but may return error or partial
        result = calculate_risk_metrics(RECORDS[:2])
        # Either error or valid output — must not raise
        assert isinstance(result, dict)

    def test_custom_risk_free_rate(self):
        from tools.risk_metrics import calculate_risk_metrics

        r1 = calculate_risk_metrics(RECORDS, risk_free_rate=0.0)
        r2 = calculate_risk_metrics(RECORDS, risk_free_rate=0.10)
        # Higher risk-free rate should reduce (or keep equal) Sharpe ratio
        if not (math.isnan(r1["sharpe_ratio"]) or math.isnan(r2["sharpe_ratio"])):
            assert r1["sharpe_ratio"] >= r2["sharpe_ratio"]

    def test_num_observations(self):
        from tools.risk_metrics import calculate_risk_metrics

        result = calculate_risk_metrics(RECORDS)
        # Should be one less than records (first return is NaN, dropped)
        assert result["num_observations"] == len(RECORDS) - 1


# ---------------------------------------------------------------------------
# technical_indicators tests
# ---------------------------------------------------------------------------

class TestTechnicalIndicators:
    def test_basic_output_keys(self):
        from tools.technical_indicators import calculate_technical_indicators

        result = calculate_technical_indicators(RECORDS)
        assert result.get("error") is None
        assert "latest" in result
        assert "series" in result
        assert "trend_signal" in result

    def test_all_default_indicators_present(self):
        from tools.technical_indicators import calculate_technical_indicators

        result = calculate_technical_indicators(RECORDS)
        expected = {
            "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "atr_14", "obv",
        }
        missing = expected - set(result["latest"].keys())
        assert not missing, f"Missing indicators: {missing}"

    def test_rsi_bounded(self):
        from tools.technical_indicators import calculate_technical_indicators

        result = calculate_technical_indicators(RECORDS)
        rsi = result["latest"]["rsi_14"]
        assert rsi is not None
        assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"

    def test_bollinger_band_ordering(self):
        from tools.technical_indicators import calculate_technical_indicators

        result = calculate_technical_indicators(RECORDS)
        upper = result["latest"]["bb_upper"]
        middle = result["latest"]["bb_middle"]
        lower = result["latest"]["bb_lower"]
        assert upper > middle > lower

    def test_sma_ordering(self):
        """SMA values should be present and be numbers."""
        from tools.technical_indicators import calculate_technical_indicators

        result = calculate_technical_indicators(RECORDS)
        for key in ("sma_20", "sma_50", "sma_200"):
            val = result["latest"].get(key)
            assert val is not None and val > 0, f"{key} = {val}"

    def test_trend_signal_is_valid(self):
        from tools.technical_indicators import calculate_technical_indicators

        result = calculate_technical_indicators(RECORDS)
        assert result["trend_signal"] in ("bullish", "bearish", "neutral")

    def test_selective_indicators(self):
        from tools.technical_indicators import calculate_technical_indicators

        result = calculate_technical_indicators(RECORDS, indicators=["rsi_14", "sma_20"])
        assert "rsi_14" in result["latest"]
        assert "sma_20" in result["latest"]
        # sma_200 should NOT be present
        assert "sma_200" not in result["latest"]

    def test_series_length(self):
        from tools.technical_indicators import calculate_technical_indicators

        result = calculate_technical_indicators(RECORDS, indicators=["sma_20"])
        series = result["series"]["sma_20"]
        assert len(series) == 10  # last 10 values

    def test_empty_records(self):
        from tools.technical_indicators import calculate_technical_indicators

        result = calculate_technical_indicators([])
        assert result.get("error") is not None


# ---------------------------------------------------------------------------
# financial_data tests (mock yfinance to avoid network calls)
# ---------------------------------------------------------------------------

class TestFinancialData:
    def test_get_stock_price_data_structure(self, monkeypatch):
        """Mock yfinance so no real HTTP request is made."""
        import pandas as pd
        from tools import financial_data as fd

        mock_df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [103.0, 104.0, 105.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [101.0, 102.0, 103.0],
                "Volume": [1_000_000, 1_100_000, 1_200_000],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        class MockFastInfo:
            last_price = 103.0
            previous_close = 102.0
            market_cap = 2_000_000_000

        class MockTicker:
            def history(self, **kwargs):
                return mock_df

            @property
            def fast_info(self):
                return MockFastInfo()

        monkeypatch.setattr(fd.yf, "Ticker", lambda _: MockTicker())

        result = fd.get_stock_price_data("AAPL", lookback_days=30)
        assert result["ticker"] == "AAPL"
        assert result["error"] is None
        assert len(result["records"]) == 3
        assert result["latest_close"] == 103.0

    def test_get_market_summary_structure(self, monkeypatch):
        import pandas as pd
        from tools import financial_data as fd

        class MockFastInfo:
            last_price = 200.0
            previous_close = 195.0
            market_cap = 3_000_000_000

        class MockTicker:
            @property
            def fast_info(self):
                return MockFastInfo()

        monkeypatch.setattr(fd.yf, "Ticker", lambda _: MockTicker())

        result = fd.get_market_summary(["AAPL", "MSFT"])
        assert "AAPL" in result
        assert "MSFT" in result
        assert result["AAPL"]["last_price"] == 200.0
        assert "change_pct" in result["AAPL"]

    def test_get_stock_price_data_empty_response(self, monkeypatch):
        import pandas as pd
        from tools import financial_data as fd

        class MockTicker:
            def history(self, **kwargs):
                return pd.DataFrame()

        monkeypatch.setattr(fd.yf, "Ticker", lambda _: MockTicker())

        result = fd.get_stock_price_data("FAKE")
        assert result["records"] == []
        assert result["error"] is not None
