"""
Risk metrics calculation tools.

Computes Sharpe ratio, Sortino ratio, maximum drawdown,
Value-at-Risk (historical and parametric) and other metrics
from a list of OHLCV records as returned by *financial_data.get_stock_price_data*.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _records_to_returns(records: list[dict]) -> pd.Series:
    """Convert a list of OHLCV dicts into a daily-return Series."""
    df = pd.DataFrame(records)
    if df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    closes = pd.to_numeric(df["close"], errors="coerce").dropna()
    return closes.pct_change().dropna()


def calculate_risk_metrics(
    records: list[dict],
    risk_free_rate: float = 0.05,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """
    Compute standard portfolio / strategy risk metrics.

    Parameters
    ----------
    records:
        List of OHLCV dicts (``close`` key required).
    risk_free_rate:
        Annualised risk-free rate (decimal, e.g. 0.05 = 5 %).
    confidence_level:
        Confidence level for VaR, e.g. 0.95.

    Returns
    -------
    dict with annualised Sharpe, Sortino, max_drawdown, VaR,
    CVaR, volatility, total_return, and CAGR.
    """
    returns = _records_to_returns(records)

    if returns.empty:
        return {"error": "Insufficient data to calculate risk metrics."}

    trading_days = 252
    daily_rf = risk_free_rate / trading_days

    # Annualised metrics
    mean_return = float(returns.mean())
    std_return = float(returns.std())
    ann_return = mean_return * trading_days
    ann_vol = std_return * math.sqrt(trading_days)

    # Sharpe ratio
    excess = returns - daily_rf
    sharpe = (
        float(excess.mean() / excess.std() * math.sqrt(trading_days))
        if excess.std() > 0
        else float("nan")
    )

    # Sortino ratio (downside deviation)
    downside = returns[returns < 0]
    downside_std = float(downside.std()) if not downside.empty else 0.0
    sortino = (
        (ann_return - risk_free_rate) / (downside_std * math.sqrt(trading_days))
        if downside_std > 0
        else float("nan")
    )

    # Maximum Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Value-at-Risk (historical)
    var_historical = float(np.percentile(returns, (1 - confidence_level) * 100))

    # Value-at-Risk (parametric / Gaussian)
    from scipy.stats import norm  # lazy import to keep startup fast
    z = norm.ppf(1 - confidence_level)
    var_parametric = float(mean_return + z * std_return)

    # Conditional VaR (Expected Shortfall)
    cvar = float(returns[returns <= var_historical].mean())

    # Total return over the period
    total_return = float(cum_returns.iloc[-1] - 1) if not cum_returns.empty else 0.0

    # CAGR
    n_years = len(returns) / trading_days
    cagr = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    # Calmar ratio
    calmar = (
        cagr / abs(max_drawdown) if max_drawdown != 0 else float("nan")
    )

    return {
        "annualised_return": round(ann_return, 6),
        "annualised_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown": round(max_drawdown, 6),
        "var_historical": round(var_historical, 6),
        "var_parametric": round(var_parametric, 6),
        "cvar": round(cvar, 6),
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "risk_free_rate": risk_free_rate,
        "confidence_level": confidence_level,
        "num_observations": len(returns),
        "error": None,
    }
