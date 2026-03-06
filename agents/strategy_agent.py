"""
StrategyAgent — generates trading strategies with LLM assistance.

The agent assembles market context (price data, technical indicators,
RAG-retrieved financial knowledge) and asks the LLM to produce a
structured set of candidate trading strategies.

When no OpenAI key is available the agent falls back to a deterministic
rule-based strategy derived purely from the technical indicators.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class StrategyAgent:
    """Generate candidate trading strategies from market context."""

    def __init__(self) -> None:
        self._llm = _build_llm()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_strategies(
        self,
        ticker: str,
        market_data: dict[str, Any],
        technical_indicators: dict[str, Any],
        rag_context: str,
    ) -> list[dict[str, Any]]:
        """
        Produce a list of candidate strategies.

        Parameters
        ----------
        ticker:
            The equity symbol being analysed.
        market_data:
            Output from ``tools.financial_data.get_stock_price_data``.
        technical_indicators:
            Output from ``tools.technical_indicators.calculate_technical_indicators``.
        rag_context:
            Relevant snippets retrieved from the financial knowledge base.

        Returns
        -------
        List of strategy dicts, each containing at minimum:
        ``name``, ``type``, ``signal``, ``entry_condition``,
        ``exit_condition``, ``rationale``.
        """
        if self._llm is None:
            return self._rule_based_strategies(ticker, technical_indicators)

        prompt = _build_strategy_prompt(
            ticker, market_data, technical_indicators, rag_context
        )
        try:
            response = self._llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return _parse_strategies(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM strategy generation failed: %s — using rule-based fallback", exc)
            return self._rule_based_strategies(ticker, technical_indicators)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rule_based_strategies(
        ticker: str,
        technical_indicators: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Deterministic fallback: derive strategies from indicator signals."""
        latest = technical_indicators.get("latest", {})
        trend = technical_indicators.get("trend_signal", "neutral")
        rsi = latest.get("rsi_14")
        sma20 = latest.get("sma_20")
        sma50 = latest.get("sma_50")
        macd = latest.get("macd")
        macd_sig = latest.get("macd_signal")

        strategies: list[dict[str, Any]] = []

        # --- Momentum strategy ---
        if trend == "bullish":
            strategies.append({
                "name": "Bullish Momentum",
                "type": "momentum",
                "signal": "BUY",
                "entry_condition": (
                    f"SMA20 ({sma20}) > SMA50 ({sma50}) — golden cross confirmed."
                ),
                "exit_condition": "SMA20 crosses below SMA50 (death cross).",
                "rationale": (
                    "Short-term average is above long-term average, indicating "
                    "upward price momentum."
                ),
                "confidence": "medium",
            })
        elif trend == "bearish":
            strategies.append({
                "name": "Bearish Momentum / Short",
                "type": "momentum",
                "signal": "SELL/SHORT",
                "entry_condition": (
                    f"SMA20 ({sma20}) < SMA50 ({sma50}) — death cross confirmed."
                ),
                "exit_condition": "SMA20 crosses above SMA50 (golden cross).",
                "rationale": (
                    "Short-term average below long-term average signals downward momentum."
                ),
                "confidence": "medium",
            })

        # --- Mean-reversion strategy based on RSI ---
        if rsi is not None:
            if rsi < 30:
                strategies.append({
                    "name": "RSI Oversold Mean Reversion",
                    "type": "mean_reversion",
                    "signal": "BUY",
                    "entry_condition": f"RSI ({rsi:.1f}) < 30 — oversold territory.",
                    "exit_condition": "RSI crosses above 50.",
                    "rationale": (
                        "Extreme oversold reading suggests near-term price recovery."
                    ),
                    "confidence": "medium",
                })
            elif rsi > 70:
                strategies.append({
                    "name": "RSI Overbought Mean Reversion",
                    "type": "mean_reversion",
                    "signal": "SELL",
                    "entry_condition": f"RSI ({rsi:.1f}) > 70 — overbought territory.",
                    "exit_condition": "RSI crosses below 50.",
                    "rationale": (
                        "Extreme overbought reading suggests near-term price pullback."
                    ),
                    "confidence": "medium",
                })

        # --- MACD crossover ---
        if macd is not None and macd_sig is not None:
            if macd > macd_sig:
                strategies.append({
                    "name": "MACD Bullish Crossover",
                    "type": "momentum",
                    "signal": "BUY",
                    "entry_condition": f"MACD ({macd:.4f}) > Signal ({macd_sig:.4f}).",
                    "exit_condition": "MACD crosses below signal line.",
                    "rationale": "Bullish MACD crossover indicates accelerating momentum.",
                    "confidence": "medium",
                })
            else:
                strategies.append({
                    "name": "MACD Bearish Crossover",
                    "type": "momentum",
                    "signal": "SELL",
                    "entry_condition": f"MACD ({macd:.4f}) < Signal ({macd_sig:.4f}).",
                    "exit_condition": "MACD crosses above signal line.",
                    "rationale": "Bearish MACD crossover indicates decelerating momentum.",
                    "confidence": "medium",
                })

        if not strategies:
            strategies.append({
                "name": "Neutral / Hold",
                "type": "neutral",
                "signal": "HOLD",
                "entry_condition": "No clear signal detected.",
                "exit_condition": "Wait for decisive indicator crossover.",
                "rationale": "Mixed or insufficient signals; no trade recommended.",
                "confidence": "low",
            })

        return strategies


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _build_llm():
    """Return an LLM instance or None if no API key is configured."""
    try:
        from config import settings
        from langchain_openai import ChatOpenAI  # type: ignore

        if not settings.openai_api_key:
            return None
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
        )
    except Exception:  # noqa: BLE001
        return None


def _build_strategy_prompt(
    ticker: str,
    market_data: dict,
    technical_indicators: dict,
    rag_context: str,
) -> str:
    latest = technical_indicators.get("latest", {})
    trend = technical_indicators.get("trend_signal", "unknown")
    latest_close = market_data.get("latest_close", "N/A")

    return f"""You are an expert quantitative trading strategy analyst.

Ticker: {ticker}
Latest Close: {latest_close}
Technical Indicators (latest values):
{json.dumps(latest, indent=2)}
Overall Trend Signal: {trend}

Relevant Financial Context (from knowledge base):
{rag_context or 'No additional context available.'}

Based on the above data, generate 2–4 actionable trading strategies.
Return a valid JSON array where each element is an object with the following keys:
  - name (string): Short strategy name
  - type (string): "momentum" | "mean_reversion" | "breakout" | "neutral"
  - signal (string): "BUY" | "SELL" | "HOLD" | "SHORT"
  - entry_condition (string): Specific entry condition
  - exit_condition (string): Specific exit condition
  - rationale (string): Brief explanation
  - confidence (string): "high" | "medium" | "low"

Return ONLY the JSON array with no additional text.
"""


def _parse_strategies(content: str) -> list[dict[str, Any]]:
    """Extract and parse JSON array from LLM response."""
    content = content.strip()
    # Find JSON array in response
    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        logger.warning("LLM did not return a JSON array; got: %s", content[:200])
        return []
    try:
        return json.loads(content[start:end])
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error in strategy response: %s", exc)
        return []
