"""
PerformanceAgent — selects and optimises the best trading strategy.

Given risk-annotated strategies and performance metrics the agent ranks
candidates and produces a final, optimised recommendation with specific
entry / exit parameters and portfolio weight suggestions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PerformanceAgent:
    """Select and optimise the top strategy from a list of candidates."""

    def __init__(self) -> None:
        self._llm = _build_llm()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def optimize_strategy(
        self,
        strategies: list[dict[str, Any]],
        risk_metrics: dict[str, Any],
        rag_context: str,
    ) -> dict[str, Any]:
        """
        Rank candidates and return an optimised final strategy.

        Parameters
        ----------
        strategies:
            Risk-annotated strategies from :class:`~agents.risk_agent.RiskAgent`.
        risk_metrics:
            Output from :func:`~tools.risk_metrics.calculate_risk_metrics`.
        rag_context:
            RAG snippets about performance optimisation.

        Returns
        -------
        A single optimised strategy dict with ``optimisation`` sub-dict.
        """
        if not strategies:
            return {"error": "No strategies to optimise."}

        # Rank by composite score
        ranked = _rank_strategies(strategies, risk_metrics)
        top = ranked[0]

        if self._llm is None:
            return _apply_quant_optimisation(top, risk_metrics)

        prompt = _build_optimise_prompt(ranked, risk_metrics, rag_context)
        try:
            response = self._llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            llm_optimisation = _parse_optimisation(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM optimisation failed: %s — using quant fallback", exc)
            llm_optimisation = {}

        result = dict(top)
        quant_opt = _apply_quant_optimisation(top, risk_metrics)
        result["optimisation"] = {
            **quant_opt.get("optimisation", {}),
            **llm_optimisation,
        }
        result["rank"] = 1
        result["all_strategies_ranked"] = [s.get("name", f"Strategy {i+1}") for i, s in enumerate(ranked)]
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank_strategies(
    strategies: list[dict],
    metrics: dict,
) -> list[dict]:
    """
    Sort strategies by a composite score:
    - +2 for BUY/SELL signal (actionable)
    - +1 for low risk level
    - -1 for high risk level
    - +1 for high confidence
    - Adjusted by Sharpe ratio when available
    """
    sharpe = metrics.get("sharpe_ratio", 0) or 0

    def score(s: dict) -> float:
        sc = 0.0
        signal = s.get("signal", "HOLD").upper()
        if signal in ("BUY", "SELL", "SHORT"):
            sc += 2
        risk_level = s.get("risk_assessment", {}).get("risk_level", "medium")
        if risk_level == "low":
            sc += 1
        elif risk_level == "high":
            sc -= 1
        confidence = s.get("confidence", "medium").lower()
        if confidence == "high":
            sc += 1
        elif confidence == "low":
            sc -= 0.5
        # Reward positive Sharpe
        sc += max(0.0, min(sharpe, 2.0))
        return sc

    return sorted(strategies, key=score, reverse=True)


def _apply_quant_optimisation(strategy: dict, metrics: dict) -> dict:
    """Attach rule-based optimisation parameters to a strategy."""
    s = dict(strategy)
    vol = metrics.get("annualised_volatility") or 0.20
    sharpe = metrics.get("sharpe_ratio") or 0

    # Target volatility position sizing: aim for 15% annualised vol contribution
    target_vol = 0.15
    position_size_pct = min(round(target_vol / vol * 100, 1), 25.0) if vol > 0 else 5.0

    # Stop loss: 1.5× daily ATR proxy (approx. vol / sqrt(252))
    daily_vol = vol / (252 ** 0.5)
    stop_loss_pct = round(1.5 * daily_vol * 100, 2)

    # Take profit: 2× stop loss (2:1 reward-to-risk)
    take_profit_pct = round(stop_loss_pct * 2, 2)

    improvements: list[str] = []
    if sharpe < 1.0:
        improvements.append("Consider adding a volatility filter to reduce false signals.")
    if abs(metrics.get("max_drawdown", 0) or 0) > 0.20:
        improvements.append("Apply trailing stop-loss to limit drawdown exposure.")
    improvements.append(
        f"Target-volatility position sizing: {position_size_pct}% of portfolio."
    )

    s["optimisation"] = {
        "position_size_pct": position_size_pct,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "reward_to_risk_ratio": 2.0,
        "improvements": improvements,
        "source": "quantitative",
    }
    return s


def _build_llm():
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


def _build_optimise_prompt(
    strategies: list[dict],
    metrics: dict,
    rag_context: str,
) -> str:
    return f"""You are a senior quantitative portfolio manager optimising a trading strategy.

Performance Metrics for the underlying asset:
{json.dumps({k: v for k, v in metrics.items() if k != "error"}, indent=2)}

Relevant Context:
{rag_context or 'No additional context available.'}

Ranked Strategies (best first):
{json.dumps(strategies, indent=2)}

Select the best strategy and return a JSON object (not an array) with optimisation parameters:
  - selected_strategy_name (string): name of the chosen strategy
  - position_size_pct (number): % of portfolio for this trade (0–25)
  - stop_loss_pct (number): stop-loss % from entry
  - take_profit_pct (number): take-profit % from entry
  - reward_to_risk_ratio (number): take_profit / stop_loss
  - rebalance_frequency (string): "daily" | "weekly" | "monthly"
  - improvements (array of strings): suggested enhancements
  - source (string): "llm"

Return ONLY the JSON object with no additional text.
"""


def _parse_optimisation(content: str) -> dict[str, Any]:
    content = content.strip()
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(content[start:end])
    except json.JSONDecodeError:
        return {}
