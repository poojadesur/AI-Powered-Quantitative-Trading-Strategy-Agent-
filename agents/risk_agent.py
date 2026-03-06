"""
RiskAgent — evaluates risk for each candidate strategy.

Combines quantitative risk metrics (Sharpe, VaR, drawdown …) with
LLM-generated qualitative commentary to produce a risk score and
actionable risk-management recommendations.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RiskAgent:
    """Evaluate and score the risk profile of candidate trading strategies."""

    def __init__(self) -> None:
        self._llm = _build_llm()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze_risk(
        self,
        strategies: list[dict[str, Any]],
        risk_metrics: dict[str, Any],
        rag_context: str,
    ) -> list[dict[str, Any]]:
        """
        Annotate each strategy with a risk assessment.

        Parameters
        ----------
        strategies:
            Candidate strategies from :class:`~agents.strategy_agent.StrategyAgent`.
        risk_metrics:
            Output from :func:`~tools.risk_metrics.calculate_risk_metrics`.
        rag_context:
            RAG snippets about risk management.

        Returns
        -------
        A new list of strategy dicts with added ``risk_assessment`` sub-dict.
        """
        if not strategies:
            return []

        # Quantitative risk gate
        quant_risk_level = _quant_risk_level(risk_metrics)

        if self._llm is None:
            return [
                _attach_quant_risk(s, risk_metrics, quant_risk_level)
                for s in strategies
            ]

        # LLM-enhanced risk assessment
        prompt = _build_risk_prompt(strategies, risk_metrics, rag_context)
        try:
            response = self._llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            llm_assessments = _parse_risk_assessments(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM risk analysis failed: %s — using quant fallback", exc)
            llm_assessments = []

        # Merge LLM assessment into strategy dicts
        result = []
        for i, strategy in enumerate(strategies):
            s = dict(strategy)
            if i < len(llm_assessments):
                s["risk_assessment"] = {
                    **_attach_quant_risk(s, risk_metrics, quant_risk_level)["risk_assessment"],
                    **llm_assessments[i],
                }
            else:
                s = _attach_quant_risk(s, risk_metrics, quant_risk_level)
            result.append(s)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quant_risk_level(metrics: dict[str, Any]) -> str:
    """Derive a simple risk level label from quantitative metrics."""
    if metrics.get("error"):
        return "unknown"
    sharpe = metrics.get("sharpe_ratio", 0) or 0
    max_dd = abs(metrics.get("max_drawdown", 0) or 0)
    vol = metrics.get("annualised_volatility", 0) or 0

    score = 0
    if sharpe > 1.5:
        score += 1
    elif sharpe < 0.5:
        score -= 1

    if max_dd > 0.30:
        score -= 2
    elif max_dd > 0.15:
        score -= 1

    if vol > 0.40:
        score -= 1
    elif vol < 0.15:
        score += 1

    if score >= 1:
        return "low"
    if score <= -2:
        return "high"
    return "medium"


def _attach_quant_risk(
    strategy: dict,
    metrics: dict,
    risk_level: str,
) -> dict:
    """Attach a quant-derived risk_assessment block to a strategy dict."""
    s = dict(strategy)
    sharpe = metrics.get("sharpe_ratio")
    max_dd = metrics.get("max_drawdown")
    var = metrics.get("var_historical")

    recommendations = []
    if max_dd is not None and abs(max_dd) > 0.20:
        recommendations.append(
            "Consider tighter stop-loss due to high historical drawdown."
        )
    if sharpe is not None and sharpe < 1.0:
        recommendations.append(
            "Low Sharpe ratio — evaluate position sizing carefully."
        )
    if var is not None:
        recommendations.append(
            f"Daily VaR (95%): {var:.2%}. Ensure position size respects risk budget."
        )

    s["risk_assessment"] = {
        "risk_level": risk_level,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "var_historical": var,
        "annualised_volatility": metrics.get("annualised_volatility"),
        "recommendations": recommendations or ["Risk profile is acceptable."],
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


def _build_risk_prompt(
    strategies: list[dict],
    metrics: dict,
    rag_context: str,
) -> str:
    return f"""You are a quantitative risk analyst evaluating trading strategies.

Risk Metrics for the underlying asset:
{json.dumps({k: v for k, v in metrics.items() if k != "error"}, indent=2)}

Relevant Risk Management Context:
{rag_context or 'No additional context available.'}

Strategies to evaluate:
{json.dumps(strategies, indent=2)}

For each strategy, provide a risk assessment JSON object with keys:
  - risk_level: "high" | "medium" | "low"
  - key_risks: list of strings describing main risks
  - recommendations: list of strings with risk-management recommendations
  - position_size_pct: suggested maximum position size as % of portfolio (number 0–100)
  - stop_loss_pct: suggested stop-loss % from entry (number)
  - source: "llm"

Return a JSON array (same length as strategies, in the same order).
Return ONLY the JSON array with no additional text.
"""


def _parse_risk_assessments(content: str) -> list[dict[str, Any]]:
    content = content.strip()
    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        return json.loads(content[start:end])
    except json.JSONDecodeError:
        return []
