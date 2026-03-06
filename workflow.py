"""
LangGraph Workflow — multi-step trading strategy generation pipeline.

The workflow is a directed graph with the following nodes:

    fetch_market_data
         │
         ▼
    retrieve_rag_context
         │
         ▼
    compute_indicators
         │
         ▼
    generate_strategies
         │
         ▼
    analyze_risk
         │
         ▼
    optimize_performance
         │
         ▼
       END

Each node operates on a shared :class:`TradingState` TypedDict.
MCP tool calls are wrapped inside the individual nodes so that every
tool invocation goes through the MCP dispatcher — keeping the orchestration
layer consistent with what the MCP server exposes.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from mcp_server import _dispatch_tool
from agents.strategy_agent import StrategyAgent
from agents.risk_agent import RiskAgent
from agents.performance_agent import PerformanceAgent
from rag.retriever import FinancialContextRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared workflow state
# ---------------------------------------------------------------------------

class TradingState(TypedDict, total=False):
    """Mutable state passed between workflow nodes."""
    # Input
    ticker: str
    lookback_days: int
    risk_free_rate: float

    # Data layer
    market_data: dict[str, Any]
    technical_indicators: dict[str, Any]
    risk_metrics: dict[str, Any]

    # RAG
    rag_context: str

    # Agent outputs
    strategies: list[dict[str, Any]]
    risk_annotated_strategies: list[dict[str, Any]]
    optimized_strategy: dict[str, Any]

    # Metadata
    errors: list[str]
    steps_completed: list[str]


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def _node_fetch_market_data(state: TradingState) -> TradingState:
    """Node 1 — fetch OHLCV data via MCP tool."""
    ticker = state.get("ticker", "AAPL")
    lookback_days = state.get("lookback_days", 252)
    logger.info("[workflow] fetch_market_data: %s (%d days)", ticker, lookback_days)

    result = _dispatch_tool(
        "get_stock_price_data",
        {"ticker": ticker, "lookback_days": lookback_days, "interval": "1d"},
    )
    errors = list(state.get("errors", []))
    steps = list(state.get("steps_completed", []))

    if result.get("error"):
        errors.append(f"fetch_market_data: {result['error']}")

    steps.append("fetch_market_data")
    return {**state, "market_data": result, "errors": errors, "steps_completed": steps}


def _node_retrieve_rag_context(state: TradingState) -> TradingState:
    """Node 2 — retrieve relevant financial context via RAG."""
    ticker = state.get("ticker", "AAPL")
    market_data = state.get("market_data", {})
    logger.info("[workflow] retrieve_rag_context for %s", ticker)

    # Build a query from available market data
    latest_close = market_data.get("latest_close", "unknown price")
    query = (
        f"Trading strategy for {ticker} at price {latest_close}. "
        "Risk management, technical indicators, momentum, mean reversion."
    )

    result = _dispatch_tool("search_financial_context", {"query": query, "k": 4})
    rag_context = result.get("context", "")

    steps = list(state.get("steps_completed", []))
    steps.append("retrieve_rag_context")
    return {**state, "rag_context": rag_context, "steps_completed": steps}


def _node_compute_indicators(state: TradingState) -> TradingState:
    """Node 3 — compute technical indicators via MCP tool."""
    market_data = state.get("market_data", {})
    ticker = state.get("ticker", "")
    records = market_data.get("records", [])
    logger.info("[workflow] compute_indicators for %s (%d records)", ticker, len(records))

    errors = list(state.get("errors", []))
    steps = list(state.get("steps_completed", []))

    if not records:
        errors.append("compute_indicators: no price records available")
        tech = {"error": "No records", "latest": {}, "series": {}, "trend_signal": "neutral"}
    else:
        tech = _dispatch_tool("calculate_technical_indicators", {"records": records})
        risk = _dispatch_tool(
            "calculate_risk_metrics",
            {
                "records": records,
                "risk_free_rate": state.get("risk_free_rate", 0.05),
                "confidence_level": 0.95,
            },
        )
        state = {**state, "risk_metrics": risk}

    steps.append("compute_indicators")
    return {**state, "technical_indicators": tech, "errors": errors, "steps_completed": steps}


def _node_generate_strategies(state: TradingState) -> TradingState:
    """Node 4 — generate candidate trading strategies."""
    ticker = state.get("ticker", "AAPL")
    logger.info("[workflow] generate_strategies for %s", ticker)

    agent = StrategyAgent()
    strategies = agent.generate_strategies(
        ticker=ticker,
        market_data=state.get("market_data", {}),
        technical_indicators=state.get("technical_indicators", {}),
        rag_context=state.get("rag_context", ""),
    )

    steps = list(state.get("steps_completed", []))
    steps.append("generate_strategies")
    return {**state, "strategies": strategies, "steps_completed": steps}


def _node_analyze_risk(state: TradingState) -> TradingState:
    """Node 5 — annotate each strategy with risk assessment."""
    ticker = state.get("ticker", "AAPL")
    logger.info("[workflow] analyze_risk for %s", ticker)

    # RAG context focused on risk management
    result = _dispatch_tool(
        "search_financial_context",
        {"query": "risk management stop loss position sizing drawdown VaR", "k": 3},
    )
    risk_rag = result.get("context", "")

    agent = RiskAgent()
    annotated = agent.analyze_risk(
        strategies=state.get("strategies", []),
        risk_metrics=state.get("risk_metrics", {}),
        rag_context=risk_rag,
    )

    steps = list(state.get("steps_completed", []))
    steps.append("analyze_risk")
    return {**state, "risk_annotated_strategies": annotated, "steps_completed": steps}


def _node_optimize_performance(state: TradingState) -> TradingState:
    """Node 6 — select and optimise the best strategy."""
    ticker = state.get("ticker", "AAPL")
    logger.info("[workflow] optimize_performance for %s", ticker)

    # RAG context focused on optimisation
    result = _dispatch_tool(
        "search_financial_context",
        {"query": "strategy optimisation performance Sharpe ratio backtesting position sizing", "k": 3},
    )
    perf_rag = result.get("context", "")

    agent = PerformanceAgent()
    optimized = agent.optimize_strategy(
        strategies=state.get("risk_annotated_strategies", []),
        risk_metrics=state.get("risk_metrics", {}),
        rag_context=perf_rag,
    )

    steps = list(state.get("steps_completed", []))
    steps.append("optimize_performance")
    return {**state, "optimized_strategy": optimized, "steps_completed": steps}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_workflow() -> StateGraph:
    """
    Assemble the LangGraph :class:`StateGraph` for the trading pipeline.

    Returns the compiled graph ready to be invoked via ``graph.invoke(state)``.
    """
    graph = StateGraph(TradingState)

    # Register nodes
    graph.add_node("fetch_market_data", _node_fetch_market_data)
    graph.add_node("retrieve_rag_context", _node_retrieve_rag_context)
    graph.add_node("compute_indicators", _node_compute_indicators)
    graph.add_node("generate_strategies", _node_generate_strategies)
    graph.add_node("analyze_risk", _node_analyze_risk)
    graph.add_node("optimize_performance", _node_optimize_performance)

    # Linear edges
    graph.set_entry_point("fetch_market_data")
    graph.add_edge("fetch_market_data", "retrieve_rag_context")
    graph.add_edge("retrieve_rag_context", "compute_indicators")
    graph.add_edge("compute_indicators", "generate_strategies")
    graph.add_edge("generate_strategies", "analyze_risk")
    graph.add_edge("analyze_risk", "optimize_performance")
    graph.add_edge("optimize_performance", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_trading_workflow(
    ticker: str,
    lookback_days: int = 252,
    risk_free_rate: float = 0.05,
) -> TradingState:
    """
    Execute the full trading strategy workflow for *ticker*.

    Parameters
    ----------
    ticker:
        Equity / ETF symbol, e.g. ``"AAPL"``.
    lookback_days:
        Historical window for market data and risk calculations.
    risk_free_rate:
        Annualised risk-free rate used in Sharpe / Sortino calculations.

    Returns
    -------
    Final :class:`TradingState` containing all intermediate and final results.
    """
    workflow = build_workflow()
    initial_state: TradingState = {
        "ticker": ticker,
        "lookback_days": lookback_days,
        "risk_free_rate": risk_free_rate,
        "errors": [],
        "steps_completed": [],
    }
    return workflow.invoke(initial_state)
