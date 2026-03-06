"""
MCP Server — Model Context Protocol server exposing financial tools.

The server is implemented with the official ``mcp`` Python SDK and exposes
the following tools that any MCP-compatible client (including LangGraph
agents) can call:

* ``get_stock_price_data``      — historical OHLCV data via yfinance
* ``get_market_summary``        — quick snapshot for a list of tickers
* ``calculate_technical_indicators`` — SMA, EMA, RSI, MACD, BB, ATR, OBV
* ``calculate_risk_metrics``    — Sharpe, Sortino, VaR, CVaR, MDD, …
* ``search_financial_context``  — RAG retrieval from the document store

Run this module directly to start the MCP server over stdio::

    python mcp_server.py

Or import :func:`create_mcp_server` to embed it in another process.
"""

from __future__ import annotations

import json
import logging

from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.types as types

from tools.financial_data import get_stock_price_data, get_market_summary
from tools.risk_metrics import calculate_risk_metrics
from tools.technical_indicators import calculate_technical_indicators
from rag.retriever import FinancialContextRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared retriever (singleton within this process)
# ---------------------------------------------------------------------------
_retriever: FinancialContextRetriever | None = None


def _get_retriever() -> FinancialContextRetriever:
    global _retriever  # noqa: PLW0603
    if _retriever is None:
        _retriever = FinancialContextRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_mcp_server() -> Server:
    """Instantiate and configure the MCP server with all trading tools."""
    server = Server("trading-strategy-agent")

    # ------------------------------------------------------------------ #
    # Tool listing
    # ------------------------------------------------------------------ #

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_stock_price_data",
                description=(
                    "Fetch historical OHLCV (Open/High/Low/Close/Volume) data for a "
                    "stock ticker from Yahoo Finance."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Equity or ETF symbol, e.g. 'AAPL'.",
                        },
                        "lookback_days": {
                            "type": "integer",
                            "description": "Number of calendar days of history (default 252).",
                            "default": 252,
                        },
                        "interval": {
                            "type": "string",
                            "description": "Bar interval: '1d', '1h', '15m', etc. (default '1d').",
                            "default": "1d",
                        },
                    },
                    "required": ["ticker"],
                },
            ),
            types.Tool(
                name="get_market_summary",
                description=(
                    "Return a quick price snapshot (last price, previous close, "
                    "change%) for a list of tickers."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of ticker symbols.",
                        },
                    },
                    "required": ["tickers"],
                },
            ),
            types.Tool(
                name="calculate_technical_indicators",
                description=(
                    "Compute technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, "
                    "ATR, OBV) from a list of OHLCV records."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "records": {
                            "type": "array",
                            "description": "OHLCV records as returned by get_stock_price_data.",
                        },
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of indicator names to compute. "
                                "Omit to compute all."
                            ),
                        },
                    },
                    "required": ["records"],
                },
            ),
            types.Tool(
                name="calculate_risk_metrics",
                description=(
                    "Compute risk metrics (Sharpe, Sortino, Calmar, max drawdown, VaR, "
                    "CVaR, CAGR) from OHLCV records."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "records": {
                            "type": "array",
                            "description": "OHLCV records as returned by get_stock_price_data.",
                        },
                        "risk_free_rate": {
                            "type": "number",
                            "description": "Annualised risk-free rate (decimal). Default 0.05.",
                            "default": 0.05,
                        },
                        "confidence_level": {
                            "type": "number",
                            "description": "Confidence level for VaR (0–1). Default 0.95.",
                            "default": 0.95,
                        },
                    },
                    "required": ["records"],
                },
            ),
            types.Tool(
                name="search_financial_context",
                description=(
                    "Search the financial knowledge base using semantic similarity (RAG) "
                    "to retrieve relevant strategy, risk, or market context."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about financial topics.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of documents to retrieve (default 4).",
                            "default": 4,
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict,
    ) -> list[types.TextContent]:
        try:
            result = _dispatch_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s raised an error", name)
            result = {"error": str(exc)}

        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    return server


def _dispatch_tool(name: str, args: dict) -> object:
    """Route a tool call to the appropriate handler."""
    if name == "get_stock_price_data":
        return get_stock_price_data(
            ticker=args["ticker"],
            lookback_days=int(args.get("lookback_days", 252)),
            interval=str(args.get("interval", "1d")),
        )

    if name == "get_market_summary":
        return get_market_summary(tickers=list(args["tickers"]))

    if name == "calculate_technical_indicators":
        return calculate_technical_indicators(
            records=list(args["records"]),
            indicators=args.get("indicators"),
        )

    if name == "calculate_risk_metrics":
        return calculate_risk_metrics(
            records=list(args["records"]),
            risk_free_rate=float(args.get("risk_free_rate", 0.05)),
            confidence_level=float(args.get("confidence_level", 0.95)),
        )

    if name == "search_financial_context":
        retriever = _get_retriever()
        context = retriever.retrieve(
            query=str(args["query"]),
            k=int(args.get("k", 4)),
        )
        return {"query": args["query"], "context": context}

    raise ValueError(f"Unknown tool: {name!r}")


# ---------------------------------------------------------------------------
# Entry point — run as an MCP stdio server
# ---------------------------------------------------------------------------

async def _run_stdio() -> None:
    from mcp.server.stdio import stdio_server

    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        init_options = InitializationOptions(
            server_name="trading-strategy-agent",
            server_version="1.0.0",
            capabilities=server.get_capabilities(
                notification_options=None,
                experimental_capabilities={},
            ),
        )
        await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run_stdio())
