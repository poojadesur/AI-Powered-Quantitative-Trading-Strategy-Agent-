"""
Tests for the LangGraph workflow and agent nodes.

The tests mock yfinance so no real network calls are made.
The LLM is absent (no API key), so agent fallback paths are exercised.
"""

from __future__ import annotations

import random
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def _make_records(n: int = 260) -> list[dict[str, Any]]:
    from datetime import date, timedelta

    rng = random.Random(99)
    price = 200.0
    records = []
    start = date(2023, 1, 3)
    for i in range(n):
        price = max(1.0, price * (1 + rng.gauss(0.0002, 0.013)))
        records.append(
            {
                "date": (start + timedelta(days=i)).isoformat(),
                "open": round(price * 0.999, 4),
                "high": round(price * 1.005, 4),
                "low": round(price * 0.995, 4),
                "close": round(price, 4),
                "volume": int(rng.uniform(5_000_000, 20_000_000)),
            }
        )
    return records


RECORDS = _make_records()
MARKET_DATA = {
    "ticker": "TSLA",
    "interval": "1d",
    "records": RECORDS,
    "latest_close": RECORDS[-1]["close"],
    "latest_volume": RECORDS[-1]["volume"],
    "error": None,
}


# ---------------------------------------------------------------------------
# StrategyAgent tests
# ---------------------------------------------------------------------------

class TestStrategyAgent:
    def test_rule_based_returns_list(self):
        from tools.technical_indicators import calculate_technical_indicators
        from agents.strategy_agent import StrategyAgent

        tech = calculate_technical_indicators(RECORDS)
        agent = StrategyAgent()
        strategies = agent._rule_based_strategies("TSLA", tech)
        assert isinstance(strategies, list)
        assert len(strategies) >= 1

    def test_each_strategy_has_required_keys(self):
        from tools.technical_indicators import calculate_technical_indicators
        from agents.strategy_agent import StrategyAgent

        tech = calculate_technical_indicators(RECORDS)
        agent = StrategyAgent()
        strategies = agent._rule_based_strategies("TSLA", tech)
        required_keys = {"name", "type", "signal", "entry_condition", "exit_condition", "rationale"}
        for s in strategies:
            missing = required_keys - s.keys()
            assert not missing, f"Strategy missing keys: {missing}"

    def test_generate_strategies_with_no_llm(self):
        """generate_strategies should work without OpenAI key (LLM is None)."""
        from tools.technical_indicators import calculate_technical_indicators
        from agents.strategy_agent import StrategyAgent

        tech = calculate_technical_indicators(RECORDS)
        agent = StrategyAgent()
        agent._llm = None  # force fallback

        strategies = agent.generate_strategies(
            ticker="TSLA",
            market_data=MARKET_DATA,
            technical_indicators=tech,
            rag_context="RSI and MACD signals indicate a trend.",
        )
        assert isinstance(strategies, list)
        assert len(strategies) >= 1

    def test_signal_values_are_valid(self):
        from tools.technical_indicators import calculate_technical_indicators
        from agents.strategy_agent import StrategyAgent

        tech = calculate_technical_indicators(RECORDS)
        agent = StrategyAgent()
        agent._llm = None
        strategies = agent.generate_strategies("TSLA", MARKET_DATA, tech, "")
        valid_signals = {"BUY", "SELL", "HOLD", "SHORT", "SELL/SHORT"}
        for s in strategies:
            assert s.get("signal") in valid_signals, f"Invalid signal: {s.get('signal')}"


# ---------------------------------------------------------------------------
# RiskAgent tests
# ---------------------------------------------------------------------------

class TestRiskAgent:
    def _get_strategies(self):
        from tools.technical_indicators import calculate_technical_indicators
        from agents.strategy_agent import StrategyAgent

        tech = calculate_technical_indicators(RECORDS)
        agent = StrategyAgent()
        agent._llm = None
        return agent.generate_strategies("TSLA", MARKET_DATA, tech, "")

    def _get_risk_metrics(self):
        from tools.risk_metrics import calculate_risk_metrics
        return calculate_risk_metrics(RECORDS)

    def test_analyze_risk_returns_list(self):
        from agents.risk_agent import RiskAgent

        strategies = self._get_strategies()
        metrics = self._get_risk_metrics()
        agent = RiskAgent()
        agent._llm = None
        result = agent.analyze_risk(strategies, metrics, "")
        assert isinstance(result, list)
        assert len(result) == len(strategies)

    def test_risk_assessment_attached(self):
        from agents.risk_agent import RiskAgent

        strategies = self._get_strategies()
        metrics = self._get_risk_metrics()
        agent = RiskAgent()
        agent._llm = None
        result = agent.analyze_risk(strategies, metrics, "")
        for s in result:
            assert "risk_assessment" in s
            ra = s["risk_assessment"]
            assert "risk_level" in ra
            assert ra["risk_level"] in ("high", "medium", "low", "unknown")

    def test_risk_level_keys_present(self):
        from agents.risk_agent import RiskAgent

        strategies = self._get_strategies()
        metrics = self._get_risk_metrics()
        agent = RiskAgent()
        agent._llm = None
        result = agent.analyze_risk(strategies, metrics, "")
        for s in result:
            ra = s["risk_assessment"]
            assert "recommendations" in ra
            assert isinstance(ra["recommendations"], list)

    def test_empty_strategies(self):
        from agents.risk_agent import RiskAgent

        agent = RiskAgent()
        result = agent.analyze_risk([], {}, "")
        assert result == []


# ---------------------------------------------------------------------------
# PerformanceAgent tests
# ---------------------------------------------------------------------------

class TestPerformanceAgent:
    def _get_risk_annotated_strategies(self):
        from tools.technical_indicators import calculate_technical_indicators
        from tools.risk_metrics import calculate_risk_metrics
        from agents.strategy_agent import StrategyAgent
        from agents.risk_agent import RiskAgent

        tech = calculate_technical_indicators(RECORDS)
        metrics = calculate_risk_metrics(RECORDS)
        strategy_agent = StrategyAgent()
        strategy_agent._llm = None
        strategies = strategy_agent.generate_strategies("TSLA", MARKET_DATA, tech, "")
        risk_agent = RiskAgent()
        risk_agent._llm = None
        return risk_agent.analyze_risk(strategies, metrics, ""), metrics

    def test_optimize_returns_dict(self):
        from agents.performance_agent import PerformanceAgent

        strategies, metrics = self._get_risk_annotated_strategies()
        agent = PerformanceAgent()
        agent._llm = None
        result = agent.optimize_strategy(strategies, metrics, "")
        assert isinstance(result, dict)

    def test_optimisation_has_required_keys(self):
        from agents.performance_agent import PerformanceAgent

        strategies, metrics = self._get_risk_annotated_strategies()
        agent = PerformanceAgent()
        agent._llm = None
        result = agent.optimize_strategy(strategies, metrics, "")
        assert "optimisation" in result
        opt = result["optimisation"]
        for key in ("position_size_pct", "stop_loss_pct", "take_profit_pct"):
            assert key in opt, f"Missing optimisation key: {key}"

    def test_position_size_in_range(self):
        from agents.performance_agent import PerformanceAgent

        strategies, metrics = self._get_risk_annotated_strategies()
        agent = PerformanceAgent()
        agent._llm = None
        result = agent.optimize_strategy(strategies, metrics, "")
        pos_size = result["optimisation"]["position_size_pct"]
        assert 0 < pos_size <= 25

    def test_empty_strategies(self):
        from agents.performance_agent import PerformanceAgent

        agent = PerformanceAgent()
        result = agent.optimize_strategy([], {}, "")
        assert "error" in result


# ---------------------------------------------------------------------------
# Workflow tests (mocking yfinance)
# ---------------------------------------------------------------------------

class TestWorkflow:
    def _mock_yfinance(self, monkeypatch):
        import pandas as pd
        from tools import financial_data as fd

        mock_df = pd.DataFrame(
            {
                "Open": [r["open"] for r in RECORDS],
                "High": [r["high"] for r in RECORDS],
                "Low": [r["low"] for r in RECORDS],
                "Close": [r["close"] for r in RECORDS],
                "Volume": [r["volume"] for r in RECORDS],
            },
            index=pd.date_range("2023-01-01", periods=len(RECORDS), freq="D"),
        )

        class MockTicker:
            def history(self, **kwargs):
                return mock_df

        monkeypatch.setattr(fd.yf, "Ticker", lambda _: MockTicker())

    def test_workflow_runs_to_completion(self, monkeypatch):
        self._mock_yfinance(monkeypatch)
        from workflow import run_trading_workflow

        state = run_trading_workflow("TSLA", lookback_days=260)
        assert "optimized_strategy" in state
        assert "steps_completed" in state

    def test_workflow_completes_all_steps(self, monkeypatch):
        self._mock_yfinance(monkeypatch)
        from workflow import run_trading_workflow

        state = run_trading_workflow("TSLA", lookback_days=260)
        expected_steps = {
            "fetch_market_data",
            "retrieve_rag_context",
            "compute_indicators",
            "generate_strategies",
            "analyze_risk",
            "optimize_performance",
        }
        completed = set(state.get("steps_completed", []))
        assert expected_steps == completed, (
            f"Missing steps: {expected_steps - completed}"
        )

    def test_workflow_market_data_populated(self, monkeypatch):
        self._mock_yfinance(monkeypatch)
        from workflow import run_trading_workflow

        state = run_trading_workflow("TSLA", lookback_days=260)
        md = state.get("market_data", {})
        assert md.get("error") is None
        assert len(md.get("records", [])) > 0

    def test_workflow_strategies_non_empty(self, monkeypatch):
        self._mock_yfinance(monkeypatch)
        from workflow import run_trading_workflow

        state = run_trading_workflow("TSLA", lookback_days=260)
        strategies = state.get("risk_annotated_strategies", [])
        assert len(strategies) >= 1

    def test_workflow_optimized_strategy_has_optimisation(self, monkeypatch):
        self._mock_yfinance(monkeypatch)
        from workflow import run_trading_workflow

        state = run_trading_workflow("TSLA", lookback_days=260)
        opt = state.get("optimized_strategy", {})
        assert "optimisation" in opt

    def test_workflow_state_ticker_preserved(self, monkeypatch):
        self._mock_yfinance(monkeypatch)
        from workflow import run_trading_workflow

        state = run_trading_workflow("AAPL", lookback_days=260)
        assert state.get("ticker") == "AAPL"

    def test_build_workflow_returns_graph(self):
        from workflow import build_workflow

        graph = build_workflow()
        assert graph is not None
