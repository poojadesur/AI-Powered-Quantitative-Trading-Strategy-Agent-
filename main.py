"""
main.py — Entry point for the AI-Powered Quantitative Trading Strategy Agent.

Usage
-----
Run interactively::

    python main.py

Or specify tickers directly::

    python main.py AAPL MSFT TSLA

The script executes the full LangGraph workflow for each ticker and
prints a formatted strategy report to stdout.
"""

from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _fmt_section(title: str, width: int = 70) -> str:
    return f"\n{'=' * width}\n{title.center(width)}\n{'=' * width}"


def _print_report(state: dict) -> None:
    ticker = state.get("ticker", "N/A")
    print(_fmt_section(f"Trading Strategy Report — {ticker}"))

    # Market snapshot
    md = state.get("market_data", {})
    if md and not md.get("error"):
        print(f"\n📈 Latest Close : ${md.get('latest_close', 'N/A'):,.2f}")
        print(f"   Records      : {len(md.get('records', []))} bars")

    # Risk metrics
    rm = state.get("risk_metrics", {})
    if rm and not rm.get("error"):
        print("\n📊 Risk Metrics:")
        for key in ("annualised_return", "annualised_volatility", "sharpe_ratio",
                    "sortino_ratio", "max_drawdown", "var_historical", "cagr"):
            val = rm.get(key)
            if val is not None:
                pct = key in ("annualised_return", "annualised_volatility",
                               "max_drawdown", "var_historical", "cagr")
                fmt = f"{val:.2%}" if pct else f"{val:.4f}"
                print(f"   {key:<28}: {fmt}")

    # Technical indicators
    ti = state.get("technical_indicators", {})
    if ti and not ti.get("error"):
        latest = ti.get("latest", {})
        trend = ti.get("trend_signal", "N/A")
        print(f"\n📉 Technical Indicators (trend: {trend.upper()}):")
        for key in ("sma_20", "sma_50", "rsi_14", "macd", "macd_signal"):
            val = latest.get(key)
            if val is not None:
                print(f"   {key:<20}: {val}")

    # RAG context
    rag = state.get("rag_context", "")
    if rag:
        preview = rag[:300].replace("\n", " ") + ("…" if len(rag) > 300 else "")
        print(f"\n🔍 RAG Context (preview): {preview}")

    # Generated strategies
    strategies = state.get("risk_annotated_strategies", state.get("strategies", []))
    if strategies:
        print(f"\n🗂️  Candidate Strategies ({len(strategies)}):")
        for i, s in enumerate(strategies, 1):
            ra = s.get("risk_assessment", {})
            print(f"\n  [{i}] {s.get('name', 'Unknown')}  [{s.get('signal', '?')}]")
            print(f"      Type      : {s.get('type', '?')}")
            print(f"      Entry     : {s.get('entry_condition', '?')}")
            print(f"      Exit      : {s.get('exit_condition', '?')}")
            print(f"      Risk Level: {ra.get('risk_level', '?')}")
            for rec in ra.get("recommendations", []):
                print(f"      ⚠  {rec}")

    # Optimised strategy
    opt = state.get("optimized_strategy", {})
    if opt and not opt.get("error"):
        print(_fmt_section("Optimised Final Strategy"))
        print(f"  Name          : {opt.get('name', 'N/A')}")
        print(f"  Signal        : {opt.get('signal', 'N/A')}")
        print(f"  Confidence    : {opt.get('confidence', 'N/A')}")
        print(f"  Entry         : {opt.get('entry_condition', 'N/A')}")
        print(f"  Exit          : {opt.get('exit_condition', 'N/A')}")
        optimisation = opt.get("optimisation", {})
        if optimisation:
            print("\n  Optimisation Parameters:")
            print(f"    Position Size : {optimisation.get('position_size_pct', 'N/A')}%")
            print(f"    Stop Loss     : {optimisation.get('stop_loss_pct', 'N/A')}%")
            print(f"    Take Profit   : {optimisation.get('take_profit_pct', 'N/A')}%")
            print(f"    R:R Ratio     : {optimisation.get('reward_to_risk_ratio', 'N/A')}")
            for imp in optimisation.get("improvements", []):
                print(f"    💡 {imp}")

    # Errors
    errors = state.get("errors", [])
    if errors:
        print("\n⚠️  Workflow Errors:")
        for err in errors:
            print(f"   - {err}")

    steps = state.get("steps_completed", [])
    print(f"\n✅ Steps completed: {' → '.join(steps)}")
    print("=" * 70)


def main(tickers: list[str] | None = None) -> None:
    from workflow import run_trading_workflow

    if tickers is None:
        if len(sys.argv) > 1:
            tickers = sys.argv[1:]
        else:
            from config import settings
            tickers = settings.default_tickers

    print(f"\n🤖 AI-Powered Quantitative Trading Strategy Agent")
    print(f"   Analysing: {', '.join(tickers)}\n")

    for ticker in tickers:
        print(f"\n⏳ Running workflow for {ticker} …")
        try:
            final_state = run_trading_workflow(
                ticker=ticker,
                lookback_days=252,
                risk_free_rate=0.05,
            )
            _print_report(dict(final_state))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Workflow failed for %s", ticker)
            print(f"\n❌ Workflow failed for {ticker}: {exc}")


if __name__ == "__main__":
    main()
