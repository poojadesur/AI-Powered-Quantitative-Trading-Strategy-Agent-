"""
DocumentStore — FAISS-backed vector store for financial documents.

Supports:
* Adding plain-text documents (e.g., earnings reports, research notes).
* Persisting / loading the index from disk.
* Similarity search with a configurable ``k``.

The store is intentionally lightweight: it uses
``langchain_community.vectorstores.FAISS`` with the
``langchain_openai.OpenAIEmbeddings`` backend (or a fake in-memory
embedding when no API key is configured, useful for testing).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def _build_embeddings():
    """Return an embeddings object, falling back to a fake for tests."""
    from config import settings

    if settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings  # type: ignore

        return OpenAIEmbeddings(api_key=settings.openai_api_key)

    # Fallback: deterministic fake embeddings (no API call required)
    from langchain_community.embeddings import FakeEmbeddings  # type: ignore

    logger.warning(
        "OPENAI_API_KEY not set — using FakeEmbeddings. "
        "Set the key for production use."
    )
    return FakeEmbeddings(size=1536)


class DocumentStore:
    """Thin wrapper around a FAISS vector store for financial text."""

    def __init__(
        self,
        persist_path: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        from config import settings

        self._persist_path = Path(persist_path or settings.vector_store_path)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embeddings = _build_embeddings()
        self._store = None  # lazy init

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """
        Split *texts* into chunks and upsert into the vector store.

        Parameters
        ----------
        texts:
            Raw text strings (e.g. earnings report paragraphs).
        metadatas:
            Optional per-document metadata dicts (same length as *texts*).
        """
        from langchain_community.vectorstores import FAISS  # type: ignore

        if metadatas is None:
            metadatas = [{}] * len(texts)

        docs: list[Document] = []
        for text, meta in zip(texts, metadatas):
            chunks = self._splitter.split_text(text)
            docs.extend(Document(page_content=c, metadata=meta) for c in chunks)

        if not docs:
            return

        if self._store is None:
            self._store = FAISS.from_documents(docs, self._embeddings)
        else:
            self._store.add_documents(docs)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Return the *k* most relevant documents for *query*."""
        if self._store is None:
            return []
        return self._store.similarity_search(query, k=k)

    def save(self) -> None:
        """Persist the FAISS index to ``self._persist_path``."""
        if self._store is None:
            return
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._persist_path))
        logger.info("Vector store saved to %s", self._persist_path)

    def load(self) -> bool:
        """Load a previously persisted FAISS index. Returns True on success."""
        from langchain_community.vectorstores import FAISS  # type: ignore

        index_file = self._persist_path / "index.faiss"
        if not index_file.exists():
            return False
        try:
            self._store = FAISS.load_local(
                str(self._persist_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Vector store loaded from %s", self._persist_path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load vector store: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Built-in financial seed documents
    # ------------------------------------------------------------------ #

    def seed_financial_knowledge(self) -> None:
        """
        Pre-populate the store with domain-specific financial knowledge
        snippets so the RAG retriever has baseline context even without
        external data ingestion.
        """
        seed_texts = [
            (
                "The Sharpe ratio measures risk-adjusted return by dividing excess "
                "return over the risk-free rate by the portfolio standard deviation. "
                "A Sharpe ratio above 1 is generally considered good; above 2 is "
                "excellent for equity strategies."
            ),
            (
                "Value at Risk (VaR) estimates the maximum expected loss over a given "
                "time horizon at a specified confidence level. Historical VaR uses "
                "actual past returns, while parametric VaR assumes a normal distribution. "
                "CVaR (Conditional VaR / Expected Shortfall) gives the average loss "
                "beyond the VaR threshold."
            ),
            (
                "The MACD (Moving Average Convergence Divergence) indicator is computed "
                "as the difference between the 12-period and 26-period EMA. A buy signal "
                "occurs when the MACD line crosses above the signal line (9-period EMA of "
                "MACD). A sell signal occurs when MACD crosses below the signal line."
            ),
            (
                "Bollinger Bands consist of a 20-period SMA (middle band) and upper/lower "
                "bands at ±2 standard deviations. A close above the upper band suggests "
                "overbought conditions; below the lower band suggests oversold conditions. "
                "Squeeze patterns (narrow bands) often precede large moves."
            ),
            (
                "The RSI (Relative Strength Index) oscillates between 0 and 100. "
                "Readings above 70 indicate overbought conditions and potential reversal. "
                "Readings below 30 indicate oversold conditions. RSI divergence from price "
                "can signal weakening momentum."
            ),
            (
                "Maximum drawdown (MDD) is the largest peak-to-trough decline in portfolio "
                "value. It is a key measure of downside risk. The Calmar ratio divides "
                "annualised return by maximum drawdown; higher values indicate better "
                "risk-adjusted performance."
            ),
            (
                "Mean reversion strategies assume that asset prices tend to return to their "
                "long-run average. Common implementations include pairs trading, Bollinger "
                "Band fade trades, and statistical arbitrage. These strategies typically "
                "perform better in range-bound, low-trend markets."
            ),
            (
                "Trend-following strategies (momentum) buy assets that have been rising and "
                "sell assets that have been falling. Common signals include moving-average "
                "crossovers (e.g., 50-day crossing above 200-day — 'golden cross') and "
                "breakouts above resistance levels."
            ),
            (
                "Position sizing using Kelly Criterion: f* = (bp - q) / b, where b = odds, "
                "p = probability of win, q = 1 - p. In practice, traders use half-Kelly or "
                "quarter-Kelly to reduce variance while retaining most of the growth. "
                "Proper position sizing can significantly improve risk-adjusted returns."
            ),
            (
                "Backtesting a trading strategy involves simulating past performance on "
                "historical data. Common pitfalls include look-ahead bias, survivorship "
                "bias, overfitting, and ignoring transaction costs and slippage. "
                "Walk-forward analysis and out-of-sample testing mitigate overfitting risk."
            ),
        ]

        metadatas = [
            {"source": "financial_knowledge_base", "topic": topic}
            for topic in [
                "sharpe_ratio", "var_cvar", "macd", "bollinger_bands",
                "rsi", "drawdown", "mean_reversion", "trend_following",
                "position_sizing", "backtesting",
            ]
        ]

        self.add_documents(seed_texts, metadatas)
        logger.info("Seeded vector store with %d financial knowledge snippets.", len(seed_texts))
