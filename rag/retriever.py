"""
FinancialContextRetriever — RAG retriever for trading strategy context.

Wraps DocumentStore to provide a simple ``retrieve(query)`` interface
that returns formatted text snippets suitable for injection into LLM prompts.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.document_store import DocumentStore

logger = logging.getLogger(__name__)


class FinancialContextRetriever:
    """
    High-level retriever built on top of :class:`DocumentStore`.

    Usage example::

        retriever = FinancialContextRetriever()
        retriever.store.seed_financial_knowledge()
        context = retriever.retrieve("What RSI strategy works in bear markets?")
    """

    def __init__(
        self,
        store: DocumentStore | None = None,
        top_k: int | None = None,
    ) -> None:
        from config import settings

        self.store = store or DocumentStore()
        self._top_k = top_k or settings.rag_top_k

        # Try to load a persisted index; seed defaults if none exists
        if not self.store.load():
            self.store.seed_financial_knowledge()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, k: int | None = None) -> str:
        """
        Retrieve the most relevant financial context for *query*.

        Returns
        -------
        str
            Concatenated text of the top-k documents, separated by newlines.
            Returns an empty string if the store is empty.
        """
        k = k or self._top_k
        docs = self.store.similarity_search(query, k=k)
        if not docs:
            return ""
        return "\n\n".join(d.page_content for d in docs)

    def retrieve_as_list(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve documents as a list of dicts (content + metadata).
        Useful for structured output or debugging.
        """
        k = k or self._top_k
        docs = self.store.similarity_search(query, k=k)
        return [
            {"content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]

    def add_market_report(
        self,
        text: str,
        ticker: str = "UNKNOWN",
        report_type: str = "market_report",
    ) -> None:
        """
        Ingest a free-form market report or news article into the store.

        Parameters
        ----------
        text:
            Full text of the report.
        ticker:
            Equity symbol this report relates to (for metadata).
        report_type:
            Category tag (e.g. ``"earnings"``, ``"analyst_note"``).
        """
        self.store.add_documents(
            [text],
            [{"ticker": ticker, "type": report_type}],
        )
        logger.info("Ingested %s report for %s (%d chars)", report_type, ticker, len(text))
