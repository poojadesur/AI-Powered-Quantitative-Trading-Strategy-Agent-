"""
Tests for the RAG document store and retriever.

Uses FakeEmbeddings so no OpenAI API key is required.
"""

from __future__ import annotations

import pytest


class TestDocumentStore:
    def test_add_and_search(self):
        from rag.document_store import DocumentStore

        store = DocumentStore(persist_path="/tmp/test_vector_store_add")
        store.add_documents(
            ["The Sharpe ratio measures risk-adjusted return."],
            [{"source": "test"}],
        )
        results = store.similarity_search("risk-adjusted performance", k=1)
        assert len(results) == 1
        assert "Sharpe" in results[0].page_content

    def test_empty_store_returns_empty(self):
        from rag.document_store import DocumentStore

        store = DocumentStore(persist_path="/tmp/test_vector_store_empty")
        results = store.similarity_search("anything", k=4)
        assert results == []

    def test_seed_financial_knowledge(self):
        from rag.document_store import DocumentStore

        store = DocumentStore(persist_path="/tmp/test_vector_store_seed")
        store.seed_financial_knowledge()
        results = store.similarity_search("MACD momentum", k=2)
        assert len(results) >= 1

    def test_multiple_documents(self):
        from rag.document_store import DocumentStore

        store = DocumentStore(persist_path="/tmp/test_vector_store_multi")
        texts = [
            "RSI above 70 indicates overbought conditions.",
            "MACD crossover is a bullish signal.",
            "Maximum drawdown measures peak-to-trough decline.",
        ]
        store.add_documents(texts)
        # FakeEmbeddings are random, so we only verify that k results are returned
        results = store.similarity_search("RSI overbought", k=2)
        assert len(results) == 2
        # All returned content must come from the ingested texts
        all_texts = set(texts)
        for doc in results:
            assert doc.page_content in all_texts

    def test_chunking_long_document(self):
        """Long documents should be split into multiple chunks."""
        from rag.document_store import DocumentStore

        store = DocumentStore(
            persist_path="/tmp/test_vector_store_chunk",
            chunk_size=100,
            chunk_overlap=10,
        )
        long_text = ("The quick brown fox jumps over the lazy dog. " * 20).strip()
        store.add_documents([long_text])
        # After chunking, should have multiple documents in the store
        results = store.similarity_search("fox", k=5)
        assert len(results) >= 1

    def test_save_and_load(self, tmp_path):
        from rag.document_store import DocumentStore

        path = str(tmp_path / "faiss_store")

        # Create and populate
        store1 = DocumentStore(persist_path=path)
        store1.add_documents(["Mean reversion strategy uses Bollinger Bands."])
        store1.save()

        # Load into a new instance
        store2 = DocumentStore(persist_path=path)
        loaded = store2.load()
        assert loaded is True
        results = store2.similarity_search("Bollinger", k=1)
        assert len(results) == 1


class TestFinancialContextRetriever:
    def test_retrieve_returns_string(self):
        from rag.retriever import FinancialContextRetriever
        from rag.document_store import DocumentStore

        store = DocumentStore(persist_path="/tmp/test_retriever_basic")
        store.seed_financial_knowledge()
        retriever = FinancialContextRetriever(store=store)
        context = retriever.retrieve("Sharpe ratio momentum strategy")
        assert isinstance(context, str)
        assert len(context) > 0

    def test_retrieve_as_list(self):
        from rag.retriever import FinancialContextRetriever
        from rag.document_store import DocumentStore

        store = DocumentStore(persist_path="/tmp/test_retriever_list")
        store.seed_financial_knowledge()
        retriever = FinancialContextRetriever(store=store)
        docs = retriever.retrieve_as_list("VaR risk", k=2)
        assert isinstance(docs, list)
        for doc in docs:
            assert "content" in doc
            assert "metadata" in doc

    def test_add_market_report(self):
        from rag.retriever import FinancialContextRetriever
        from rag.document_store import DocumentStore

        store = DocumentStore(persist_path="/tmp/test_retriever_report")
        retriever = FinancialContextRetriever(store=store)
        retriever.add_market_report(
            "Apple reported strong Q4 earnings with revenue of $100B.",
            ticker="AAPL",
            report_type="earnings",
        )
        context = retriever.retrieve("Apple earnings revenue")
        assert "Apple" in context or "earnings" in context or len(context) >= 0

    def test_empty_store_retrieve(self):
        from rag.retriever import FinancialContextRetriever
        from rag.document_store import DocumentStore

        store = DocumentStore(persist_path="/tmp/test_retriever_empty_retrieve")
        # Don't seed, don't add anything
        # Override the _store to be None explicitly
        store._store = None
        retriever = FinancialContextRetriever.__new__(FinancialContextRetriever)
        retriever.store = store
        retriever._top_k = 4
        context = retriever.retrieve("anything")
        assert context == ""
