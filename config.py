"""
Configuration management for the AI Trading Strategy Agent.

Loads settings from environment variables / .env file.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    # ------------------------------------------------------------------ #
    # LLM / OpenAI
    # ------------------------------------------------------------------ #
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.0, alias="OPENAI_TEMPERATURE")

    # ------------------------------------------------------------------ #
    # MCP Server
    # ------------------------------------------------------------------ #
    mcp_host: str = Field(default="localhost", alias="MCP_HOST")
    mcp_port: int = Field(default=8765, alias="MCP_PORT")

    # ------------------------------------------------------------------ #
    # RAG / Vector Store
    # ------------------------------------------------------------------ #
    # Directory where the FAISS index is persisted between runs
    vector_store_path: str = Field(
        default="./data/vector_store", alias="VECTOR_STORE_PATH"
    )
    # Number of documents to retrieve per query
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")
    # Chunk size (characters) used when splitting documents
    rag_chunk_size: int = Field(default=500, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=50, alias="RAG_CHUNK_OVERLAP")

    # ------------------------------------------------------------------ #
    # Trading defaults
    # ------------------------------------------------------------------ #
    default_lookback_days: int = Field(default=252, alias="DEFAULT_LOOKBACK_DAYS")
    default_risk_free_rate: float = Field(
        default=0.05, alias="DEFAULT_RISK_FREE_RATE"
    )
    default_confidence_level: float = Field(
        default=0.95, alias="DEFAULT_CONFIDENCE_LEVEL"
    )
    default_tickers: list[str] = Field(
        default=["AAPL", "MSFT", "GOOGL"],
        alias="DEFAULT_TICKERS",
    )

    # ------------------------------------------------------------------ #
    # Pydantic settings
    # ------------------------------------------------------------------ #
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }


# Singleton — import this everywhere
settings = Settings()
