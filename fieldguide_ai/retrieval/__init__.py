"""Compose shared catalog, vector search, and retrieval source records."""

from fieldguide_ai.retrieval.factory import (
    build_catalog,
    build_embedder,
    build_retriever,
    build_store,
    persist_store,
)
from fieldguide_ai.retrieval.modes import RetrievalMode, retriever_config_for
from fieldguide_ai.retrieval.settings import RetrievalSettings
from fieldguide_ai.retrieval.sources import (
    RetrievalDiagnostics,
    RetrievedSource,
    sources_from_result,
)

__all__ = [
    "RetrievalDiagnostics",
    "RetrievalMode",
    "RetrievalSettings",
    "RetrievedSource",
    "build_catalog",
    "build_embedder",
    "build_retriever",
    "build_store",
    "persist_store",
    "retriever_config_for",
    "sources_from_result",
]
