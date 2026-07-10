from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fieldguide_ai.ingestion.models import DocumentChunk


@dataclass(frozen=True)
class VectorSearchResult:
    """A provider-neutral vector search result."""

    chunk_id: str
    content: str
    metadata: dict[str, Any]
    distance: float


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts in input order."""


class VectorStore(ABC):
    @abstractmethod
    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Insert or update chunks by chunk ID."""

    @abstractmethod
    def replace_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Replace all indexed chunks for the supplied documents."""

    @abstractmethod
    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        """Delete every chunk belonging to the supplied document IDs."""

    @abstractmethod
    def query(self, query_text: str, n_results: int = 10) -> list[VectorSearchResult]:
        """Return the nearest indexed chunks in nearest-first order."""


def validate_embeddings(
    embeddings: Sequence[Sequence[float]], expected_count: int
) -> None:
    if len(embeddings) != expected_count:
        raise ValueError(
            f"embedding provider returned {len(embeddings)} embeddings for {expected_count} texts"
        )
    if embeddings and not embeddings[0]:
        raise ValueError("embedding vectors cannot be empty")
    dimensions = {len(embedding) for embedding in embeddings}
    if len(dimensions) > 1:
        raise ValueError("embedding vectors must have consistent dimensions")
