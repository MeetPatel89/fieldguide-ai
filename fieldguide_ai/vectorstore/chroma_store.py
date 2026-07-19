"""Chroma-backed vector-store implementation."""

from collections.abc import Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from fieldguide_ai.ingestion.models import DocumentChunk
from fieldguide_ai.vectorstore.base import (
    EmbeddingProvider,
    VectorSearchResult,
    VectorStore,
    validate_embeddings,
)
from fieldguide_ai.vectorstore.metadata import serialize_chunk_metadata

DEFAULT_COLLECTION_NAME = "documents"


class ChromaVectorStore(VectorStore):
    """Persist and search document chunks in a named Chroma collection."""

    def __init__(
        self,
        path: str,
        embedding_provider: EmbeddingProvider,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: ClientAPI | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.client = client or chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Insert or update chunks by chunk ID."""
        if not chunks:
            return
        embeddings = self._embed_chunks(chunks)
        self._upsert(chunks, embeddings)

    def replace_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Replace all indexed chunks for the supplied documents."""
        if not chunks:
            return

        # Embed before deleting old records so remote failures leave the index intact.
        embeddings = self._embed_chunks(chunks)
        self.delete_documents(_unique_doc_ids(chunks))
        self._upsert(chunks, embeddings)

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        """Delete every chunk belonging to the supplied document IDs."""
        for doc_id in dict.fromkeys(doc_ids):
            self.collection.delete(where={"doc_id": doc_id})

    def get_collection(self) -> Collection:
        """Return the underlying Chroma collection."""
        return self.collection

    def query(self, query_text: str, n_results: int = 10) -> list[VectorSearchResult]:
        """Return the nearest indexed chunks in nearest-first order."""
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")

        query_embeddings = self.embedding_provider.embed_texts([query_text])
        validate_embeddings(query_embeddings, 1)
        response = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        return [
            VectorSearchResult(
                chunk_id=chunk_id,
                content=documents[index] if index < len(documents) else "",
                metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
                distance=float(distances[index]) if index < len(distances) else 0.0,
            )
            for index, chunk_id in enumerate(ids)
        ]

    def _embed_chunks(self, chunks: Sequence[DocumentChunk]) -> list[list[float]]:
        embeddings = self.embedding_provider.embed_texts(
            [chunk.content for chunk in chunks]
        )
        validate_embeddings(embeddings, len(chunks))
        return embeddings

    def _upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=[list(embedding) for embedding in embeddings],
            metadatas=[serialize_chunk_metadata(chunk) for chunk in chunks],
        )


def _unique_doc_ids(chunks: Sequence[DocumentChunk]) -> list[str]:
    return list(dict.fromkeys(chunk.doc_id for chunk in chunks))
