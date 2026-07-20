"""Chroma-backed vector-store implementation."""

from collections.abc import Sequence

import chromadb
from chromadb.api import ClientAPI

from fieldguide_ai.errors import VectorStoreOperationError
from fieldguide_ai.ingestion.models import DocumentChunk
from fieldguide_ai.vectorstore.base import (
    EmbeddingProvider,
    VectorSearchResult,
    VectorStore,
    validate_embeddings,
)
from fieldguide_ai.vectorstore.metadata import serialize_chunk_metadata

DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_CHROMA_PATH = "chroma_db"


class ChromaVectorStore(VectorStore):
    """Persist and search document chunks in a named Chroma collection."""

    def __init__(
        self,
        path: str,
        embedding_provider: EmbeddingProvider,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: ClientAPI | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        try:
            self._client = (
                client if client is not None else chromadb.PersistentClient(path=path)
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name
            )
        except Exception as error:
            raise VectorStoreOperationError(
                f"could not open Chroma collection {collection_name!r}"
            ) from error

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
            try:
                self._collection.delete(where={"doc_id": doc_id})
            except Exception as error:
                raise VectorStoreOperationError(
                    f"could not delete Chroma document {doc_id!r}"
                ) from error

    def query(self, query_text: str, n_results: int = 10) -> list[VectorSearchResult]:
        """Return the nearest indexed chunks in nearest-first order."""
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")

        query_embeddings = self._embedding_provider.embed_texts([query_text])
        validate_embeddings(query_embeddings, 1)
        try:
            response = self._collection.query(
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
                    metadata=(
                        dict(metadatas[index] or {}) if index < len(metadatas) else {}
                    ),
                    distance=(
                        float(distances[index]) if index < len(distances) else 0.0
                    ),
                )
                for index, chunk_id in enumerate(ids)
            ]
        except Exception as error:
            raise VectorStoreOperationError("Chroma query failed") from error

    def _embed_chunks(self, chunks: Sequence[DocumentChunk]) -> list[list[float]]:
        embeddings = self._embedding_provider.embed_texts(
            [chunk.content for chunk in chunks]
        )
        validate_embeddings(embeddings, len(chunks))
        return embeddings

    def _upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.content for chunk in chunks],
                embeddings=[list(embedding) for embedding in embeddings],
                metadatas=[serialize_chunk_metadata(chunk) for chunk in chunks],
            )
        except Exception as error:
            raise VectorStoreOperationError("Chroma upsert failed") from error


def _unique_doc_ids(chunks: Sequence[DocumentChunk]) -> list[str]:
    return list(dict.fromkeys(chunk.doc_id for chunk in chunks))
