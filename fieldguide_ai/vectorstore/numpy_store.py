import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fieldguide_ai.ingestion.models import DocumentChunk
from fieldguide_ai.vectorstore.base import (
    EmbeddingProvider,
    VectorSearchResult,
    VectorStore,
    validate_embeddings,
)
from fieldguide_ai.vectorstore.metadata import serialize_chunk_metadata


@dataclass(frozen=True)
class _StoredVector:
    chunk_id: str
    doc_id: str
    content: str
    metadata: dict[str, Any]
    embedding: NDArray[np.float64]


class NumpyVectorStore(VectorStore):
    """A small cosine-distance vector store with optional local persistence."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        path: str | Path | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.path = Path(path) if path is not None else None
        self._records: dict[str, _StoredVector] = {}
        if self.path is not None and self.path.exists():
            self._records = self._load(self.path)

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        new_records = self._build_records(chunks)
        records = dict(self._records)
        records.update(new_records)
        self._commit(records)

    def replace_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return

        new_records = self._build_records(chunks)
        doc_ids = {chunk.doc_id for chunk in chunks}
        records = {
            chunk_id: record
            for chunk_id, record in self._records.items()
            if record.doc_id not in doc_ids
        }
        records.update(new_records)
        self._commit(records)

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        deleted_doc_ids = set(doc_ids)
        if not deleted_doc_ids:
            return
        records = {
            chunk_id: record
            for chunk_id, record in self._records.items()
            if record.doc_id not in deleted_doc_ids
        }
        if len(records) != len(self._records):
            self._commit(records)

    def query(self, query_text: str, n_results: int = 10) -> list[VectorSearchResult]:
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")
        if not self._records:
            return []

        query_embeddings = self.embedding_provider.embed_texts([query_text])
        validate_embeddings(query_embeddings, 1)
        query_vector = np.asarray(query_embeddings[0], dtype=np.float64)

        records = list(self._records.values())
        matrix = np.vstack([record.embedding for record in records])
        if query_vector.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"query embedding dimension {query_vector.shape[0]} does not match index "
                f"dimension {matrix.shape[1]}"
            )

        query_norm = np.linalg.norm(query_vector)
        record_norms = np.linalg.norm(matrix, axis=1)
        if query_norm == 0 or np.any(record_norms == 0):
            raise ValueError("cosine distance requires non-zero embedding vectors")

        similarities = (matrix @ query_vector) / (record_norms * query_norm)
        distances = 1.0 - np.clip(similarities, -1.0, 1.0)
        ids = np.asarray([record.chunk_id for record in records])
        ordered_indices = np.lexsort((ids, distances))[:n_results]

        return [
            VectorSearchResult(
                chunk_id=records[index].chunk_id,
                content=records[index].content,
                metadata=dict(records[index].metadata),
                distance=float(distances[index]),
            )
            for index in ordered_indices
        ]

    def _build_records(self, chunks: Sequence[DocumentChunk]) -> dict[str, _StoredVector]:
        embeddings = self.embedding_provider.embed_texts([chunk.content for chunk in chunks])
        validate_embeddings(embeddings, len(chunks))
        self._validate_index_dimensions(embeddings[0])

        records: dict[str, _StoredVector] = {}
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vector = np.asarray(embedding, dtype=np.float64)
            if np.linalg.norm(vector) == 0:
                raise ValueError("cosine distance requires non-zero embedding vectors")
            records[chunk.chunk_id] = _StoredVector(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                metadata=dict(serialize_chunk_metadata(chunk)),
                embedding=vector,
            )
        return records

    def _validate_index_dimensions(self, embedding: Sequence[float]) -> None:
        if not self._records:
            return
        current_dimension = next(iter(self._records.values())).embedding.shape[0]
        if len(embedding) != current_dimension:
            raise ValueError(
                f"embedding dimension {len(embedding)} does not match index dimension "
                f"{current_dimension}"
            )

    def _commit(self, records: dict[str, _StoredVector]) -> None:
        if self.path is not None:
            self._persist(self.path, records)
        self._records = records

    @staticmethod
    def _persist(path: Path, records: Mapping[str, _StoredVector]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered_records = sorted(records.values(), key=lambda record: record.chunk_id)
        embeddings = (
            np.vstack([record.embedding for record in ordered_records])
            if ordered_records
            else np.empty((0, 0), dtype=np.float64)
        )

        temporary_path: str | None = None
        try:
            with NamedTemporaryFile(
                mode="wb", suffix=".npz", dir=path.parent, delete=False
            ) as temporary_file:
                temporary_path = temporary_file.name
                np.savez_compressed(
                    temporary_file,
                    chunk_ids=np.asarray([record.chunk_id for record in ordered_records]),
                    doc_ids=np.asarray([record.doc_id for record in ordered_records]),
                    contents=np.asarray([record.content for record in ordered_records]),
                    metadatas=np.asarray(
                        [json.dumps(record.metadata, sort_keys=True) for record in ordered_records]
                    ),
                    embeddings=embeddings,
                )
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _load(path: Path) -> dict[str, _StoredVector]:
        try:
            with np.load(path, allow_pickle=False) as data:
                chunk_ids = data["chunk_ids"]
                doc_ids = data["doc_ids"]
                contents = data["contents"]
                metadatas = data["metadatas"]
                embeddings = np.asarray(data["embeddings"], dtype=np.float64)
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(f"invalid NumPy vector store at {path}") from error

        item_count = len(chunk_ids)
        if not (len(doc_ids) == len(contents) == len(metadatas) == len(embeddings) == item_count):
            raise ValueError(f"invalid NumPy vector store at {path}: inconsistent record counts")
        if embeddings.ndim != 2:
            raise ValueError(f"invalid NumPy vector store at {path}: embeddings must be a matrix")

        records: dict[str, _StoredVector] = {}
        try:
            for index in range(item_count):
                metadata = json.loads(str(metadatas[index]))
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be a JSON object")
                chunk_id = str(chunk_ids[index])
                records[chunk_id] = _StoredVector(
                    chunk_id=chunk_id,
                    doc_id=str(doc_ids[index]),
                    content=str(contents[index]),
                    metadata=metadata,
                    embedding=embeddings[index],
                )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid NumPy vector store at {path}") from error
        return records
