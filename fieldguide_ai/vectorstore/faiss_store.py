"""FAISS-backed vector store with local persistence."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import faiss
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

DEFAULT_FAISS_PATH = "faiss_index"


@dataclass(frozen=True)
class _StoredChunk:
    numeric_id: int
    chunk_id: str
    content: str
    metadata: dict[str, Any]


class FaissVectorStore(VectorStore):
    """Persist normalized embeddings in a FAISS inner-product index."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        path: str | Path = DEFAULT_FAISS_PATH,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.path = Path(path)
        self.index_path = Path(f"{self.path}.faiss")
        self.metadata_path = Path(f"{self.path}.json")
        self._index: faiss.Index | None = None
        self._records: dict[int, _StoredChunk] = {}
        self._load_if_present()

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Insert or update chunks by chunk ID."""
        if not chunks:
            return
        embeddings = self._embed_chunks(chunks)
        self._upsert(chunks, embeddings, removed_doc_ids=set())

    def replace_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Replace every indexed chunk belonging to the supplied documents."""
        if not chunks:
            return
        embeddings = self._embed_chunks(chunks)
        self._upsert(
            chunks,
            embeddings,
            removed_doc_ids={chunk.doc_id for chunk in chunks},
        )

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        """Delete every chunk belonging to the supplied document IDs."""
        deleted_doc_ids = set(doc_ids)
        if not deleted_doc_ids:
            return
        records = {
            numeric_id: record
            for numeric_id, record in self._records.items()
            if record.metadata.get("doc_id") not in deleted_doc_ids
        }
        removed_ids = set(self._records) - set(records)
        if removed_ids:
            index = faiss.clone_index(self._index)
            index.remove_ids(np.asarray(sorted(removed_ids), dtype=np.int64))
            self._commit(index, records)

    def query(self, query_text: str, n_results: int = 10) -> list[VectorSearchResult]:
        """Return nearest chunks by cosine distance in nearest-first order."""
        if n_results <= 0:
            raise ValueError("n_results must be greater than zero")
        if not self._records or self._index is None:
            return []

        embeddings = self.embedding_provider.embed_texts([query_text])
        validate_embeddings(embeddings, 1)
        query_vector = self._normalized_matrix(embeddings)
        if query_vector.shape[1] != self._index.d:
            raise ValueError(
                f"query embedding dimension {query_vector.shape[1]} does not match "
                f"index dimension {self._index.d}"
            )

        # Search every record so tie ordering is stable across FAISS versions.
        similarities, numeric_ids = self._index.search(query_vector, len(self._records))
        results: list[VectorSearchResult] = []
        for similarity, numeric_id in zip(similarities[0], numeric_ids[0], strict=True):
            record = self._records.get(int(numeric_id))
            if record is None:
                continue
            results.append(
                VectorSearchResult(
                    chunk_id=record.chunk_id,
                    content=record.content,
                    metadata=dict(record.metadata),
                    distance=1.0 - float(np.clip(similarity, -1.0, 1.0)),
                )
            )
        results.sort(key=lambda result: (result.distance, result.chunk_id))
        return results[:n_results]

    def _embed_chunks(self, chunks: Sequence[DocumentChunk]) -> NDArray[np.float32]:
        embeddings = self.embedding_provider.embed_texts(
            [chunk.content for chunk in chunks]
        )
        validate_embeddings(embeddings, len(chunks))
        matrix = self._normalized_matrix(embeddings)
        if self._index is not None and matrix.shape[1] != self._index.d:
            raise ValueError(
                f"embedding dimension {matrix.shape[1]} does not match index dimension "
                f"{self._index.d}"
            )
        return matrix

    @staticmethod
    def _normalized_matrix(
        embeddings: Sequence[Sequence[float]],
    ) -> NDArray[np.float32]:
        matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0):
            raise ValueError("cosine distance requires non-zero embedding vectors")
        faiss.normalize_L2(matrix)
        return matrix

    def _upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: NDArray[np.float32],
        removed_doc_ids: set[str],
    ) -> None:
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique within an indexing batch")
        records = {
            numeric_id: record
            for numeric_id, record in self._records.items()
            if record.metadata.get("doc_id") not in removed_doc_ids
        }
        chunk_ids = {chunk.chunk_id for chunk in chunks}
        records = {
            numeric_id: record
            for numeric_id, record in records.items()
            if record.chunk_id not in chunk_ids
        }
        removed_ids = set(self._records) - set(records)

        if self._index is None:
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(embeddings.shape[1]))
        else:
            index = faiss.clone_index(self._index)
            if removed_ids:
                index.remove_ids(np.asarray(sorted(removed_ids), dtype=np.int64))

        existing_ids = {
            record.chunk_id: numeric_id for numeric_id, record in self._records.items()
        }
        next_id = max(self._records, default=-1) + 1
        numeric_ids: list[int] = []
        for chunk in chunks:
            numeric_id = existing_ids.get(chunk.chunk_id)
            if numeric_id is None:
                numeric_id = next_id
                next_id += 1
            numeric_ids.append(numeric_id)
            records[numeric_id] = _StoredChunk(
                numeric_id=numeric_id,
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=dict(serialize_chunk_metadata(chunk)),
            )

        index.add_with_ids(embeddings, np.asarray(numeric_ids, dtype=np.int64))
        self._commit(index, records)

    def _commit(self, index: faiss.Index, records: dict[int, _StoredChunk]) -> None:
        self._persist(index, records)
        self._index = index
        self._records = records

    def _persist(self, index: faiss.Index, records: Mapping[int, _StoredChunk]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_index: str | None = None
        temporary_metadata: str | None = None
        payload = {
            str(numeric_id): {
                "chunk_id": record.chunk_id,
                "content": record.content,
                "metadata": record.metadata,
            }
            for numeric_id, record in sorted(records.items())
        }
        try:
            with NamedTemporaryFile(
                suffix=".faiss", dir=self.path.parent, delete=False
            ) as temporary_file:
                temporary_index = temporary_file.name
            faiss.write_index(index, temporary_index)
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".json",
                dir=self.path.parent,
                delete=False,
            ) as temporary_file:
                temporary_metadata = temporary_file.name
                json.dump(payload, temporary_file, sort_keys=True)
            Path(temporary_index).replace(self.index_path)
            Path(temporary_metadata).replace(self.metadata_path)
        finally:
            for temporary_path in (temporary_index, temporary_metadata):
                if temporary_path is not None and Path(temporary_path).exists():
                    Path(temporary_path).unlink()

    def _load_if_present(self) -> None:
        index_exists = self.index_path.exists()
        metadata_exists = self.metadata_path.exists()
        if not index_exists and not metadata_exists:
            return
        if index_exists != metadata_exists:
            raise ValueError(f"incomplete FAISS vector store at {self.path}")

        try:
            index = faiss.read_index(str(self.index_path))
            if not isinstance(index, faiss.IndexIDMap2):
                raise ValueError("index must use IndexIDMap2")
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("metadata sidecar must be a JSON object")
            records: dict[int, _StoredChunk] = {}
            for numeric_id_text, item in payload.items():
                if not isinstance(item, dict) or not isinstance(
                    item.get("metadata"), dict
                ):
                    raise ValueError("invalid stored chunk")
                numeric_id = int(numeric_id_text)
                records[numeric_id] = _StoredChunk(
                    numeric_id=numeric_id,
                    chunk_id=str(item["chunk_id"]),
                    content=str(item["content"]),
                    metadata=dict(item["metadata"]),
                )
            if index.ntotal != len(records):
                raise ValueError("index and metadata record counts differ")
            stored_ids = {int(value) for value in faiss.vector_to_array(index.id_map)}
            if stored_ids != set(records):
                raise ValueError("index and metadata IDs differ")
            chunk_ids = {record.chunk_id for record in records.values()}
            if len(chunk_ids) != len(records):
                raise ValueError("chunk IDs must be unique")
        except (
            json.JSONDecodeError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(f"invalid FAISS vector store at {self.path}") from error
        self._index = index
        self._records = records
