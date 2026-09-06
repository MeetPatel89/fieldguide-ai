"""Application source records and diagnostics for shared retrieval results."""

from dataclasses import dataclass
from typing import Self

from vectorstore import (
    CatalogDocument,
    RetrievalCatalog,
    RetrievalHit,
    RetrievalResult,
    RetrievalTimings,
)

from fieldguide_ai.retrieval.modes import RetrievalMode


@dataclass(frozen=True)
class RetrievedSource:
    """A retrieved chunk with document provenance and per-signal ranks."""

    chunk_id: str
    doc_id: str
    source: str | None
    title: str | None
    section_path: str | None
    text: str
    score: float
    dense_rank: int | None
    lexical_rank: int | None

    @classmethod
    def from_hit(
        cls, hit: RetrievalHit, document: CatalogDocument | None = None
    ) -> Self:
        """Keep hit content and ranks, adding document metadata when available."""
        chunk = hit.chunk
        if document is not None and document.doc_id != chunk.doc_id:
            raise ValueError("source document does not match the retrieved chunk")
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            source=document.source if document is not None else None,
            title=document.title if document is not None else None,
            section_path=chunk.section_path,
            text=chunk.text,
            score=hit.score,
            dense_rank=hit.dense_rank,
            lexical_rank=hit.lexical_rank,
        )


def sources_from_result(
    result: RetrievalResult, catalog: RetrievalCatalog
) -> tuple[RetrievedSource, ...]:
    """Hydrate document provenance in one lookup, preserving hit order.

    Missing documents leave source/title unset without discarding retrieved
    text. Empty results avoid a catalog query entirely.
    """
    if not result.hits:
        return ()
    doc_ids = list(dict.fromkeys(hit.chunk.doc_id for hit in result.hits))
    documents = {
        document.doc_id: document
        for document in catalog.find({"doc_id": {"$in": doc_ids}}, limit=len(doc_ids))
    }
    return tuple(
        RetrievedSource.from_hit(hit, documents.get(hit.chunk.doc_id))
        for hit in result.hits
    )


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Requested mode and the library's observed retrieval health and timing."""

    mode: RetrievalMode
    degraded: bool
    errors: tuple[str, ...]
    provider: str | None
    timings: RetrievalTimings

    @classmethod
    def from_result(cls, result: RetrievalResult, mode: RetrievalMode) -> Self:
        """Preserve diagnostics even when a configured signal yields no hits."""
        return cls(
            mode=RetrievalMode(mode),
            degraded=result.degraded,
            errors=tuple(result.errors),
            provider=result.provider,
            timings=result.timings,
        )
