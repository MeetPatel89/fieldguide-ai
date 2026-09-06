"""Preview and ingest Markdown through the shared catalog pipeline."""

from pathlib import Path

from vectorstore import (
    CatalogChunk,
    EmbeddingRouter,
    IngestionError,
    IngestionPipeline,
    IngestionResult,
    MarkdownSectionChunker,
    MarkdownSourceAdapter,
    Record,
    SourceAdapterError,
    VectorStore,
)

from fieldguide_ai.errors import (
    ConfigurationError,
    DocumentLoadError,
    VectorStoreOperationError,
)
from fieldguide_ai.retrieval.factory import (
    build_catalog,
    build_embedder,
    build_store,
    persist_store,
)
from fieldguide_ai.retrieval.modes import RetrievalMode
from fieldguide_ai.retrieval.settings import RetrievalSettings


def _build_chunker(max_words: int) -> MarkdownSectionChunker:
    try:
        return MarkdownSectionChunker(max_words=max_words)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error


def chunk_corpus(
    path: str | Path, max_words: int
) -> tuple[list[Record], list[CatalogChunk]]:
    """Preview shared Markdown records and chunks without service clients."""
    chunker = _build_chunker(max_words)
    try:
        records = list(MarkdownSourceAdapter().iter_records(path))
    except SourceAdapterError as error:
        raise DocumentLoadError(str(error)) from error
    chunks = [chunk for record in records for chunk in chunker.chunk(record)]
    return records, chunks


def index_corpus(
    path: str | Path, settings: RetrievalSettings, max_words: int
) -> IngestionResult:
    """Ingest Markdown into an existing catalog and persist configured vectors.

    Lexical mode creates no embedding router or vector stores. Dense and
    hybrid modes pair the embedder's space with the configured store. Schema
    creation is an explicit, separate ``build_catalog`` operation.
    """
    chunker = _build_chunker(max_words)
    with build_catalog(settings) as catalog:
        store: VectorStore | None = None
        router: EmbeddingRouter | None = None
        stores: dict[str, VectorStore] = {}
        if settings.mode is not RetrievalMode.LEXICAL:
            embedder = build_embedder(settings)
            store = build_store(settings, embedder.spec.dimension)
            stores[embedder.spec.space_id] = store
            router = EmbeddingRouter(embedder)
        pipeline = IngestionPipeline(catalog, stores, router, chunker=chunker)
        try:
            result = pipeline.ingest_source(MarkdownSourceAdapter(), path)
        except SourceAdapterError as error:
            raise DocumentLoadError(str(error)) from error
        except IngestionError as error:
            raise VectorStoreOperationError(
                f"could not index corpus at {path}: {error}"
            ) from error
        if store is not None:
            persist_store(store, settings)
        return result
