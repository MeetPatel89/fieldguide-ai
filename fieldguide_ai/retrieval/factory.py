"""Compose vectorstore-ai infrastructure at the application boundary."""

import os
from pathlib import Path

from vectorstore import (
    EmbeddingProvider,
    FaissVectorStore,
    NumpyVectorStore,
    OpenAIEmbedding,
    PostgresDocumentCatalog,
    RetrievalCatalog,
    Retriever,
    VectorStore,
    create_store,
)
from vectorstore import build_retriever as compose_retriever

from fieldguide_ai.errors import (
    ConfigurationError,
    InvalidVectorStoreError,
    VectorStoreOperationError,
)
from fieldguide_ai.retrieval.modes import RetrievalMode, retriever_config_for
from fieldguide_ai.retrieval.settings import RetrievalSettings


def build_embedder(
    settings: RetrievalSettings, *, api_key: str | None = None
) -> OpenAIEmbedding:
    """Create the configured embedder using an explicit key or the environment."""
    if settings.embedding_model is None:
        raise ConfigurationError("an embedding model is required to build an embedder")
    resolved_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
    if not resolved_key or not resolved_key.strip():
        raise ConfigurationError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAIEmbedding(model=settings.embedding_model, api_key=resolved_key)


def build_store(settings: RetrievalSettings, dimension: int) -> VectorStore:
    """Create a store, loading a NumPy or FAISS directory when it exists.

    Existing invalid indexes raise an error rather than being replaced with
    an empty store. ``dimension`` must come from the embedder's spec.
    """
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ConfigurationError("embedding dimension must be a positive integer")
    if settings.store_type is None:
        raise ConfigurationError("a store type is required to build a vector store")
    if settings.store_type == "chroma":
        if settings.store_path is None:
            raise ConfigurationError("Chroma requires a store path")
        return create_store(
            "chroma",
            path=settings.store_path,
            collection_name=settings.collection_name,
        )

    store = create_store(settings.store_type, dimension=dimension)
    if settings.store_path is not None and Path(settings.store_path).exists():
        if not isinstance(store, (NumpyVectorStore, FaissVectorStore)):
            raise ConfigurationError("the configured store does not support loading")
        try:
            store = store.load(settings.store_path)
        except (OSError, ValueError) as error:
            raise InvalidVectorStoreError(
                f"cannot load {settings.store_type} store at {settings.store_path}"
            ) from error
        if store.dimension is not None and store.dimension != dimension:
            raise ConfigurationError(
                f"stored vector dimension {store.dimension} does not match "
                f"embedding dimension {dimension}"
            )
    return store


def persist_store(store: VectorStore, settings: RetrievalSettings) -> None:
    """Save NumPy/FAISS stores; Chroma persists writes automatically.

    A missing path deliberately leaves an in-memory store unpersisted.
    """
    if settings.store_type == "chroma" or settings.store_path is None:
        return
    if not (
        (settings.store_type == "numpy" and isinstance(store, NumpyVectorStore))
        or (settings.store_type == "faiss" and isinstance(store, FaissVectorStore))
    ):
        raise ConfigurationError("store does not match the configured persistence type")
    try:
        store.save(settings.store_path)
    except (OSError, ValueError, RuntimeError) as error:
        raise VectorStoreOperationError(
            f"cannot persist {settings.store_type} store at {settings.store_path}"
        ) from error


def build_catalog(
    settings: RetrievalSettings, *, initialize_schema: bool = False
) -> PostgresDocumentCatalog:
    """Build a Postgres catalog, creating its schema only when requested."""
    if settings.catalog_dsn is None:
        raise ConfigurationError(
            "a catalog DSN is required to build a Postgres catalog, "
            "including for dense retrieval"
        )
    return PostgresDocumentCatalog(
        settings.catalog_dsn, initialize_schema=initialize_schema
    )


def build_retriever(
    settings: RetrievalSettings,
    top_k: int,
    *,
    catalog: RetrievalCatalog | None = None,
    embedder: EmbeddingProvider | None = None,
) -> Retriever:
    """Compose the selected signals, optionally reusing caller-owned dependencies.

    Lexical mode never constructs or uses an embedder or vector store. Pass
    the same catalog to source conversion to hydrate document provenance.
    Schema initialization remains an explicit call to ``build_catalog``.
    """
    config = retriever_config_for(settings.mode, top_k)
    resolved_catalog = catalog if catalog is not None else build_catalog(settings)
    if settings.mode is RetrievalMode.LEXICAL:
        return compose_retriever(resolved_catalog, config=config)
    resolved_embedder = embedder if embedder is not None else build_embedder(settings)
    store = build_store(settings, resolved_embedder.spec.dimension)
    return compose_retriever(
        resolved_catalog,
        primary=resolved_embedder,
        primary_store=store,
        config=config,
    )
