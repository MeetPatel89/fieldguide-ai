"""Vector-store providers and shared interfaces."""

from fieldguide_ai.vectorstore.base import (
    EmbeddingProvider,
    VectorSearchResult,
    VectorStore,
)
from fieldguide_ai.vectorstore.chroma_store import (
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)
from fieldguide_ai.vectorstore.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    OpenAIEmbeddingProvider,
)
from fieldguide_ai.vectorstore.faiss_store import DEFAULT_FAISS_PATH, FaissVectorStore
from fieldguide_ai.vectorstore.metadata import serialize_chunk_metadata
from fieldguide_ai.vectorstore.numpy_store import NumpyVectorStore

__all__ = [
    "ChromaVectorStore",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_FAISS_PATH",
    "EmbeddingProvider",
    "FaissVectorStore",
    "OpenAIEmbeddingProvider",
    "NumpyVectorStore",
    "VectorSearchResult",
    "VectorStore",
    "serialize_chunk_metadata",
]
