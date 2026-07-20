"""Vector-store providers and shared interfaces."""

from fieldguide_ai.vectorstore.base import (
    DocumentIndex,
    EmbeddingProvider,
    VectorSearcher,
    VectorSearchResult,
    VectorStore,
)
from fieldguide_ai.vectorstore.chroma_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)
from fieldguide_ai.vectorstore.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    OpenAIEmbeddingProvider,
)
from fieldguide_ai.vectorstore.factory import build_vector_store
from fieldguide_ai.vectorstore.faiss_store import DEFAULT_FAISS_PATH, FaissVectorStore
from fieldguide_ai.vectorstore.metadata import serialize_chunk_metadata
from fieldguide_ai.vectorstore.numpy_store import DEFAULT_NUMPY_PATH, NumpyVectorStore

__all__ = [
    "ChromaVectorStore",
    "DEFAULT_CHROMA_PATH",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_FAISS_PATH",
    "DEFAULT_NUMPY_PATH",
    "DocumentIndex",
    "EmbeddingProvider",
    "FaissVectorStore",
    "OpenAIEmbeddingProvider",
    "NumpyVectorStore",
    "VectorSearchResult",
    "VectorSearcher",
    "VectorStore",
    "build_vector_store",
    "serialize_chunk_metadata",
]
