"""Application composition for supported vector-store implementations."""

from fieldguide_ai.errors import ConfigurationError
from fieldguide_ai.vectorstore.base import EmbeddingProvider, VectorStore
from fieldguide_ai.vectorstore.chroma_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)
from fieldguide_ai.vectorstore.faiss_store import DEFAULT_FAISS_PATH, FaissVectorStore
from fieldguide_ai.vectorstore.numpy_store import DEFAULT_NUMPY_PATH, NumpyVectorStore


def build_vector_store(
    provider_name: str,
    embedding_provider: EmbeddingProvider,
    path: str | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> VectorStore:
    """Build a configured vector store at the application boundary."""
    if provider_name == "chroma":
        return ChromaVectorStore(
            path=path or DEFAULT_CHROMA_PATH,
            collection_name=collection_name,
            embedding_provider=embedding_provider,
        )
    if provider_name == "numpy":
        return NumpyVectorStore(
            path=path or DEFAULT_NUMPY_PATH,
            embedding_provider=embedding_provider,
        )
    if provider_name == "faiss":
        return FaissVectorStore(
            path=path or DEFAULT_FAISS_PATH,
            embedding_provider=embedding_provider,
        )
    raise ConfigurationError(f"unsupported vector store provider: {provider_name}")
