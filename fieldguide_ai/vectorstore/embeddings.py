"""Embedding providers for vector-store implementations."""

from collections.abc import Sequence

from openai import OpenAI

from fieldguide_ai.errors import ConfigurationError, EmbeddingError
from fieldguide_ai.vectorstore.base import EmbeddingProvider

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings for text batches using the OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str | None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        client: OpenAI | None = None,
    ) -> None:
        if not model.strip():
            raise ConfigurationError("embedding model must not be blank")
        if client is None and not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        self._model = model
        self._client = client if client is not None else OpenAI(api_key=api_key)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts in input order."""
        if not texts:
            return []

        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=list(texts),
                encoding_format="float",
            )
            return [list(item.embedding) for item in response.data]
        except Exception as error:
            raise EmbeddingError(
                f"OpenAI embedding failed for model {self._model!r}"
            ) from error
