"""Immutable configuration for composing catalog-backed retrieval."""

from dataclasses import dataclass, field
from pathlib import Path

from fieldguide_ai.errors import ConfigurationError
from fieldguide_ai.retrieval.modes import RetrievalMode

DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class RetrievalSettings:
    """Validated retrieval inputs, with credentials supplied by the caller.

    Lexical retrieval needs only a catalog DSN. Dense signals additionally
    require a store and embedding model. NumPy and FAISS can be ephemeral
    when ``store_path`` is None; otherwise their path is a directory.
    Chroma always requires a persistence path.

    Dense settings may omit the DSN when a catalog is injected into the
    retriever factory. The default Postgres catalog factory requires a DSN
    in every mode because dense hits also need catalog chunk hydration.
    """

    mode: RetrievalMode = RetrievalMode.DENSE
    store_type: str | None = "chroma"
    store_path: str | Path | None = DEFAULT_CHROMA_PATH
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_model: str | None = DEFAULT_EMBEDDING_MODEL
    catalog_dsn: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Reject incomplete configuration before creating infrastructure."""
        try:
            object.__setattr__(self, "mode", RetrievalMode(self.mode))
        except ValueError as error:
            raise ConfigurationError(
                f"unsupported retrieval mode: {self.mode}"
            ) from error
        if self.store_type not in {None, "numpy", "faiss", "chroma"}:
            raise ConfigurationError(f"unsupported vector store: {self.store_type}")
        if self.store_path is not None and not str(self.store_path).strip():
            raise ConfigurationError("store path must not be blank")
        if not self.collection_name.strip():
            raise ConfigurationError("collection name must not be blank")
        if self.embedding_model is not None and not self.embedding_model.strip():
            raise ConfigurationError("embedding model must not be blank")
        if self.catalog_dsn is not None and not self.catalog_dsn.strip():
            raise ConfigurationError("catalog DSN must not be blank")
        if self.mode is not RetrievalMode.DENSE and self.catalog_dsn is None:
            raise ConfigurationError(
                "lexical and hybrid retrieval require a catalog DSN"
            )
        if self.mode is not RetrievalMode.LEXICAL:
            if self.store_type is None:
                raise ConfigurationError("dense and hybrid retrieval require a store")
            if self.embedding_model is None:
                raise ConfigurationError(
                    "dense and hybrid retrieval require an embedding model"
                )
            if self.store_type == "chroma" and self.store_path is None:
                raise ConfigurationError("Chroma requires a store path")
