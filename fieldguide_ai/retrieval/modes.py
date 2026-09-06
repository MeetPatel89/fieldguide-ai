"""Application retrieval modes and their shared-library configuration."""

from enum import StrEnum

from vectorstore import RetrieverConfig

from fieldguide_ai.errors import ConfigurationError


class RetrievalMode(StrEnum):
    """Signals used to retrieve context for a question."""

    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


def retriever_config_for(mode: RetrievalMode, top_k: int) -> RetrieverConfig:
    """Enable the requested signals and limit the final result count."""
    try:
        mode = RetrievalMode(mode)
    except ValueError as error:
        raise ConfigurationError(f"unsupported retrieval mode: {mode}") from error
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ConfigurationError("top_k must be a positive integer")
    return RetrieverConfig(
        dense_enabled=mode is not RetrievalMode.LEXICAL,
        lexical_enabled=mode is not RetrievalMode.DENSE,
        final_top_k=top_k,
    )
