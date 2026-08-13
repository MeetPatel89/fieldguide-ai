"""Retrieval-grounded chat orchestration."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from model_runtime import ChatSession, Message

from fieldguide_ai.errors import ConfigurationError
from fieldguide_ai.vectorstore import VectorSearcher, VectorSearchResult


@dataclass(frozen=True)
class KnowledgeAnswer:
    """An assistant answer and the source chunks used to produce it."""

    answer: str
    sources: tuple[VectorSearchResult, ...]

    def __iter__(self) -> Iterator[str | tuple[VectorSearchResult, ...]]:
        """Allow callers to unpack the answer and sources as a pair."""
        yield self.answer
        yield self.sources


class KnowledgeBot:
    """Wrap an LLM provider with optional vector retrieval."""

    def __init__(
        self,
        provider: ChatSession,
        vector_store: VectorSearcher | None = None,
    ) -> None:
        self._provider = provider
        self._vector_store = vector_store

    def ask(self, question: str, top_k: int = 5) -> KnowledgeAnswer:
        """Answer a question using the nearest indexed chunks when configured."""
        if top_k <= 0:
            raise ConfigurationError("top_k must be greater than zero")
        if self._vector_store is None:
            return KnowledgeAnswer(self._provider.chat_sync(question), ())

        sources = self._vector_store.query(question, n_results=top_k)
        if not sources:
            return KnowledgeAnswer(self._provider.chat_sync(question), ())

        augmented_question = _build_augmented_question(question, sources)
        result = self._provider.complete_turn_sync(
            question,
            generation_message=Message.user(augmented_question),
        )
        return KnowledgeAnswer(result.text, tuple(sources))


def _build_augmented_question(
    question: str,
    sources: Sequence[VectorSearchResult],
) -> str:
    context_parts: list[str] = []
    for index, source in enumerate(sources, start=1):
        source_path = source.metadata.get("source_path", "unknown")
        section = source.metadata.get("section_path") or source.metadata.get(
            "section_title", "unknown"
        )
        context_parts.append(
            f"[Source {index}]\n"
            f"Path: {source_path}\n"
            f"Section: {section}\n"
            f"Content:\n{source.content}"
        )
    context = "\n\n".join(context_parts)
    return (
        "Answer the user's question using the retrieved context below. "
        "If the context is insufficient, say so rather than inventing facts.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"User question:\n{question}"
    )
