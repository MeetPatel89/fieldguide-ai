"""Retrieval-grounded chat orchestration."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from fieldguide_ai.chat import ChatMessage, GenerationResult
from fieldguide_ai.errors import ConfigurationError
from fieldguide_ai.vectorstore import VectorSearcher, VectorSearchResult


class ChatSession(Protocol):
    """Conversation behavior required by retrieval orchestration."""

    def chat(self, message: str | ChatMessage) -> str:
        """Send and record one conversation turn."""
        ...

    def complete_turn(
        self,
        message: str | ChatMessage,
        *,
        generation_message: ChatMessage | None = None,
    ) -> GenerationResult:
        """Generate and atomically record a conversation turn."""
        ...


@dataclass(frozen=True, init=False)
class KnowledgeAnswer:
    """An assistant answer and the source chunks used to produce it."""

    answer: str
    _sources: tuple[VectorSearchResult, ...] = field(repr=False)

    def __init__(self, answer: str, sources: Sequence[VectorSearchResult]) -> None:
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "_sources", tuple(sources))

    @property
    def sources(self) -> list[VectorSearchResult]:
        """A copy of the retrieval sources in ranking order."""
        return list(self._sources)

    def __iter__(self) -> Iterator[str | list[VectorSearchResult]]:
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
            return KnowledgeAnswer(self._provider.chat(question), ())

        sources = self._vector_store.query(question, n_results=top_k)
        if not sources:
            return KnowledgeAnswer(self._provider.chat(question), ())

        augmented_question = _build_augmented_question(question, sources)
        result = self._provider.complete_turn(
            question,
            generation_message=ChatMessage(
                role="user",
                content=augmented_question,
            ),
        )
        return KnowledgeAnswer(result.text, sources)


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
