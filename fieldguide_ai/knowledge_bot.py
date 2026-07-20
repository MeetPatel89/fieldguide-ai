"""Retrieval-grounded chat orchestration."""

from collections.abc import Iterator
from dataclasses import dataclass

from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers import LLMProvider
from fieldguide_ai.vectorstore import VectorSearchResult, VectorStore


@dataclass(frozen=True)
class KnowledgeAnswer:
    """An assistant answer and the source chunks used to produce it."""

    answer: str
    sources: list[VectorSearchResult]

    def __iter__(self) -> Iterator[str | list[VectorSearchResult]]:
        """Allow callers to unpack the answer and sources as a pair."""
        yield self.answer
        yield self.sources


class KnowledgeBot:
    """Wrap an LLM provider with optional vector retrieval."""

    def __init__(
        self,
        provider: LLMProvider,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.provider = provider
        self.vector_store = vector_store

    def ask(self, question: str, top_k: int = 5) -> KnowledgeAnswer:
        """Answer a question using the nearest indexed chunks when configured."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.vector_store is None:
            return KnowledgeAnswer(self.provider.chat(question), [])

        sources = self.vector_store.query(question, n_results=top_k)
        if not sources:
            return KnowledgeAnswer(self.provider.chat(question), [])

        prior_history = self.provider.get_history()
        self.provider.add_message(ChatMessage(role="user", content=question))
        augmented_question = _build_augmented_question(question, sources)
        result = self.provider.generate(
            [
                *prior_history,
                ChatMessage(role="user", content=augmented_question),
            ]
        )
        self.provider.add_message(ChatMessage(role="assistant", content=result.text))
        return KnowledgeAnswer(result.text, sources)


def _build_augmented_question(question: str, sources: list[VectorSearchResult]) -> str:
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
