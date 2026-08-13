"""Tests for retrieval-grounded chat orchestration."""

import unittest
from collections.abc import Sequence

from model_runtime import Message

from fieldguide_ai.knowledge_bot import KnowledgeBot
from fieldguide_ai.vectorstore import VectorSearchResult
from tests.session_fakes import FakeChatModel, make_session


class FakeStore:
    """Recording vector search fake."""

    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.queries: list[tuple[str, int]] = []

    def query(self, query_text: str, n_results: int = 10) -> list[VectorSearchResult]:
        """Record the query and return configured results."""
        self.queries.append((query_text, n_results))
        return self.results

    def index_chunks(self, chunks: Sequence[object]) -> None:
        """Accept unused index operations required by the store protocol."""

    def replace_chunks(self, chunks: Sequence[object]) -> None:
        """Accept unused replacement operations required by the store protocol."""

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        """Accept unused deletion operations required by the store protocol."""


class KnowledgeBotTest(unittest.TestCase):
    """Verify retrieval context and visible history remain distinct."""

    def test_augments_generation_but_keeps_raw_question_in_history(self) -> None:
        source = VectorSearchResult(
            chunk_id="DOC::0",
            content="The escalation threshold is severity two.",
            metadata={
                "source_path": "docs/runbook.md",
                "section_path": "Runbook > Escalation",
            },
            distance=0.1,
        )
        adapter = FakeChatModel(response_text="Grounded answer")
        provider = make_session(adapter)
        store = FakeStore([source])

        response = KnowledgeBot(provider, store).ask("When should I escalate?", top_k=3)

        self.assertEqual(response.answer, "Grounded answer")
        self.assertEqual(response.sources, (source,))
        self.assertEqual(store.queries, [("When should I escalate?", 3)])
        generated_message = adapter.requests[-1][1].messages[-1]
        self.assertIn("docs/runbook.md", generated_message.text)
        self.assertIn("severity two", generated_message.text)
        self.assertEqual(
            provider.history,
            (
                Message.user("When should I escalate?"),
                Message.assistant("Grounded answer"),
            ),
        )

    def test_plain_chat_fallback_returns_no_sources(self) -> None:
        adapter = FakeChatModel(response_text="Grounded answer")
        provider = make_session(adapter)

        answer, sources = KnowledgeBot(provider).ask("Hello")

        self.assertEqual(answer, "Grounded answer")
        self.assertEqual(sources, ())
        self.assertEqual(
            adapter.requests[0][1].messages,
            (Message.user("Hello"),),
        )

    def test_rejects_non_positive_top_k_even_without_a_store(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be greater than zero"):
            KnowledgeBot(make_session()).ask("Hello", top_k=0)


if __name__ == "__main__":
    unittest.main()
