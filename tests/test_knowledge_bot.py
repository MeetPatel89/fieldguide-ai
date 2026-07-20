import unittest
from collections.abc import Sequence

from fieldguide_ai.generation import GenerationResult
from fieldguide_ai.knowledge_bot import KnowledgeBot
from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers.base import LLMProvider
from fieldguide_ai.vectorstore import VectorSearchResult


class FakeProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.generated_messages: list[ChatMessage] = []

    def generate(self, messages: list[ChatMessage]) -> GenerationResult:
        self.generated_messages = list(messages)
        return self._record_generation(
            GenerationResult(text="Grounded answer", provider="fake", model="fake")
        )


class FakeStore:
    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.queries: list[tuple[str, int]] = []

    def query(self, query_text: str, n_results: int = 10) -> list[VectorSearchResult]:
        self.queries.append((query_text, n_results))
        return self.results

    def index_chunks(self, chunks: Sequence[object]) -> None:
        pass

    def replace_chunks(self, chunks: Sequence[object]) -> None:
        pass

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        pass


class KnowledgeBotTest(unittest.TestCase):
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
        provider = FakeProvider()
        store = FakeStore([source])

        response = KnowledgeBot(provider, store).ask("When should I escalate?", top_k=3)

        self.assertEqual(response.answer, "Grounded answer")
        self.assertEqual(response.sources, [source])
        self.assertEqual(store.queries, [("When should I escalate?", 3)])
        self.assertIn("docs/runbook.md", provider.generated_messages[-1].content)
        self.assertIn("severity two", provider.generated_messages[-1].content)
        self.assertEqual(
            provider.get_history(),
            [
                ChatMessage(role="user", content="When should I escalate?"),
                ChatMessage(role="assistant", content="Grounded answer"),
            ],
        )

    def test_plain_chat_fallback_returns_no_sources(self) -> None:
        provider = FakeProvider()

        answer, sources = KnowledgeBot(provider).ask("Hello")

        self.assertEqual(answer, "Grounded answer")
        self.assertEqual(sources, [])
        self.assertEqual(
            provider.generated_messages, [ChatMessage(role="user", content="Hello")]
        )

    def test_rejects_non_positive_top_k_even_without_a_store(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be greater than zero"):
            KnowledgeBot(FakeProvider()).ask("Hello", top_k=0)
