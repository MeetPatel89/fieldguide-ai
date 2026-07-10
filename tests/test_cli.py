import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fieldguide_ai.cli import index_corpus, parse_args, preview_chunks, print_history, run_chat_loop
from fieldguide_ai.messages import ChatMessage
from fieldguide_ai.providers.base import LLMProvider


class FakeProvider(LLMProvider):
    def generate(self, messages: list[ChatMessage]) -> str:
        user_messages = [message.content for message in messages if message.role == "user"]
        return f"reply to {user_messages[-1]}"


class FakeVectorStore:
    def __init__(self) -> None:
        self.replacements = []

    def replace_chunks(self, chunks) -> None:
        self.replacements.append(list(chunks))

    def delete_documents(self, doc_ids) -> None:
        pass


class CliTest(unittest.TestCase):
    def test_chat_loop_maintains_history_across_turns(self) -> None:
        provider = FakeProvider()
        input_stream = io.StringIO("First question\nFollow up\n:quit\n")
        output_stream = io.StringIO()

        run_chat_loop(provider, input_stream=input_stream, output_stream=output_stream)

        self.assertEqual(
            [message.role for message in provider.get_history()],
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertIn("Assistant> reply to First question", output_stream.getvalue())
        self.assertIn("Assistant> reply to Follow up", output_stream.getvalue())

    def test_chat_loop_can_clear_history_but_keeps_system_prompt(self) -> None:
        provider = FakeProvider()
        input_stream = io.StringIO("Question\n:clear\n:quit\n")
        output_stream = io.StringIO()

        run_chat_loop(provider, input_stream=input_stream, output_stream=output_stream)

        self.assertEqual(provider.get_history(), [provider.get_history()[0]])
        self.assertEqual(provider.get_history()[0].role, "system")
        self.assertIn("History cleared.", output_stream.getvalue())

    def test_print_history_writes_numbered_messages(self) -> None:
        provider = FakeProvider(
            message_history=[
                ChatMessage(role="system", content="Rules"),
                ChatMessage(role="user", content="Hello"),
            ]
        )
        output_stream = io.StringIO()

        print_history(provider, output_stream=output_stream)

        self.assertEqual(output_stream.getvalue(), "1. system: Rules\n2. user: Hello\n")

    def test_preview_chunks_prints_corpus_summary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "doc.md"
            path.write_text(
                "\n".join(
                    [
                        "---",
                        "doc_id: DOC-1",
                        "title: Test Doc",
                        "doc_type: runbook",
                        "---",
                        "",
                        "# Test Doc",
                        "",
                        "## Summary",
                        "",
                        "This is a short section.",
                    ]
                ),
                encoding="utf-8",
            )
            output_stream = io.StringIO()

            preview_chunks(tmpdir, max_words=900, limit=5, output_stream=output_stream)

        output = output_stream.getvalue()
        self.assertIn("Loaded 1 documents and created 1 chunks.", output)
        self.assertIn("DOC-1::chunk-0000 [runbook]", output)
        self.assertIn("Test Doc > Summary", output)

    def test_index_corpus_reports_indexed_counts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doc.md").write_text("# Test\n\nBody", encoding="utf-8")
            output_stream = io.StringIO()
            store = FakeVectorStore()

            result = index_corpus(
                tmpdir,
                vector_store=store,
                max_words=900,
                output_stream=output_stream,
            )

        self.assertEqual(result.document_count, 1)
        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(output_stream.getvalue(), "Indexed 1 documents and 1 chunks.\n")

    def test_parses_numpy_indexing_configuration(self) -> None:
        args = parse_args(
            [
                "--index-corpus",
                "docs",
                "--vector-store",
                "numpy",
                "--store-path",
                "custom.npz",
            ]
        )

        self.assertEqual(args.index_corpus, "docs")
        self.assertEqual(args.vector_store, "numpy")
        self.assertEqual(args.store_path, "custom.npz")
