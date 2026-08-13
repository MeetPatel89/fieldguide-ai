"""Tests for the flag-based Fieldguide CLI."""

import io
import unittest
from collections.abc import Sequence
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from model_runtime import (
    ChatSession,
    Message,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ProviderUnavailableError,
)

from fieldguide_ai.cli import (
    DEMO_QUESTION,
    index_corpus,
    main,
    parse_args,
    preview_chunks,
    print_history,
    run_chat_loop,
    run_demo,
)
from fieldguide_ai.config import DEFAULT_SYSTEM_PROMPT
from fieldguide_ai.ingestion.models import DocumentChunk
from tests.session_fakes import FakeChatModel, make_session


def reply_to_last_user(model_id: str, request: ModelRequest) -> ModelResponse:
    """Build a deterministic response from the last normalized user message."""
    user_messages = [
        message.text for message in request.messages if message.role is MessageRole.USER
    ]
    return ModelResponse(
        message=Message.assistant(f"reply to {user_messages[-1]}"),
        model=model_id,
    )


def fake_provider() -> tuple[FakeChatModel, ChatSession]:
    """Build a session whose adapter replies to the current user text."""
    adapter = FakeChatModel(response_factory=reply_to_last_user)
    return adapter, make_session(adapter)


class FakeVectorStore:
    """Recording vector replacement fake."""

    def __init__(self) -> None:
        self.replacements: list[list[DocumentChunk]] = []

    def replace_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        """Record replacement chunks."""
        self.replacements.append(list(chunks))

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        """Accept unused deletion operations."""


class CliTest(unittest.TestCase):
    """Verify stateful CLI flow and deterministic indexing commands."""

    def test_chat_loop_maintains_history_across_turns(self) -> None:
        _, provider = fake_provider()
        input_stream = io.StringIO("First question\nFollow up\n:quit\n")
        output_stream = io.StringIO()

        run_chat_loop(provider, input_stream=input_stream, output_stream=output_stream)

        self.assertEqual(provider.system_prompt, DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(
            [message.role for message in provider.history],
            [
                MessageRole.USER,
                MessageRole.ASSISTANT,
                MessageRole.USER,
                MessageRole.ASSISTANT,
            ],
        )
        self.assertIn("Assistant> reply to First question", output_stream.getvalue())
        self.assertIn("Assistant> reply to Follow up", output_stream.getvalue())

    def test_chat_loop_can_clear_history_but_keeps_system_prompt(self) -> None:
        _, provider = fake_provider()

        run_chat_loop(
            provider,
            input_stream=io.StringIO("Question\n:clear\n:quit\n"),
            output_stream=io.StringIO(),
        )

        self.assertEqual(provider.history, ())
        self.assertEqual(provider.system_prompt, DEFAULT_SYSTEM_PROMPT)

    def test_chat_loop_reuses_custom_system_prompt_after_clear(self) -> None:
        _, provider = fake_provider()

        run_chat_loop(
            provider,
            input_stream=io.StringIO("Question\n:clear\n:quit\n"),
            output_stream=io.StringIO(),
            system_prompt="Answer only from the field guide.",
        )

        self.assertEqual(provider.history, ())
        self.assertEqual(provider.system_prompt, "Answer only from the field guide.")

    def test_print_history_writes_system_then_turns(self) -> None:
        provider = make_session(
            history=(Message.user("Hello"),),
            system_prompt="Rules",
        )
        output_stream = io.StringIO()

        print_history(provider, output_stream=output_stream)

        self.assertEqual(output_stream.getvalue(), "1. system: Rules\n2. user: Hello\n")

    def test_demo_uses_runtime_session_record(self) -> None:
        adapter, provider = fake_provider()
        output_stream = io.StringIO()

        run_demo(provider, output_stream)

        self.assertEqual(provider.system_prompt, DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(provider.history[0], Message.user(DEMO_QUESTION))
        self.assertEqual(
            adapter.requests[0][1].messages[0], Message.system(DEFAULT_SYSTEM_PROMPT)
        )
        self.assertIn(f"reply to {DEMO_QUESTION}", output_stream.getvalue())

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
        self.assertEqual(
            output_stream.getvalue(), "Indexed 1 documents and 1 chunks.\n"
        )

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

    def test_parses_faiss_and_retrieval_configuration(self) -> None:
        args = parse_args(
            [
                "--index-corpus",
                "docs",
                "--vector-store",
                "faiss",
                "--store-path",
                "custom-index",
                "--top-k",
                "7",
            ]
        )

        self.assertEqual(args.vector_store, "faiss")
        self.assertEqual(args.store_path, "custom-index")
        self.assertEqual(args.top_k, 7)

    def test_rejects_non_positive_retrieval_count(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--top-k", "0"])

    def test_main_renders_model_runtime_errors_without_a_traceback(self) -> None:
        error = ProviderUnavailableError(
            "provider offline",
            retryable=False,
            provider="openai",
        )
        stderr = io.StringIO()

        with (
            patch("fieldguide_ai.cli.build_provider", side_effect=error),
            redirect_stderr(stderr),
            self.assertRaisesRegex(SystemExit, "1"),
        ):
            main(["--vector-store", "none"])

        self.assertIn("Error: provider offline", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
