"""Tests for the flag-based Fieldguide CLI."""

import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
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
from vectorstore import IngestionResult

from fieldguide_ai.cli import (
    DEMO_QUESTION,
    main,
    parse_args,
    preview_chunks,
    print_history,
    run_chat_loop,
    run_demo,
)
from fieldguide_ai.config import DEFAULT_SYSTEM_PROMPT
from fieldguide_ai.retrieval import RetrievalSettings, chunk_corpus
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

    def test_preview_details_serializes_catalog_chunk_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doc.md").write_text("# Test\n\nBody", encoding="utf-8")
            output_stream = io.StringIO()
            preview_chunks(
                tmpdir,
                max_words=900,
                limit=1,
                output_stream=output_stream,
                details=True,
            )
            _, chunks = chunk_corpus(tmpdir, max_words=900)

        payload = json.loads(output_stream.getvalue().split("\n", 1)[1])
        self.assertEqual(payload["chunk_id"], chunks[0].chunk_id)
        self.assertEqual(payload["doc_id"], "doc")
        self.assertEqual(payload["text"], chunks[0].text)
        self.assertEqual(payload["section_path"], "Test")
        self.assertEqual(payload["content_hash"], chunks[0].content_hash)
        self.assertTrue(payload["active"])

    def test_main_indexes_with_shared_settings_and_reports_counts(self) -> None:
        output_stream = io.StringIO()
        dsn = "postgresql://test@localhost/test"
        with (
            patch("fieldguide_ai.cli.load_dotenv"),
            patch.dict(os.environ, {"POSTGRES_CONNECTIONSTRING": dsn}, clear=True),
            patch(
                "fieldguide_ai.cli.index_corpus", return_value=IngestionResult(2, 5)
            ) as index,
            patch("fieldguide_ai.cli.build_provider") as provider,
            patch("fieldguide_ai.cli.build_vector_store") as legacy_store,
            patch("fieldguide_ai.cli.OpenAIEmbeddingProvider") as legacy_embedder,
            redirect_stdout(output_stream),
        ):
            main(
                [
                    "--index-corpus",
                    "docs",
                    "--vector-store",
                    "numpy",
                    "--store-path",
                    "shared-index",
                    "--collection-name",
                    "guides",
                    "--embedding-model",
                    "text-embedding-3-large",
                    "--chunk-max-words",
                    "450",
                ]
            )

        index.assert_called_once_with(
            path="docs",
            settings=RetrievalSettings(
                store_type="numpy",
                store_path="shared-index",
                collection_name="guides",
                embedding_model="text-embedding-3-large",
                catalog_dsn=dsn,
            ),
            max_words=450,
        )
        provider.assert_not_called()
        legacy_store.assert_not_called()
        legacy_embedder.assert_not_called()
        self.assertEqual(
            output_stream.getvalue(), "Indexed 2 documents and 5 chunks.\n"
        )

    def test_indexing_defaults_use_shared_store_directories(self) -> None:
        for store_type in ("chroma", "numpy", "faiss"):
            with (
                self.subTest(store_type=store_type),
                patch("fieldguide_ai.cli.load_dotenv"),
                patch(
                    "fieldguide_ai.cli.index_corpus", return_value=IngestionResult(1, 1)
                ) as index,
                redirect_stdout(io.StringIO()),
            ):
                main(["--index-corpus", "docs", "--vector-store", store_type])
            settings = index.call_args.kwargs["settings"]
            expected = "chroma_db" if store_type == "chroma" else f"{store_type}_index"
            self.assertEqual(settings.store_path, expected)

    def test_init_catalog_creates_schema_without_models_or_indexing(self) -> None:
        dsn = "postgresql://test@localhost/test"
        output_stream = io.StringIO()
        with (
            patch("fieldguide_ai.cli.load_dotenv"),
            patch.dict(os.environ, {"POSTGRES_CONNECTIONSTRING": dsn}, clear=True),
            patch("fieldguide_ai.cli.build_catalog") as catalog,
            patch("fieldguide_ai.cli.build_provider") as provider,
            patch("fieldguide_ai.cli.OpenAIEmbeddingProvider") as legacy_embedder,
            patch("fieldguide_ai.cli.index_corpus") as index,
            redirect_stdout(output_stream),
        ):
            main(["--init-catalog"])

        catalog.assert_called_once_with(
            RetrievalSettings(catalog_dsn=dsn), initialize_schema=True
        )
        catalog.return_value.__exit__.assert_called_once()
        provider.assert_not_called()
        legacy_embedder.assert_not_called()
        index.assert_not_called()
        self.assertEqual(output_stream.getvalue(), "Catalog schema initialized.\n")

    def test_init_catalog_and_indexing_require_a_dsn(self) -> None:
        for arguments in (["--init-catalog"], ["--index-corpus", "docs"]):
            stderr = io.StringIO()
            with (
                self.subTest(arguments=arguments),
                patch("fieldguide_ai.cli.load_dotenv"),
                patch.dict(os.environ, {}, clear=True),
                redirect_stderr(stderr),
                self.assertRaisesRegex(SystemExit, "1"),
            ):
                main(arguments)
            self.assertIn("catalog DSN is required", stderr.getvalue())

    def test_catalog_initialization_is_a_separate_corpus_action(self) -> None:
        for action in ("--chunk-corpus", "--index-corpus"):
            with (
                self.subTest(action=action),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(["--init-catalog", action, "docs"])

    def test_main_preview_needs_no_credentials_or_catalog(self) -> None:
        with (
            TemporaryDirectory() as tmpdir,
            patch("fieldguide_ai.cli.load_dotenv"),
            patch.dict(os.environ, {}, clear=True),
            patch("fieldguide_ai.cli.build_catalog") as catalog,
            patch("fieldguide_ai.cli.build_provider") as provider,
            redirect_stdout(io.StringIO()) as output_stream,
        ):
            Path(tmpdir, "doc.md").write_text("# Test\n\nBody", encoding="utf-8")
            main(["--chunk-corpus", tmpdir])

        catalog.assert_not_called()
        provider.assert_not_called()
        self.assertIn(
            "Loaded 1 documents and created 1 chunks.", output_stream.getvalue()
        )

    def test_main_reports_missing_markdown_without_a_traceback(self) -> None:
        with TemporaryDirectory() as tmpdir:
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaisesRegex(SystemExit, "1"):
                main(["--chunk-corpus", str(Path(tmpdir, "missing.md"))])
        self.assertIn("Error: Markdown source does not exist", stderr.getvalue())

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
