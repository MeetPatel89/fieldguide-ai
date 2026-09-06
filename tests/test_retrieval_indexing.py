"""Offline coverage for the shared Markdown ingestion boundary."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import override
from unittest.mock import patch

from vectorstore import (
    CatalogChunk,
    EmbeddingProvider,
    EmbeddingSpec,
    SqliteDocumentCatalog,
)

from fieldguide_ai.errors import (
    ConfigurationError,
    DocumentLoadError,
    VectorStoreOperationError,
)
from fieldguide_ai.retrieval import (
    RetrievalMode,
    RetrievalSettings,
    build_store,
    chunk_corpus,
    index_corpus,
)

INDEXING = "fieldguide_ai.retrieval.indexing"
TEST_DSN = "postgresql://test@localhost/test"


class RecordingEmbedding(EmbeddingProvider):
    """Produce fixed-width vectors while recording exactly which text was embedded."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    @property
    @override
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(provider="test", model="recording", dimension=2)

    @override
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[1.0, float(len(text.split()))] for text in texts]


class CorpusIndexingTest(unittest.TestCase):
    @override
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.corpus = self.root / "docs"
        self.corpus.mkdir()
        self.document = self.corpus / "guide.md"
        self.document.write_text(
            "---\ndoc_id: GUIDE\ntitle: Support Guide\ndoc_type: runbook\n"
            "priority: 2\n---\n# Support Guide\n\n## Investigate\n\n"
            + " ".join(f"detail{index}" for index in range(220))
            + "\n\n## Resolve\n\nRestart the worker after reviewing the incident.\n",
            encoding="utf-8",
        )
        self.catalog_path = self.root / "catalog.sqlite"

    def open_catalog(self, settings: RetrievalSettings) -> SqliteDocumentCatalog:
        """Replace only the production database boundary with local SQLite."""
        return SqliteDocumentCatalog(self.catalog_path)

    def test_catalog_only_ingest_matches_preview_without_dense_dependencies(
        self,
    ) -> None:
        settings = RetrievalSettings(
            mode=RetrievalMode.LEXICAL,
            store_type=None,
            store_path=None,
            embedding_model=None,
            catalog_dsn=TEST_DSN,
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                f"{INDEXING}.build_catalog", side_effect=self.open_catalog
            ) as catalog,
            patch(f"{INDEXING}.build_embedder") as embedder,
            patch(f"{INDEXING}.build_store") as store,
            patch(f"{INDEXING}.EmbeddingRouter") as router,
            patch(f"{INDEXING}.persist_store") as persist,
        ):
            records, preview = chunk_corpus(self.corpus, max_words=120)
            result = index_corpus(self.corpus, settings, max_words=120)

        catalog.assert_called_once_with(settings)
        embedder.assert_not_called()
        store.assert_not_called()
        router.assert_not_called()
        persist.assert_not_called()
        self.assertEqual(result.document_count, 1)
        self.assertGreater(result.chunk_count, 2)
        self.assertEqual(result.chunk_count, len(preview))
        self.assertEqual(result.embedded_count, 0)
        self.assertEqual(result.embedded_by_space, {})
        self.assertTrue(all(isinstance(chunk, CatalogChunk) for chunk in preview))
        self.assertTrue(all(len(chunk.text.split()) <= 120 for chunk in preview))
        self.assertEqual(records[0].source, "guide.md")
        self.assertEqual(preview[0].section_path, "Support Guide > Investigate")
        with SqliteDocumentCatalog(self.catalog_path) as saved:
            self.assertEqual(
                saved.get_chunks([chunk.chunk_id for chunk in preview]), preview
            )
            documents = saved.find()
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].doc_id, "GUIDE")
            self.assertEqual(documents[0].title, "Support Guide")
            self.assertEqual(documents[0].source, "guide.md")
            self.assertEqual(documents[0].doc_type, "runbook")
            self.assertEqual(documents[0].attributes["priority"], 2)
            self.assertEqual(
                [hit.chunk_id for hit in saved.search_lexical("Restart")],
                [preview[-1].chunk_id],
            )

    def test_dense_and_hybrid_ingest_persist_vectors_and_skip_unchanged_text(
        self,
    ) -> None:
        for mode, store_type in (
            (RetrievalMode.DENSE, "numpy"),
            (RetrievalMode.HYBRID, "numpy"),
            (RetrievalMode.DENSE, "faiss"),
            (RetrievalMode.HYBRID, "faiss"),
        ):
            with self.subTest(mode=mode, store_type=store_type):
                self.catalog_path = self.root / f"{mode}-{store_type}.sqlite"
                settings = RetrievalSettings(
                    mode=mode,
                    store_type=store_type,
                    store_path=self.root / f"{mode}-{store_type}-vectors",
                    catalog_dsn=TEST_DSN,
                )
                embedder = RecordingEmbedding()
                _, expected = chunk_corpus(self.corpus, max_words=120)
                with (
                    patch(f"{INDEXING}.build_catalog", side_effect=self.open_catalog),
                    patch(f"{INDEXING}.build_embedder", return_value=embedder),
                ):
                    result = index_corpus(self.corpus, settings, max_words=120)
                    repeated = index_corpus(self.corpus, settings, max_words=120)

                self.assertEqual(result.document_count, 1)
                self.assertEqual(result.chunk_count, len(expected))
                self.assertEqual(
                    result.embedded_by_space, {embedder.spec.space_id: len(expected)}
                )
                self.assertEqual(embedder.batches, [[chunk.text for chunk in expected]])
                self.assertEqual(repeated.embedded_count, 0)
                self.assertEqual(repeated.skipped_embedding_count, len(expected))
                reloaded = build_store(settings, embedder.spec.dimension)
                self.assertEqual(reloaded.dimension, embedder.spec.dimension)
                self.assertEqual(reloaded.count(), len(expected))
                vectors = reloaded.get([chunk.chunk_id for chunk in expected])
                self.assertEqual(
                    [chunk.text for chunk in vectors],
                    [chunk.text for chunk in expected],
                )
                self.assertEqual(vectors[0].metadata["source"], "guide.md")
                self.assertEqual(vectors[0].metadata["doc_id"], "GUIDE")
                with SqliteDocumentCatalog(self.catalog_path) as saved:
                    self.assertEqual(
                        saved.get_chunks([chunk.chunk_id for chunk in expected]),
                        expected,
                    )
                    self.assertEqual(saved.stale_chunk_ids(embedder.spec), [])

    def test_reindex_replaces_superseded_chunks_and_preserves_other_documents(
        self,
    ) -> None:
        Path(self.corpus, "other.md").write_text(
            "# Other\n\nKeep this document.", encoding="utf-8"
        )
        settings = RetrievalSettings(
            store_type="numpy", store_path=self.root / "vectors", catalog_dsn=TEST_DSN
        )
        embedder = RecordingEmbedding()
        with (
            patch(f"{INDEXING}.build_catalog", side_effect=self.open_catalog),
            patch(f"{INDEXING}.build_embedder", return_value=embedder),
        ):
            original = index_corpus(self.corpus, settings, max_words=120)
            self.document.write_text(
                "---\ndoc_id: GUIDE\ntitle: Support Guide\n---\n"
                "# Support Guide\n\nReplacement procedure.\n",
                encoding="utf-8",
            )
            replacement = index_corpus(self.corpus, settings, max_words=120)

        self.assertEqual(replacement.document_count, 2)
        self.assertEqual(replacement.chunk_count, 2)
        self.assertEqual(replacement.removed_chunk_count, original.chunk_count - 2)
        self.assertEqual(replacement.embedded_count, 1)
        self.assertEqual(replacement.skipped_embedding_count, 1)
        _, expected = chunk_corpus(self.corpus, max_words=120)
        store = build_store(settings, embedder.spec.dimension)
        self.assertEqual(store.count(), 2)
        self.assertEqual(
            [chunk.text for chunk in store.get([chunk.chunk_id for chunk in expected])],
            [chunk.text for chunk in expected],
        )
        with SqliteDocumentCatalog(self.catalog_path) as saved:
            self.assertEqual(
                saved.get_chunks([chunk.chunk_id for chunk in expected]), expected
            )
            self.assertEqual(saved.get_chunks(["GUIDE::chunk-0001"]), [])

    def test_embedding_failure_is_reported_without_persisting_success(self) -> None:
        settings = RetrievalSettings(
            store_type="numpy", store_path=self.root / "vectors", catalog_dsn=TEST_DSN
        )
        embedder = RecordingEmbedding()
        with (
            patch(f"{INDEXING}.build_catalog", side_effect=self.open_catalog),
            patch(f"{INDEXING}.build_embedder", return_value=embedder),
            patch.object(embedder, "embed_texts", side_effect=RuntimeError("offline")),
            patch(f"{INDEXING}.persist_store") as persist,
            self.assertRaisesRegex(VectorStoreOperationError, "offline"),
        ):
            index_corpus(self.corpus, settings, max_words=120)

        persist.assert_not_called()
        self.assertFalse((self.root / "vectors").exists())
        with SqliteDocumentCatalog(self.catalog_path) as saved:
            self.assertEqual(len(saved.find()), 1)
            self.assertTrue(saved.stale_chunk_ids(embedder.spec))

    def test_invalid_chunk_sizes_fail_before_catalog_or_embeddings(self) -> None:
        for max_words in (0, -1, 75):
            with (
                self.subTest(max_words=max_words),
                patch(f"{INDEXING}.build_catalog") as catalog,
                patch(f"{INDEXING}.build_embedder") as embedder,
            ):
                with self.assertRaises(ConfigurationError):
                    index_corpus(self.corpus, RetrievalSettings(), max_words)
                with self.assertRaises(ConfigurationError):
                    chunk_corpus(self.corpus, max_words)
                catalog.assert_not_called()
                embedder.assert_not_called()

    def test_missing_source_is_reported_without_persisting(self) -> None:
        settings = RetrievalSettings(mode=RetrievalMode.LEXICAL, catalog_dsn=TEST_DSN)
        with (
            patch(f"{INDEXING}.build_catalog", side_effect=self.open_catalog),
            patch(f"{INDEXING}.persist_store") as persist,
            self.assertRaises(DocumentLoadError),
        ):
            index_corpus(self.corpus / "missing.md", settings, max_words=120)
        persist.assert_not_called()
