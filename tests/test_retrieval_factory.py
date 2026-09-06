import os
import unittest
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import Mock, patch

from vectorstore import (
    CatalogChunk,
    Chunk,
    EmbeddingProvider,
    EmbeddingSpec,
    NumpyVectorStore,
    OpenAIEmbedding,
    RankedHit,
    RetrievalCatalog,
)

from fieldguide_ai.errors import (
    ConfigurationError,
    InvalidVectorStoreError,
    VectorStoreOperationError,
)
from fieldguide_ai.retrieval import (
    RetrievalMode,
    RetrievalSettings,
    build_catalog,
    build_embedder,
    build_retriever,
    build_store,
    persist_store,
    retriever_config_for,
)

FACTORY = "fieldguide_ai.retrieval.factory"
TEST_DSN = "postgresql://test:test@localhost/fieldguide_test"


class RetrievalModeTest(unittest.TestCase):
    def test_modes_select_only_the_requested_signals(self) -> None:
        for mode, dense, lexical in (
            (RetrievalMode.DENSE, True, False),
            (RetrievalMode.LEXICAL, False, True),
            (RetrievalMode.HYBRID, True, True),
        ):
            with self.subTest(mode=mode):
                config = retriever_config_for(mode, top_k=7)
                self.assertEqual(config.dense_enabled, dense)
                self.assertEqual(config.lexical_enabled, lexical)
                self.assertEqual(config.final_top_k, 7)
                self.assertEqual(str(mode), mode.value)

    def test_rejects_invalid_counts_and_modes(self) -> None:
        for top_k in (0, -1, True, 1.5):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ConfigurationError, "positive integer"):
                    retriever_config_for(RetrievalMode.DENSE, cast(int, top_k))
        with self.assertRaisesRegex(ConfigurationError, "unsupported retrieval mode"):
            retriever_config_for(cast(RetrievalMode, "unknown"), 3)


class RetrievalSettingsTest(unittest.TestCase):
    def test_settings_are_frozen_and_do_not_expose_the_dsn_in_repr(self) -> None:
        settings = RetrievalSettings(catalog_dsn=TEST_DSN)
        with self.assertRaises(FrozenInstanceError):
            settings.mode = RetrievalMode.LEXICAL  # type: ignore[misc]  # ty: ignore[invalid-assignment]
        self.assertNotIn(TEST_DSN, repr(settings))

    def test_normalizes_a_serialized_mode(self) -> None:
        settings = RetrievalSettings(mode=cast(RetrievalMode, "dense"))
        self.assertIs(settings.mode, RetrievalMode.DENSE)

    def test_lexical_needs_no_store_or_embedding_model(self) -> None:
        settings = RetrievalSettings(
            mode=RetrievalMode.LEXICAL,
            store_type=None,
            store_path=None,
            embedding_model=None,
            catalog_dsn=TEST_DSN,
        )
        self.assertIsNone(settings.store_type)
        self.assertIsNone(settings.embedding_model)

    def test_dense_settings_can_use_an_injected_catalog_and_ephemeral_store(
        self,
    ) -> None:
        settings = RetrievalSettings(store_type="numpy", store_path=None)
        self.assertIsNone(settings.catalog_dsn)
        self.assertIsNone(settings.store_path)

    def test_incomplete_or_unknown_configuration_is_rejected(self) -> None:
        cases: list[tuple[Callable[[], RetrievalSettings], str]] = [
            (
                lambda: RetrievalSettings(mode=cast(RetrievalMode, "unknown")),
                "unsupported retrieval mode",
            ),
            (
                lambda: RetrievalSettings(store_type="unknown"),
                "unsupported vector store",
            ),
            (lambda: RetrievalSettings(store_type=None), "require a store"),
            (lambda: RetrievalSettings(embedding_model=None), "require an embedding"),
            (lambda: RetrievalSettings(embedding_model="  "), "embedding model"),
            (lambda: RetrievalSettings(store_path="  "), "store path"),
            (lambda: RetrievalSettings(store_path=None), "Chroma requires"),
            (lambda: RetrievalSettings(collection_name="  "), "collection name"),
            (lambda: RetrievalSettings(catalog_dsn="  "), "catalog DSN"),
        ]
        for constructor, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ConfigurationError, message):
                    constructor()
        for mode in (RetrievalMode.LEXICAL, RetrievalMode.HYBRID):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(
                    ConfigurationError, "require a catalog DSN"
                ):
                    RetrievalSettings(mode=mode)
        settings = RetrievalSettings(mode=RetrievalMode.HYBRID, catalog_dsn=TEST_DSN)
        with self.assertRaisesRegex(ConfigurationError, "require a store"):
            replace(settings, store_type=None)
        with self.assertRaisesRegex(ConfigurationError, "require an embedding model"):
            replace(settings, embedding_model=None)


class RetrievalInfrastructureFactoryTest(unittest.TestCase):
    def test_embedder_receives_the_model_and_explicit_or_environment_key(self) -> None:
        settings = RetrievalSettings(embedding_model="text-embedding-3-large")
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "environment-key"}, clear=True),
            patch(f"{FACTORY}.OpenAIEmbedding") as constructor,
        ):
            self.assertIs(build_embedder(settings), constructor.return_value)
            constructor.assert_called_once_with(
                model="text-embedding-3-large", api_key="environment-key"
            )
            constructor.reset_mock()
            build_embedder(settings, api_key="explicit-key")
            constructor.assert_called_once_with(
                model="text-embedding-3-large", api_key="explicit-key"
            )

    def test_embedder_requires_credentials_before_creating_a_client(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(f"{FACTORY}.OpenAIEmbedding") as constructor,
        ):
            with self.assertRaisesRegex(ConfigurationError, "OPENAI_API_KEY"):
                build_embedder(RetrievalSettings())
            with self.assertRaisesRegex(ConfigurationError, "OPENAI_API_KEY"):
                build_embedder(RetrievalSettings(), api_key="  ")
            constructor.assert_not_called()

    def test_catalog_schema_creation_is_explicit(self) -> None:
        settings = RetrievalSettings(catalog_dsn=TEST_DSN)
        with patch(f"{FACTORY}.PostgresDocumentCatalog") as constructor:
            self.assertIs(build_catalog(settings), constructor.return_value)
            constructor.assert_called_once_with(TEST_DSN, initialize_schema=False)
            constructor.reset_mock()
            build_catalog(settings, initialize_schema=True)
            constructor.assert_called_once_with(TEST_DSN, initialize_schema=True)

    def test_default_dense_catalog_factory_also_requires_a_dsn(self) -> None:
        with patch(f"{FACTORY}.PostgresDocumentCatalog") as constructor:
            with self.assertRaisesRegex(ConfigurationError, "catalog DSN"):
                build_catalog(RetrievalSettings())
            constructor.assert_not_called()

    def test_numpy_and_faiss_save_load_round_trip_preserves_search(self) -> None:
        chunks = [
            Chunk(id="alpha", text="Alpha source", metadata={"doc_id": "DOC-1"}),
            Chunk(id="beta", text="Beta source", metadata={"doc_id": "DOC-2"}),
        ]
        with TemporaryDirectory() as directory:
            for store_type in ("numpy", "faiss"):
                with self.subTest(store_type=store_type):
                    path = Path(directory) / store_type / "index"
                    settings = RetrievalSettings(store_type=store_type, store_path=path)
                    store = build_store(settings, dimension=2)
                    self.assertEqual(store.dimension, 2)
                    self.assertEqual(store.count(), 0)
                    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

                    persist_store(store, settings)
                    reloaded = build_store(settings, dimension=2)
                    hits = reloaded.search([0.0, 1.0], k=1)

                    self.assertIsNot(reloaded, store)
                    self.assertEqual(reloaded.dimension, 2)
                    self.assertEqual(reloaded.count(), 2)
                    self.assertEqual(hits[0].chunk, chunks[1])
                    self.assertAlmostEqual(hits[0].score, 1.0)
                    self.assertTrue(path.is_dir())

    def test_rejects_persisted_dimension_mismatches(self) -> None:
        with TemporaryDirectory() as directory:
            for store_type in ("numpy", "faiss"):
                with self.subTest(store_type=store_type):
                    settings = RetrievalSettings(
                        store_type=store_type, store_path=Path(directory) / store_type
                    )
                    persist_store(build_store(settings, dimension=2), settings)
                    with self.assertRaisesRegex(ConfigurationError, "dimension"):
                        build_store(settings, dimension=3)

    def test_existing_invalid_indexes_are_not_silently_replaced(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid"
            path.mkdir()
            marker = path / "chunks.json"
            marker.write_text("invalid data", encoding="utf-8")
            for store_type in ("numpy", "faiss"):
                with self.subTest(store_type=store_type):
                    settings = RetrievalSettings(store_type=store_type, store_path=path)
                    with self.assertRaises(InvalidVectorStoreError) as raised:
                        build_store(settings, dimension=2)
                    self.assertIsNotNone(raised.exception.__cause__)
                    self.assertEqual(marker.read_text(encoding="utf-8"), "invalid data")

    def test_chroma_receives_path_and_collection_and_needs_no_explicit_save(
        self,
    ) -> None:
        settings = RetrievalSettings(
            store_type="chroma", store_path=Path("custom-index"), collection_name="kb"
        )
        with patch(f"{FACTORY}.create_store") as constructor:
            store = build_store(settings, dimension=1536)
            constructor.assert_called_once_with(
                "chroma", path=Path("custom-index"), collection_name="kb"
            )
            persist_store(store, settings)
            constructor.return_value.save.assert_not_called()

    def test_numpy_without_a_path_stays_in_memory(self) -> None:
        settings = RetrievalSettings(store_type="numpy", store_path=None)
        store = build_store(settings, dimension=2)
        store.upsert([Chunk(id="one", text="Text")], [[1.0, 0.0]])
        with patch.object(NumpyVectorStore, "save") as save:
            persist_store(store, settings)
            save.assert_not_called()
        self.assertEqual(store.count(), 1)

    def test_persistence_errors_keep_the_infrastructure_cause(self) -> None:
        settings = RetrievalSettings(store_type="numpy", store_path="index")
        store = NumpyVectorStore(dimension=2)
        error = OSError("disk unavailable")
        with patch.object(NumpyVectorStore, "save", side_effect=error):
            with self.assertRaises(VectorStoreOperationError) as raised:
                persist_store(store, settings)
        self.assertIs(raised.exception.__cause__, error)


class RetrieverCompositionTest(unittest.TestCase):
    def test_lexical_build_and_query_do_not_use_dense_dependencies(self) -> None:
        settings = RetrievalSettings(
            mode=RetrievalMode.LEXICAL,
            store_type=None,
            store_path=None,
            embedding_model=None,
            catalog_dsn=TEST_DSN,
        )
        catalog = Mock(spec=RetrievalCatalog)
        chunk = CatalogChunk("incident-1", "incident", "INC-20431 recovery steps")
        catalog.search_lexical.return_value = [RankedHit(chunk.chunk_id, 1, 0.5)]
        catalog.get_chunks.return_value = [chunk]
        embedder = Mock(spec=EmbeddingProvider)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(f"{FACTORY}.build_embedder") as embedder_factory,
            patch(f"{FACTORY}.build_store") as store_factory,
            patch(f"{FACTORY}.build_catalog") as catalog_factory,
        ):
            retriever = build_retriever(settings, 3, catalog=catalog, embedder=embedder)
            result = retriever.retrieve("INC-20431")
            embedder_factory.assert_not_called()
            store_factory.assert_not_called()
            catalog_factory.assert_not_called()
        self.assertEqual(embedder.mock_calls, [])
        self.assertFalse(retriever.config.dense_enabled)
        self.assertTrue(retriever.config.lexical_enabled)
        self.assertEqual(result.final_top_k, 3)
        self.assertIsNone(result.provider)
        self.assertIsNone(result.timings.dense_ms)
        self.assertFalse(result.degraded)
        self.assertEqual(result.hits[0].chunk, chunk)
        self.assertIsNone(result.hits[0].dense_rank)
        self.assertEqual(result.hits[0].lexical_rank, 1)
        catalog.search_lexical.assert_called_once()

    def test_dense_modes_bind_store_dimension_to_the_injected_embedder(self) -> None:
        for mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID):
            with self.subTest(mode=mode):
                settings = RetrievalSettings(mode=mode, catalog_dsn=TEST_DSN)
                catalog = Mock(spec=RetrievalCatalog)
                embedder = Mock(spec=EmbeddingProvider)
                embedder.spec = EmbeddingSpec("test", "test-model", 3)
                store = NumpyVectorStore(dimension=3)
                with (
                    patch(f"{FACTORY}.build_embedder") as embedder_factory,
                    patch(f"{FACTORY}.build_catalog") as catalog_factory,
                    patch(
                        f"{FACTORY}.build_store", return_value=store
                    ) as store_factory,
                    patch(f"{FACTORY}.compose_retriever") as compose,
                ):
                    retriever = build_retriever(
                        settings, 4, catalog=catalog, embedder=embedder
                    )
                    self.assertIs(retriever, compose.return_value)
                    embedder_factory.assert_not_called()
                    catalog_factory.assert_not_called()
                    store_factory.assert_called_once_with(settings, 3)
                    compose.assert_called_once_with(
                        catalog,
                        primary=embedder,
                        primary_store=store,
                        config=retriever_config_for(mode, 4),
                    )

    def test_default_factory_builds_all_dense_dependencies(self) -> None:
        settings = RetrievalSettings(catalog_dsn=TEST_DSN)
        catalog = Mock(spec=RetrievalCatalog)
        embedder = Mock(spec=OpenAIEmbedding)
        embedder.spec = EmbeddingSpec("openai", "text-embedding-3-small", 1536)
        with (
            patch(f"{FACTORY}.build_catalog", return_value=catalog) as catalog_factory,
            patch(
                f"{FACTORY}.build_embedder", return_value=embedder
            ) as embedder_factory,
            patch(
                f"{FACTORY}.build_store", return_value=NumpyVectorStore(dimension=1536)
            ) as store_factory,
        ):
            retriever = build_retriever(settings, 6)
            catalog_factory.assert_called_once_with(settings)
            embedder_factory.assert_called_once_with(settings)
            store_factory.assert_called_once_with(settings, 1536)
        self.assertEqual(retriever.config.final_top_k, 6)
        self.assertTrue(retriever.config.dense_enabled)
        self.assertFalse(retriever.config.lexical_enabled)

    def test_invalid_top_k_fails_before_infrastructure_construction(self) -> None:
        with patch(f"{FACTORY}.build_catalog") as catalog_factory:
            with self.assertRaises(ConfigurationError):
                build_retriever(RetrievalSettings(), 0)
            catalog_factory.assert_not_called()
