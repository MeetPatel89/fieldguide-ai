import unittest
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from fieldguide_ai.errors import EmbeddingError, VectorStoreOperationError
from fieldguide_ai.ingestion.models import DocumentChunk
from fieldguide_ai.vectorstore import (
    ChromaVectorStore,
    EmbeddingProvider,
    NumpyVectorStore,
    OpenAIEmbeddingProvider,
    serialize_chunk_metadata,
)


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.embeddings[text] for text in texts]


def make_chunk(
    chunk_id: str,
    doc_id: str,
    content: str,
    chunk_index: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        source_path=Path(f"docs/{doc_id}.md"),
        section_path=(doc_id, "Summary"),
        chunk_index=chunk_index,
        metadata={"doc_type": "runbook"},
    )


class OpenAIEmbeddingProviderTest(unittest.TestCase):
    def test_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            OpenAIEmbeddingProvider(api_key=None)

    def test_embed_texts_calls_openai_with_configured_model(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
        )
        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-3-large", client=client
        )

        self.assertEqual(
            provider.embed_texts(["first", "second"]), [[0.1, 0.2], [0.3, 0.4]]
        )
        client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large",
            input=["first", "second"],
            encoding_format="float",
        )

    def test_embed_texts_noops_for_empty_input(self) -> None:
        client = Mock()
        provider = OpenAIEmbeddingProvider(api_key="test-key", client=client)
        self.assertEqual(provider.embed_texts([]), [])
        client.embeddings.create.assert_not_called()

    def test_injected_client_does_not_require_an_api_key(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = Mock(data=[])

        provider = OpenAIEmbeddingProvider(api_key=None, client=client)

        self.assertEqual(provider.embed_texts(["text"]), [])

    def test_translates_sdk_errors_and_preserves_context(self) -> None:
        client = Mock()
        sdk_error = ConnectionError("network unavailable")
        client.embeddings.create.side_effect = sdk_error
        provider = OpenAIEmbeddingProvider(api_key=None, client=client)

        with self.assertRaises(EmbeddingError) as raised:
            provider.embed_texts(["text"])

        self.assertIs(raised.exception.__cause__, sdk_error)


class ChromaVectorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.collection = Mock()
        self.client = Mock()
        self.client.get_or_create_collection.return_value = self.collection
        self.embedding_provider = FakeEmbeddingProvider(
            {"First chunk": [1.0, 0.0], "query": [1.0, 0.0]}
        )
        self.store = ChromaVectorStore(
            path="/tmp/chroma",
            collection_name="knowledge-base",
            embedding_provider=self.embedding_provider,
            client=self.client,
        )

    def test_creates_or_reuses_named_collection(self) -> None:
        self.client.get_or_create_collection.assert_called_once_with(
            name="knowledge-base"
        )

    def test_result_metadata_is_a_defensive_copy(self) -> None:
        self.collection.query.return_value = {
            "ids": [["DOC-1::chunk-0000"]],
            "documents": [["First chunk"]],
            "metadatas": [[{"doc_id": "DOC-1", "tags": ["original"]}]],
            "distances": [[0.125]],
        }

        result = self.store.query("query", n_results=1)[0]
        result.metadata["tags"].append("changed")

        self.assertEqual(result.metadata["tags"], ["original"])

    def test_indexes_and_replaces_document_chunks(self) -> None:
        chunk = make_chunk("DOC-1::chunk-0000", "DOC-1", "First chunk")
        self.store.replace_chunks([chunk])

        self.collection.delete.assert_called_once_with(where={"doc_id": "DOC-1"})
        kwargs = self.collection.upsert.call_args.kwargs
        self.assertEqual(kwargs["ids"], ["DOC-1::chunk-0000"])
        self.assertEqual(kwargs["embeddings"], [[1.0, 0.0]])
        self.assertEqual(kwargs["metadatas"][0]["doc_id"], "DOC-1")

    def test_query_returns_normalized_results(self) -> None:
        self.collection.query.return_value = {
            "ids": [["DOC-1::chunk-0000"]],
            "documents": [["First chunk"]],
            "metadatas": [[{"doc_id": "DOC-1"}]],
            "distances": [[0.125]],
        }

        results = self.store.query("query", n_results=3)

        self.assertEqual(results[0].chunk_id, "DOC-1::chunk-0000")
        self.assertEqual(results[0].content, "First chunk")
        self.assertEqual(results[0].distance, 0.125)
        self.collection.query.assert_called_once_with(
            query_embeddings=[[1.0, 0.0]],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

    def test_query_translates_chroma_errors(self) -> None:
        chroma_error = RuntimeError("database unavailable")
        self.collection.query.side_effect = chroma_error

        with self.assertRaises(VectorStoreOperationError) as raised:
            self.store.query("query")

        self.assertIs(raised.exception.__cause__, chroma_error)

    def test_empty_index_is_a_noop(self) -> None:
        self.store.index_chunks([])
        self.collection.upsert.assert_not_called()

    def test_embedding_failure_does_not_delete_existing_document(self) -> None:
        self.embedding_provider.embeddings.clear()

        with self.assertRaises(KeyError):
            self.store.replace_chunks(
                [make_chunk("DOC-1::chunk-0000", "DOC-1", "unavailable")]
            )

        self.collection.delete.assert_not_called()
        self.collection.upsert.assert_not_called()


class NumpyVectorStoreTest(unittest.TestCase):
    def test_cosine_query_is_nearest_first_with_stable_ties(self) -> None:
        provider = FakeEmbeddingProvider(
            {
                "alpha": [1.0, 0.0],
                "beta": [0.0, 1.0],
                "same": [1.0, 0.0],
                "q": [1.0, 0.0],
            }
        )
        store = NumpyVectorStore(provider)
        store.index_chunks(
            [
                make_chunk("z", "DOC-1", "same"),
                make_chunk("a", "DOC-2", "alpha"),
                make_chunk("b", "DOC-3", "beta"),
            ]
        )

        results = store.query("q", n_results=3)

        self.assertEqual([result.chunk_id for result in results], ["a", "z", "b"])
        self.assertAlmostEqual(results[0].distance, 0.0)
        self.assertAlmostEqual(results[2].distance, 1.0)

    def test_replace_removes_stale_document_chunks(self) -> None:
        provider = FakeEmbeddingProvider(
            {
                "old zero": [1.0, 0.0],
                "old one": [0.0, 1.0],
                "new": [1.0, 1.0],
                "q": [1.0, 1.0],
            }
        )
        store = NumpyVectorStore(provider)
        store.index_chunks(
            [
                make_chunk("DOC-1::chunk-0000", "DOC-1", "old zero"),
                make_chunk("DOC-1::chunk-0001", "DOC-1", "old one", 1),
            ]
        )
        store.replace_chunks([make_chunk("DOC-1::chunk-0000", "DOC-1", "new")])

        results = store.query("q", n_results=10)

        self.assertEqual([result.chunk_id for result in results], ["DOC-1::chunk-0000"])
        self.assertEqual(results[0].content, "new")

    def test_persistence_round_trip_uses_safe_numpy_data(self) -> None:
        provider = FakeEmbeddingProvider({"alpha": [1.0, 0.0], "q": [1.0, 0.0]})
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index.npz"
            NumpyVectorStore(provider, path=path).index_chunks(
                [make_chunk("DOC-1::chunk-0000", "DOC-1", "alpha")]
            )

            reloaded = NumpyVectorStore(provider, path=path)
            results = reloaded.query("q")

        self.assertEqual(results[0].chunk_id, "DOC-1::chunk-0000")
        self.assertEqual(results[0].metadata["doc_id"], "DOC-1")

    def test_rejects_incompatible_embedding_dimensions(self) -> None:
        provider = FakeEmbeddingProvider({"alpha": [1.0, 0.0], "beta": [1.0, 0.0, 0.0]})
        store = NumpyVectorStore(provider)
        store.index_chunks([make_chunk("a", "DOC-1", "alpha")])

        with self.assertRaisesRegex(ValueError, "does not match index dimension"):
            store.index_chunks([make_chunk("b", "DOC-2", "beta")])


class ChunkMetadataSerializationTest(unittest.TestCase):
    def test_serializes_nested_metadata_for_all_stores(self) -> None:
        chunk = DocumentChunk(
            chunk_id="DOC-2::chunk-0003",
            doc_id="DOC-2",
            content="Body",
            source_path=Path("docs/doc-2.md"),
            section_path=("Doc 2", "Troubleshooting"),
            chunk_index=3,
            metadata={
                "related_records": ["INC-1", "KB-2"],
                "servicenow": {"number": "KB0002", "priority": 2},
            },
        )

        metadata = serialize_chunk_metadata(chunk)

        self.assertEqual(metadata["section_path"], "Doc 2 > Troubleshooting")
        self.assertEqual(metadata["section_path_json"], '["Doc 2", "Troubleshooting"]')
        self.assertEqual(metadata["related_records"], '["INC-1", "KB-2"]')
        self.assertEqual(metadata["servicenow"], '{"number": "KB0002", "priority": 2}')
