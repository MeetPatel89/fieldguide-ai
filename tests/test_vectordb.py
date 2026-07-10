from pathlib import Path
import unittest
from unittest.mock import Mock

from fieldguide_ai.ingestion.models import DocumentChunk
from fieldguide_ai.vectordb import (
    ChromaVectorStore,
    OpenAIEmbeddingProvider,
    serialize_chunk_metadata,
)


class OpenAIEmbeddingProviderTest(unittest.TestCase):
    def test_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            OpenAIEmbeddingProvider(api_key=None)

    def test_embed_texts_calls_openai_with_configured_model(self) -> None:
        client = Mock()
        client.embeddings.create.return_value = Mock(
            data=[
                Mock(embedding=[0.1, 0.2]),
                Mock(embedding=[0.3, 0.4]),
            ]
        )
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-large",
            client=client,
        )

        embeddings = provider.embed_texts(["first", "second"])

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])
        client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large",
            input=["first", "second"],
            encoding_format="float",
        )

    def test_embed_texts_noops_for_empty_input(self) -> None:
        client = Mock()
        provider = OpenAIEmbeddingProvider(api_key="test-key", client=client)

        embeddings = provider.embed_texts([])

        self.assertEqual(embeddings, [])
        client.embeddings.create.assert_not_called()


class ChromaVectorStoreTest(unittest.TestCase):
    def test_creates_or_reuses_named_collection(self) -> None:
        client = Mock()
        collection = Mock()
        client.get_or_create_collection.return_value = collection

        store = ChromaVectorStore(
            path="/tmp/chroma",
            collection_name="knowledge-base",
            embedding_provider=Mock(),
            client=client,
        )

        self.assertIs(store.get_collection(), collection)
        client.get_or_create_collection.assert_called_once_with(name="knowledge-base")

    def test_index_chunks_upserts_chunk_payloads(self) -> None:
        collection = Mock()
        client = Mock()
        client.get_or_create_collection.return_value = collection
        embedding_provider = Mock()
        embedding_provider.embed_texts.return_value = [[0.11, 0.22], [0.33, 0.44]]
        store = ChromaVectorStore(
            path="/tmp/chroma",
            collection_name="documents",
            embedding_provider=embedding_provider,
            client=client,
        )
        chunks = [
            DocumentChunk(
                chunk_id="DOC-1::chunk-0000",
                doc_id="DOC-1",
                content="First chunk",
                source_path=Path("docs/doc-1.md"),
                section_path=("Doc 1", "Summary"),
                chunk_index=0,
                metadata={"doc_type": "runbook", "related_records": ["INC-1"]},
            ),
            DocumentChunk(
                chunk_id="DOC-1::chunk-0001",
                doc_id="DOC-1",
                content="Second chunk",
                source_path=Path("docs/doc-1.md"),
                section_path=("Doc 1", "Steps"),
                chunk_index=1,
                metadata={"servicenow": {"number": "KB0001"}},
            ),
        ]

        store.index_chunks(chunks)

        embedding_provider.embed_texts.assert_called_once_with(["First chunk", "Second chunk"])
        collection.upsert.assert_called_once()
        kwargs = collection.upsert.call_args.kwargs
        self.assertEqual(kwargs["ids"], ["DOC-1::chunk-0000", "DOC-1::chunk-0001"])
        self.assertEqual(kwargs["documents"], ["First chunk", "Second chunk"])
        self.assertEqual(kwargs["embeddings"], [[0.11, 0.22], [0.33, 0.44]])
        self.assertEqual(kwargs["metadatas"][0]["section_path"], "Doc 1 > Summary")
        self.assertEqual(kwargs["metadatas"][0]["related_records"], '["INC-1"]')
        self.assertEqual(kwargs["metadatas"][1]["servicenow"], '{"number": "KB0001"}')

    def test_index_chunks_noops_for_empty_input(self) -> None:
        collection = Mock()
        client = Mock()
        client.get_or_create_collection.return_value = collection
        embedding_provider = Mock()
        store = ChromaVectorStore(
            path="/tmp/chroma",
            embedding_provider=embedding_provider,
            client=client,
        )

        store.index_chunks([])

        embedding_provider.embed_texts.assert_not_called()
        collection.upsert.assert_not_called()

    def test_index_chunks_requires_embedding_provider(self) -> None:
        collection = Mock()
        client = Mock()
        client.get_or_create_collection.return_value = collection
        store = ChromaVectorStore(
            path="/tmp/chroma",
            embedding_provider=None,
            client=client,
        )

        with self.assertRaises(ValueError):
            store.index_chunks(
                [
                    DocumentChunk(
                        chunk_id="DOC-1::chunk-0000",
                        doc_id="DOC-1",
                        content="Chunk",
                        source_path=Path("doc.md"),
                        section_path=("Doc",),
                        chunk_index=0,
                        metadata={},
                    )
                ]
            )


class ChunkMetadataSerializationTest(unittest.TestCase):
    def test_serializes_nested_metadata_for_chroma(self) -> None:
        chunk = DocumentChunk(
            chunk_id="DOC-2::chunk-0003",
            doc_id="DOC-2",
            content="Body",
            source_path=Path("docs/doc-2.md"),
            section_path=("Doc 2", "Troubleshooting"),
            chunk_index=3,
            metadata={
                "doc_type": "known_issue",
                "related_records": ["INC-1", "KB-2"],
                "servicenow": {"number": "KB0002", "priority": 2},
                "source_facts": [{"kind": "ticket", "id": "INC-1"}],
            },
        )

        metadata = serialize_chunk_metadata(chunk)

        self.assertEqual(metadata["chunk_id"], "DOC-2::chunk-0003")
        self.assertEqual(metadata["doc_id"], "DOC-2")
        self.assertEqual(metadata["source_path"], "docs/doc-2.md")
        self.assertEqual(metadata["chunk_index"], 3)
        self.assertEqual(metadata["section_path"], "Doc 2 > Troubleshooting")
        self.assertEqual(metadata["section_path_json"], '["Doc 2", "Troubleshooting"]')
        self.assertEqual(metadata["section_title"], "Troubleshooting")
        self.assertEqual(metadata["doc_type"], "known_issue")
        self.assertEqual(metadata["related_records"], '["INC-1", "KB-2"]')
        self.assertEqual(metadata["servicenow"], '{"number": "KB0002", "priority": 2}')
        self.assertEqual(metadata["source_facts"], '[{"id": "INC-1", "kind": "ticket"}]')
