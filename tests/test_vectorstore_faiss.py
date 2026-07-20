import json
import unittest
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from fieldguide_ai.ingestion.models import DocumentChunk
from fieldguide_ai.vectorstore import EmbeddingProvider, FaissVectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embeddings[text] for text in texts]


def make_chunk(chunk_id: str, doc_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        source_path=Path(f"docs/{doc_id}.md"),
        section_path=(doc_id, "Summary"),
        chunk_index=0,
        metadata={"doc_type": "runbook"},
    )


class FaissVectorStoreTest(unittest.TestCase):
    def test_index_query_replace_delete_and_persistence(self) -> None:
        provider = FakeEmbeddingProvider(
            {
                "alpha": [1.0, 0.0],
                "beta": [0.0, 1.0],
                "updated": [1.0, 1.0],
                "query": [1.0, 0.0],
            }
        )
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "knowledge"
            store = FaissVectorStore(provider, path)
            store.index_chunks(
                [
                    make_chunk("A::0", "A", "alpha"),
                    make_chunk("B::0", "B", "beta"),
                ]
            )

            results = store.query("query")
            self.assertEqual([result.chunk_id for result in results], ["A::0", "B::0"])
            self.assertAlmostEqual(results[0].distance, 0.0)

            store.replace_chunks([make_chunk("A::1", "A", "updated")])
            self.assertNotIn(
                "A::0", [result.chunk_id for result in store.query("query")]
            )
            store.delete_documents(["B"])

            reloaded = FaissVectorStore(provider, path)
            persisted_results = reloaded.query("query")
            self.assertEqual(
                [result.chunk_id for result in persisted_results], ["A::1"]
            )
            self.assertEqual(persisted_results[0].metadata["source_path"], "docs/A.md")
            self.assertTrue(Path(f"{path}.faiss").exists())
            sidecar = json.loads(Path(f"{path}.json").read_text(encoding="utf-8"))
            self.assertEqual(len(sidecar), 1)

    def test_rejects_zero_and_incompatible_vectors(self) -> None:
        provider = FakeEmbeddingProvider(
            {"alpha": [1.0, 0.0], "zero": [0.0, 0.0], "wide": [1.0, 0.0, 0.0]}
        )
        with TemporaryDirectory() as tmpdir:
            store = FaissVectorStore(provider, Path(tmpdir) / "index")
            store.index_chunks([make_chunk("A::0", "A", "alpha")])
            with self.assertRaisesRegex(ValueError, "non-zero"):
                store.index_chunks([make_chunk("B::0", "B", "zero")])
            with self.assertRaisesRegex(ValueError, "does not match index dimension"):
                store.index_chunks([make_chunk("C::0", "C", "wide")])
