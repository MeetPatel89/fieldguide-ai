import unittest
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

from fieldguide_ai.ingestion import DocumentIndexingPipeline
from fieldguide_ai.ingestion.models import DocumentChunk, MarkdownDocument
from fieldguide_ai.vectorstore import VectorSearchResult, VectorStore


class RecordingVectorStore(VectorStore):
    def __init__(self) -> None:
        self.replacements = []
        self.deletions = []

    def index_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        raise AssertionError("pipeline must use document replacement")

    def replace_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        self.replacements.append(list(chunks))

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        self.deletions.append(list(doc_ids))

    def query(self, query_text: str, n_results: int = 10) -> list[VectorSearchResult]:
        return []


class DocumentIndexingPipelineTest(unittest.TestCase):
    def test_loads_chunks_and_replaces_each_document(self) -> None:
        store = RecordingVectorStore()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.md").write_text("# A\n\nAlpha", encoding="utf-8")
            (root / "b.md").write_text("# B\n\nBeta", encoding="utf-8")

            result = DocumentIndexingPipeline(store).index_path(root)

        self.assertEqual(result.document_count, 2)
        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(len(store.replacements), 2)

    def test_empty_document_deletes_its_existing_index(self) -> None:
        store = RecordingVectorStore()
        document = MarkdownDocument(
            source_path=Path("empty.md"), body="", metadata={"doc_id": "EMPTY"}
        )

        result = DocumentIndexingPipeline(store).index_documents([document])

        self.assertEqual(result.chunk_count, 0)
        self.assertEqual(store.deletions, [["EMPTY"]])

    def test_path_loading_is_an_injected_boundary(self) -> None:
        store = RecordingVectorStore()
        requested_paths: list[str | Path] = []
        document = MarkdownDocument(source_path=Path("virtual.md"), body="# Virtual")

        def load_documents(path: str | Path) -> list[MarkdownDocument]:
            requested_paths.append(path)
            return [document]

        result = DocumentIndexingPipeline(
            store,
            document_loader=load_documents,
        ).index_path("memory://corpus")

        self.assertEqual(requested_paths, ["memory://corpus"])
        self.assertEqual(result.document_count, 1)
        self.assertEqual(result.chunk_count, 1)
