from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fieldguide_ai.ingestion import DocumentIndexingPipeline
from fieldguide_ai.ingestion.models import MarkdownDocument
from fieldguide_ai.vectorstore import VectorStore


class RecordingVectorStore(VectorStore):
    def __init__(self) -> None:
        self.replacements = []
        self.deletions = []

    def index_chunks(self, chunks) -> None:
        raise AssertionError("pipeline must use document replacement")

    def replace_chunks(self, chunks) -> None:
        self.replacements.append(list(chunks))

    def delete_documents(self, doc_ids) -> None:
        self.deletions.append(list(doc_ids))

    def query(self, query_text, n_results=10):
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
