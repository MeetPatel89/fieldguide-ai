import unittest
from dataclasses import replace
from unittest.mock import Mock

from vectorstore import (
    CatalogChunk,
    CatalogDocument,
    QueryKind,
    RetrievalCatalog,
    RetrievalHit,
    RetrievalResult,
    RetrievalTimings,
)

from fieldguide_ai.retrieval import (
    RetrievalDiagnostics,
    RetrievalMode,
    RetrievedSource,
    sources_from_result,
)


def make_result(hits: tuple[RetrievalHit, ...] = ()) -> RetrievalResult:
    return RetrievalResult(
        hits=hits,
        query_id="query-1",
        query_kind=QueryKind.NATURAL,
        provider="openai",
        model="text-embedding-3-small",
        space_id="test-space",
        provider_reason="primary",
        fallback_occurred=False,
        degraded=False,
        dense_candidates=3,
        lexical_candidates=2,
        dense_weight=1.0,
        lexical_weight=1.0,
        timings=RetrievalTimings(
            dense_ms=4.0,
            lexical_ms=2.0,
            total_ms=7.0,
            embedding_ms=3.0,
            dense_search_ms=1.0,
            fusion_ms=1.0,
        ),
    )


class RetrievedSourceTest(unittest.TestCase):
    def test_document_lookup_is_batched_and_preserves_hit_order_and_ranks(self) -> None:
        first = RetrievalHit(
            CatalogChunk("B-1", "DOC-B", "First text", section_path="Guide > Recovery"),
            score=0.03,
            dense_rank=1,
            lexical_rank=2,
        )
        second = RetrievalHit(
            CatalogChunk("A-1", "DOC-A", "Second text"),
            score=0.02,
            lexical_rank=1,
        )
        third = RetrievalHit(
            CatalogChunk("B-2", "DOC-B", "Third text"), score=0.01, dense_rank=2
        )
        catalog = Mock(spec=RetrievalCatalog)
        catalog.find.return_value = [
            CatalogDocument("DOC-A", title="A title", source="docs/a.md"),
            CatalogDocument("DOC-B", title="B title", source="docs/b.md"),
        ]

        sources = sources_from_result(make_result((first, second, third)), catalog)

        catalog.find.assert_called_once_with(
            {"doc_id": {"$in": ["DOC-B", "DOC-A"]}}, limit=2
        )
        self.assertEqual([source.chunk_id for source in sources], ["B-1", "A-1", "B-2"])
        self.assertEqual(
            sources[0],
            RetrievedSource(
                chunk_id="B-1",
                doc_id="DOC-B",
                source="docs/b.md",
                title="B title",
                section_path="Guide > Recovery",
                text="First text",
                score=0.03,
                dense_rank=1,
                lexical_rank=2,
            ),
        )
        self.assertIsNone(sources[1].dense_rank)
        self.assertIsNone(sources[2].lexical_rank)
        self.assertEqual(sources[2].source, "docs/b.md")

    def test_missing_document_metadata_does_not_drop_a_hit(self) -> None:
        hit = RetrievalHit(CatalogChunk("one", "missing", "Retrieved text"), score=0.1)
        catalog = Mock(spec=RetrievalCatalog)
        catalog.find.return_value = []
        sources = sources_from_result(make_result((hit,)), catalog)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].doc_id, "missing")
        self.assertEqual(sources[0].text, "Retrieved text")
        self.assertIsNone(sources[0].source)
        self.assertIsNone(sources[0].title)
        self.assertIsNone(sources[0].section_path)

    def test_empty_hits_do_not_query_the_catalog(self) -> None:
        catalog = Mock(spec=RetrievalCatalog)
        self.assertEqual(sources_from_result(make_result(), catalog), ())
        catalog.find.assert_not_called()

    def test_lookup_limit_covers_more_than_the_catalog_default(self) -> None:
        hits = tuple(
            RetrievalHit(CatalogChunk(f"chunk-{i}", f"doc-{i}", "Text"), score=0.01)
            for i in range(101)
        )
        catalog = Mock(spec=RetrievalCatalog)
        catalog.find.return_value = []
        self.assertEqual(len(sources_from_result(make_result(hits), catalog)), 101)
        self.assertEqual(catalog.find.call_args.kwargs["limit"], 101)

    def test_mismatched_document_cannot_be_attached_to_a_hit(self) -> None:
        hit = RetrievalHit(CatalogChunk("one", "DOC-1", "Text"), score=0.1)
        with self.assertRaisesRegex(ValueError, "does not match"):
            RetrievedSource.from_hit(hit, CatalogDocument("DOC-2"))


class RetrievalDiagnosticsTest(unittest.TestCase):
    def test_diagnostics_preserve_requested_mode_and_all_timing_fields(self) -> None:
        result = replace(
            make_result(), degraded=True, errors=("lexical unavailable: unavailable",)
        )
        diagnostics = RetrievalDiagnostics.from_result(result, RetrievalMode.HYBRID)
        self.assertIs(diagnostics.mode, RetrievalMode.HYBRID)
        self.assertTrue(diagnostics.degraded)
        self.assertEqual(diagnostics.errors, result.errors)
        self.assertEqual(diagnostics.provider, "openai")
        self.assertEqual(diagnostics.timings, result.timings)

    def test_lexical_diagnostics_preserve_absent_provider_and_skipped_timings(
        self,
    ) -> None:
        result = replace(
            make_result(),
            provider=None,
            model=None,
            space_id=None,
            provider_reason=None,
            timings=RetrievalTimings(dense_ms=None, lexical_ms=2.0, total_ms=2.5),
        )
        diagnostics = RetrievalDiagnostics.from_result(result, RetrievalMode.LEXICAL)
        self.assertFalse(diagnostics.degraded)
        self.assertEqual(diagnostics.errors, ())
        self.assertIsNone(diagnostics.provider)
        self.assertIsNone(diagnostics.timings.embedding_ms)
        self.assertIsNone(diagnostics.timings.dense_ms)
