"""#73: catalog / query routing / evidence coverage, using unseen synthetic reports."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

import app
from report_library import ReportLibrary, classify_question, select_reports
from test_app import LocalEmbeddings, write_pdf


class ReportLibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "reports.json"
        self.catalog = ReportLibrary(self.root / "catalog.sqlite3")
        self.specs = [
            dict(id="new-solar", title="하늘 태양광 보고서", published_date="2028-02-15", topics=["태양광"],
                 summary="재생에너지 발전의 관측 결과", path="solar.pdf"),
            dict(id="new-freight", title="별빛 물류 보고서", published_date="2028-01-10", topics=["물류"],
                 summary="물류 산업 전망", path="freight.pdf"),
            dict(id="new-crop", title="새로운 농업 보고서", published_date="2027-12-20", topics=["농업"],
                 summary="농업 수확 분석", path="crop.pdf"),
        ]
        for spec, text in zip(self.specs, ("apple solar energy", "banana freight transport", "cherry crops")):
            write_pdf(self.root / spec["path"], [text])
        self.manifest.write_text(json.dumps(self.specs, ensure_ascii=False), encoding="utf-8")
        self.reports = self.catalog.sync(self.manifest)
        self.patch = patch.object(app, "LOG_PATH", self.root / "qa.jsonl")
        self.patch.start()
        self.addCleanup(self.patch.stop)
        app._answer_cache.clear()

    def test_persistence_metadata_search_and_manifest_updates(self):
        reopened = ReportLibrary(self.catalog.path)
        self.assertEqual(len(reopened.list_reports()), 3)
        self.assertEqual(reopened.search("2028년 2월 태양광 보고서 찾아줘")[0]["report_id"], "new-solar")
        self.assertEqual(reopened.search("별빛 물류")[0]["report_id"], "new-freight")
        self.assertEqual(reopened.search("2030년 보고서"), [])
        self.manifest.write_text(json.dumps(self.specs[:1], ensure_ascii=False), encoding="utf-8")
        self.assertEqual(len(self.catalog.sync(self.manifest)), 1)

    def test_date_and_topic_are_intersected_not_merged(self):
        records = [dict(r, published_date="2028-02-15") for r in self.reports]
        chosen = select_reports("2028년 2월 태양광 보고서 찾아줘", records, "documents")
        self.assertEqual([r["report_id"] for r in chosen], ["new-solar"])

    def test_intent_and_report_selection_do_not_depend_on_seed_reports(self):
        for question, expected in [("태양광 보고서 찾아줘", "documents"), ("태양광 생산량은?", "fact"),
                                   ("태양광과 물류의 공통점 비교", "comparison"), ("에너지 종합 제안", "synthesis")]:
            self.assertEqual(classify_question(question), expected)
        chosen = select_reports("태양광과 물류를 비교해줘", self.reports, "comparison")
        self.assertEqual({r["report_id"] for r in chosen}, {"new-solar", "new-freight"})
        self.assertEqual(len(select_reports("모든 보고서 종합 제안", self.reports, "synthesis")), 3)
        # Observation years must not be confused with publication dates.
        self.assertEqual(select_reports("2026년 태양광 생산량", self.reports, "fact")[0]["report_id"], "new-solar")

    def make_db(self):
        with patch.object(app, "OpenAIEmbeddings", return_value=LocalEmbeddings()), \
             patch.object(app, "VisualElementChunker", return_value=Mock(extract_visual_chunks=Mock(return_value=[]))):
            db, _ = app.build_db([r["path"] for r in self.reports], "test", report_records=self.reports)
        self.assertIsNotNone(db)
        self.addCleanup(app._delete_db, db)
        return db

    def fake_llm(self, **kwargs):
        return FakeMessagesListChatModel(responses=[AIMessage(content="offline grounded response",
                usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})],
                                        callbacks=kwargs.get("callbacks"))

    def test_policy_comparison_keeps_recommendation_section_despite_biased_ranking(self):
        overview = Document(page_content="general overview", metadata={"source_filename": "solar.pdf"})
        advice = Document(page_content="specific recommendations", metadata={
            "source_filename": "solar.pdf", "section": "4. 시사점"})
        db = Mock()
        db.max_marginal_relevance_search.return_value = [overview]
        db.similarity_search.return_value = [overview]
        db.get.return_value = {"documents": [advice.page_content], "metadatas": [advice.metadata]}
        candidates = app._add_file_candidates(db, [overview], "정책 과제 비교", ["solar.pdf"])
        picked = app._cover_files([overview], candidates, ["solar.pdf"], "정책 과제 비교")
        self.assertIn(advice, picked)
        self.assertEqual(picked[0], advice)

    def test_new_upload_is_classified_searchable_and_survives_manifest_refresh(self):
        metadata = dict(title="New solar outlook", published_date="2028-03-01",
                        topics=["solar"], summary="A new solar report.")
        with patch.object(app, "OpenAIEmbeddings", return_value=LocalEmbeddings()), \
             patch.object(app, "fitz", None), \
             patch.object(app, "_extract_report_metadata", return_value=metadata) as extract, \
             patch.object(app, "ReportLibrary", return_value=self.catalog):
            db, message = app.build_db(self.root / "solar.pdf", "test", analyze_metadata=True)
        self.addCleanup(app._delete_db, db)
        extract.assert_called_once()
        self.assertIn("자동 분류 완료", message)
        self.assertEqual(db._report_catalog[0]["topics"], ["solar"])
        self.assertEqual(db.get(include=["metadatas"])["metadatas"][0]["report_summary"], metadata["summary"])
        (self.root / "solar.pdf").unlink()
        reports = self.catalog.sync(self.manifest)
        uploaded = next(r for r in reports if r["title"] == metadata["title"])
        self.assertTrue(uploaded["available"])
        self.assertIn(uploaded, self.catalog.search("solar"))
        self.assertEqual(app.select_reports("solar 보고서 찾아줘", db._report_catalog, "documents")[0]["title"], metadata["title"])

    def test_classification_failure_keeps_pdf_and_discloses_partial_status(self):
        with patch.object(app, "OpenAIEmbeddings", return_value=LocalEmbeddings()), \
             patch.object(app, "fitz", None), \
             patch.object(app, "_extract_report_metadata", side_effect=ValueError("bad metadata")), \
             patch.object(app, "ReportLibrary") as library:
            db, message = app.build_db(self.root / "solar.pdf", "test", analyze_metadata=True)
        self.addCleanup(app._delete_db, db)
        self.assertIsNotNone(db)
        self.assertIn("본문 사용 가능", message)
        library.assert_not_called()
        self.assertEqual(db._report_catalog[0]["topics"], [])

    def test_structured_metadata_extraction_validates_date_and_limits_excerpt(self):
        parsed = app.ReportMetadata(title="새 보고서", published_date="2028-01-02",
                                    topics=["에너지"], summary="문서 요약")
        with patch.object(app, "ChatOpenAI") as model:
            invoke = model.return_value.with_structured_output.return_value.invoke
            invoke.return_value = parsed
            data = app._extract_report_metadata([Document(page_content="a" * 6000)] * 5, "test")
            self.assertEqual(data["topics"], ["에너지"])
            self.assertLessEqual(len(invoke.call_args.args[0][1][1]), 12004)
            invoke.return_value = parsed.model_copy(update={"published_date": "2028-02-31"})
            with self.assertRaises(ValueError):
                app._extract_report_metadata([Document(page_content="text")], "test")

    def test_index_and_catalog_answer_use_titles_dates_ids_without_provider(self):
        db = self.make_db()
        metadata = db.get(include=["metadatas"])["metadatas"]
        self.assertEqual({r["report_id"] for r in metadata}, {r["report_id"] for r in self.reports})
        self.assertTrue(all("published_date" in r and "element_type" in r for r in metadata))
        with patch.object(app, "ChatOpenAI") as llm:
            result = list(app.answer("2028년 2월 태양광 보고서 찾아줘", "test", db))[-1]
        llm.assert_not_called()
        self.assertIn("하늘 태양광 보고서", result)
        self.assertIn("2028-02-15", result)
        self.assertNotIn("별빛 물류", result)

    def test_comparison_retrieves_each_selected_report_and_excludes_unrelated(self):
        db = self.make_db()
        # Simulate a globally biased retriever. Per-report retrieval must recover the other report.
        retrieval = Mock(invoke=Mock(return_value=db.similarity_search("apple", k=1)))
        with patch.object(app, "ChatOpenAI", side_effect=self.fake_llm), \
             patch.object(app.MultiQueryRetriever, "from_llm", return_value=retrieval):
            answer = list(app.answer("태양광과 물류를 비교해줘", "test", db))[-1]
        self.assertIn("하늘 태양광 보고서", answer)
        self.assertIn("별빛 물류 보고서", answer)
        self.assertNotIn("새로운 농업 보고서", answer)
        records = [json.loads(line) for line in (self.root / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
        coverage = next(r for r in records if r["event"] == "evidence_coverage")
        self.assertEqual(coverage["missing"], [])
        self.assertEqual(set(coverage["covered"]), {"solar.pdf", "freight.pdf"})

    def test_missing_report_evidence_is_explicit(self):
        db = self.make_db()
        first = db.similarity_search("apple", k=1)
        with patch.object(app, "ChatOpenAI", side_effect=self.fake_llm), \
             patch.object(app.MultiQueryRetriever, "from_llm", return_value=Mock(invoke=Mock(return_value=first))), \
             patch.object(app, "_add_file_candidates", side_effect=lambda db, docs, query, files: docs):
            answer = list(app.answer("태양광과 물류를 비교해줘", "test", db))[-1]
        self.assertIn("확인 한계", answer)
        self.assertIn("freight.pdf", answer)

    def test_visual_content_and_document_coverage_survive_context_budget(self):
        docs = [Document(page_content="관측치 3%", metadata={"source_filename": "a.pdf", "report_id": "a", "page": 0,
                                                           "page_text": "x" * 15000}),
                Document(page_content="[원본 표] 2028 전망치 7%", metadata={"source_filename": "b.pdf", "report_id": "b", "page": 0,
                                                                       "element_type": "table", "page_text": "unrelated page"}),
                Document(page_content="[차트] 2028 분석값 9%", metadata={"source_filename": "b.pdf", "report_id": "b", "page": 0,
                                                                   "element_type": "image", "page_text": "unrelated page"})]
        expanded = app._expand_to_pages(docs)
        self.assertEqual({d.metadata["source_filename"] for d in expanded}, {"a.pdf", "b.pdf"})
        text = app._format_docs(expanded)
        self.assertIn("전망치 7%", text)
        self.assertIn("분석값 9%", text)
        self.assertLessEqual(sum(len(d.page_content) for d in expanded), app.MAX_CONTEXT_CHARS)
        self.assertTrue(any(d.metadata["context_truncated"] for d in expanded))

    def test_ui_search_and_missing_pdf_do_not_use_api(self):
        request = SimpleNamespace(session_hash="s73")
        with patch.object(app, "_library_records", side_effect=self.catalog.list_reports), \
             patch.object(app, "build_db") as build:
            result, choices = app.search_library("태양광")
            self.assertIn("하늘 태양광", result)
            self.assertEqual(choices["value"], ["new-solar"])
            (self.root / "solar.pdf").unlink()
            db, message, paths = app.register_library(["new-solar"], request)
        self.assertIsNone(db)
        self.assertIn("PDF가 없습니다", message)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
