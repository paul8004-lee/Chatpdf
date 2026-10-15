"""Run: .venv/Scripts/python.exe -m unittest test_app -v (no API calls)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import app


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.lower().count(word)) for word in ("apple", "banana", "cherry")] + [0.01]


def write_pdf(path, pages):
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


class PDFWorkflowTest(unittest.TestCase):
    def test_embedding_and_cleanup_failure_returns_status(self):
        db = Mock()
        db.add_documents.side_effect = RuntimeError("embedding failed")
        db.delete_collection.side_effect = RuntimeError("cleanup failed")
        with tempfile.TemporaryDirectory() as folder:
            pdf_path = Path(folder) / "failed.pdf"
            write_pdf(pdf_path, ["apple orchard"])
            with patch.object(app, "OpenAIEmbeddings", return_value=LocalEmbeddings()), \
                 patch.object(app, "Chroma", return_value=db), \
                 self.assertLogs("app", level="ERROR") as logs:
                result, message = app.build_db(str(pdf_path), "test")
        self.assertIsNone(result)
        self.assertIn("처리 실패", message)
        self.assertIn("failed.pdf", message)
        db.delete_collection.assert_called_once_with()
        self.assertIn("cleanup failed", "\n".join(logs.output))

    def test_multiple_isolation_single_and_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            app, "OpenAIEmbeddings", return_value=LocalEmbeddings()
        ):
            first, second, replacement, broken = [Path(folder) / name for name in
                                                  ("first.pdf", "second.pdf", "new.pdf", "broken.pdf")]
            write_pdf(first, ["apple orchard", "apple harvest"])
            write_pdf(second, ["banana plantation"])
            write_pdf(replacement, ["cherry garden"])
            broken.write_text("invalid PDF")
            db, message = app.build_db([str(first), str(second)], "test")
            self.assertIn("2개 파일 · 3페이지 · 3개 청크", message)
            for word, name in (("apple", first.name), ("banana", second.name)):
                doc = db.similarity_search(word, k=1)[0]
                self.assertEqual(doc.metadata["source_filename"], name)
                self.assertIn("page", doc.metadata)
                self.assertIn("source", doc.metadata)
            new_db, message = app.build_db([str(replacement)], "test")
            self.assertIn("1개 파일 · 1페이지 · 1개 청크", message)
            self.assertEqual(set(new_db.get()["documents"]), {"cherry garden"})
            self.assertNotEqual(db._collection.name, new_db._collection.name)
            db.delete_collection()
            new_db.delete_collection()
            # #42: 손상된 파일은 제외하고 정상 파일만 등록
            partial_db, message = app.build_db([str(first), str(broken)], "test")
            self.assertIn("broken.pdf", message)
            self.assertIsNotNone(partial_db)
            partial_db.delete_collection()
            failed_db, message = app.build_db([str(broken)], "test")
            self.assertIn("broken.pdf", message)
            self.assertIsNone(failed_db)
            # A failure while embedding the second file must remove partial chunks.
            original_add = app.Chroma.add_documents
            partial_collections = []
            def fail_second(db, docs, **kwargs):
                if docs[0].metadata["source_filename"] == second.name:
                    partial_collections.append(db)
                    raise RuntimeError("embedding failed")
                return original_add(db, docs, **kwargs)
            with patch.object(app.Chroma, "add_documents", fail_second):
                db, message = app.build_db([str(first), str(second)], "test")
            self.assertIsNone(db)
            self.assertIn("second.pdf", message)
            self.assertIsNone(partial_collections[0]._chroma_collection)
            self.assertIsNone(app.build_db([str(first)], "")[0])
            self.assertIsNone(app.build_db([], "test")[0])
            write_pdf(replacement, [""])
            self.assertIn("new.pdf", app.build_db([str(replacement)], "test")[1])
            write_pdf(replacement, ["cherry garden"])
            single_db, message = app.build_db(str(replacement), "test")
            self.assertIn("1개 파일", message)
            single_db.delete_collection()


class PartialFailureTest(unittest.TestCase):
    """#42: PDF 1개가 실패해도 나머지 PDF는 등록되어야 함"""

    def test_failed_files_are_skipped_with_reason(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            app, "OpenAIEmbeddings", return_value=LocalEmbeddings()
        ):
            good, image_only, broken = [Path(folder) / name for name in
                                        ("good.pdf", "image_only.pdf", "broken.pdf")]
            write_pdf(good, ["apple orchard", "apple harvest"])
            write_pdf(image_only, [""])
            broken.write_text("invalid PDF")
            db, message = app.build_db([str(image_only), str(good), str(broken)], "test")
            self.assertIsNotNone(db)
            self.assertIn("1개 파일 · 2페이지", message)
            self.assertIn("제외된 파일 2개", message)
            self.assertIn("image_only.pdf: 텍스트가 없는 PDF", message)
            self.assertIn("broken.pdf: PDF를 읽을 수 없음", message)
            self.assertEqual({d.metadata["source_filename"] for d in db.similarity_search("apple", k=5)},
                             {"good.pdf"})
            db.delete_collection()

    def test_all_failed_returns_none_with_every_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            image_only, broken = Path(folder) / "image_only.pdf", Path(folder) / "broken.pdf"
            write_pdf(image_only, [""])
            broken.write_text("invalid PDF")
            db, message = app.build_db([str(image_only), str(broken)], "test")
        self.assertIsNone(db)
        self.assertIn("등록할 수 있는 PDF가 없습니다", message)
        self.assertIn("image_only.pdf", message)
        self.assertIn("broken.pdf", message)

    def test_embedding_failure_still_fails_everything(self):
        with tempfile.TemporaryDirectory() as folder:
            good = Path(folder) / "good.pdf"
            write_pdf(good, ["apple orchard"])
            with patch.object(app, "OpenAIEmbeddings", side_effect=RuntimeError("bad key")):
                db, message = app.build_db([str(good)], "test")
        self.assertIsNone(db)
        self.assertIn("API 키/연결", message)


class SourceFormatTest(unittest.TestCase):
    """#28: 출처는 원본 파일명(source_filename)과 실제 PDF 페이지(page + 1) 기준"""

    def doc(self, **meta):
        return Mock(metadata=meta, page_content="text")

    def test_original_filename_and_real_page_win(self):
        d = self.doc(source_filename="original.pdf", source="/tmp/renamed.pdf", page=0, page_label="iv")
        self.assertEqual(app._page_label(d.metadata), "1")
        self.assertIn("original.pdf (p.1)", app._format_sources([d]))
        self.assertNotIn("renamed.pdf", app._format_sources([d]))
        self.assertIn("[p.1]", app._format_docs([d]))

    def test_fallbacks(self):
        windows = self.doc(source=r"C:\data\report.pdf", page=4)
        self.assertIn("report.pdf (p.5)", app._format_sources([windows]))
        self.assertEqual(app._page_label({"page_label": "iv"}), "iv")
        self.assertEqual(app._page_label({}), "?")
        self.assertIn("문서 (p.?)", app._format_sources([self.doc()]))

    def test_multiple_files_sorted_and_deduplicated(self):
        docs = [self.doc(source_filename="a.pdf", page=9), self.doc(source_filename="b.pdf", page=0),
                self.doc(source_filename="a.pdf", page=1), self.doc(source_filename="a.pdf", page=9)]
        self.assertIn("a.pdf (p.2, 10) / b.pdf (p.1)", app._format_sources(docs))


class PdfSelectionTest(unittest.TestCase):
    """#41: PDF를 나눠서 선택해도 이전 선택이 유지되어야 함"""

    def test_selecting_again_appends(self):
        paths, _ = app.select_pdf(["/tmp/a/first.pdf"], None)
        paths, status = app.select_pdf(["/tmp/b/second.pdf", "/tmp/c/third.pdf"], paths)
        self.assertEqual(paths, ["/tmp/a/first.pdf", "/tmp/b/second.pdf", "/tmp/c/third.pdf"])
        self.assertIn("총 3개", status)
        for name in ("first.pdf", "second.pdf", "third.pdf"):
            self.assertIn(name, status)

    def test_same_name_is_replaced_not_duplicated(self):
        paths, _ = app.select_pdf(["/tmp/a/report.pdf", "/tmp/a/other.pdf"], None)
        paths, status = app.select_pdf(["/tmp/z/report.pdf"], paths)
        self.assertEqual(paths, ["/tmp/z/report.pdf", "/tmp/a/other.pdf"])
        self.assertIn("총 2개", status)

    def test_single_file_and_empty_selection(self):
        paths, _ = app.select_pdf("/tmp/a/one.pdf", None)
        self.assertEqual(paths, ["/tmp/a/one.pdf"])
        same, _ = app.select_pdf(None, paths)
        self.assertEqual(same, paths)

    def test_clear_selection(self):
        paths, status = app.clear_selection()
        self.assertIsNone(paths)
        self.assertIn("선택한 PDF가 없습니다", status)


class PageContextTest(unittest.TestCase):
    """#37: 고른 조각을 그 페이지 전체(레이아웃 보존 텍스트)로 넓혀 표·목록이 잘리지 않게 함"""

    def doc(self, text, page_text=None, name="a.pdf", page=0):
        meta = {"source_filename": name, "page": page}
        if page_text is not None:
            meta["page_text"] = page_text
        return app.Document(page_content=text, metadata=meta)

    def test_build_db_stores_page_text_for_every_chunk(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            app, "OpenAIEmbeddings", return_value=LocalEmbeddings()
        ):
            pdf = Path(folder) / "table.pdf"
            write_pdf(pdf, ["apple orchard", "banana plantation"])
            db, _ = app.build_db([str(pdf)], "test")
            stored = db.get(include=["metadatas"])["metadatas"]
            db.delete_collection()
        self.assertEqual({m["page"]: m["page_text"] for m in stored},
                         {0: "apple orchard", 1: "banana plantation"})

    def test_expands_to_page_once_and_keeps_order(self):
        docs = [self.doc("row 3", "header\nrow 1\nrow 2\nrow 3", page=4),
                self.doc("row 1", "header\nrow 1\nrow 2\nrow 3", page=4),
                self.doc("other", "other page", page=7)]
        expanded = app._expand_to_pages(docs)
        self.assertEqual([d.page_content for d in expanded], ["header\nrow 1\nrow 2\nrow 3", "other page"])
        self.assertEqual([d.metadata["page"] for d in expanded], [4, 7])
        self.assertIn("a.pdf (p.5, 8)", app._format_sources(expanded))

    def test_same_page_number_in_different_files_is_kept(self):
        docs = [self.doc("x", "page of a", name="a.pdf"), self.doc("y", "page of b", name="b.pdf")]
        self.assertEqual(len(app._expand_to_pages(docs)), 2)

    def test_old_db_without_page_text_and_size_limit(self):
        self.assertEqual(app._expand_to_pages([self.doc("only chunk")])[0].page_content, "only chunk")
        big = "x" * (app.MAX_CONTEXT_CHARS - 100)   # 첫 페이지 후 남은 여유로는 다음 페이지 전체가 안 들어감
        docs = [self.doc("c1", big, page=0), self.doc("c2", big, page=1)]
        self.assertEqual([d.page_content for d in app._expand_to_pages(docs)], [big, "c2"])


class QuestionLockMessageTest(unittest.TestCase):
    """#55: 질문창이 막혀 있을 때 그 이유를 안내해야 함"""

    def test_initial_question_box_explains_why_locked(self):
        box = next(b for b in app.demo.blocks.values() if getattr(b, "elem_id", None) == "question-box")
        self.assertFalse(box.interactive)
        self.assertEqual(box.placeholder, app.QUESTION_NO_PDF)

    def test_processing_and_result_placeholders(self):
        request = Mock(session_hash="s55")
        started = app.start_register(request)
        self.assertEqual(started[2]["placeholder"], app.QUESTION_PROCESSING)
        self.assertFalse(started[2]["interactive"])
        failed = app.finish_register(None, "❌ 처리 실패", [])
        self.assertEqual(failed[0], "❌ 처리 실패")              # 결과 문구와 잠금 상태가 함께 바뀜
        self.assertEqual(failed[1]["placeholder"], app.QUESTION_NOT_REGISTERED)
        self.assertFalse(failed[1]["interactive"])
        self.assertFalse(failed[2]["interactive"])
        ready = app.finish_register(Mock(), "✅ 처리 완료", ["/tmp/x.pdf"])
        self.assertEqual(ready[0], "✅ 처리 완료")
        self.assertEqual(ready[1]["placeholder"], app.QUESTION_READY)
        self.assertTrue(ready[1]["interactive"])
        self.assertTrue(ready[2]["interactive"])

    def test_failed_registration_message_says_question_needs_pdf(self):
        request = Mock(session_hash="s55")
        with patch.object(app, "build_db", return_value=(None, "❌ 처리 실패: 등록할 수 있는 PDF가 없습니다.")):
            db, message = app.register_pdf(["/tmp/x.pdf"], request)
        self.assertIsNone(db)
        self.assertIn("처리 실패", message)
        self.assertIn("등록에 성공해야 질문할 수 있습니다", message)

    def test_every_failure_path_explains_question_lock(self):
        request = Mock(session_hash="s55")
        self.assertIn("등록에 성공해야 질문할 수 있습니다", app.register_pdf(None, request)[1])
        with patch.object(app, "OPENAI_API_KEY", None):
            self.assertIn("등록에 성공해야 질문할 수 있습니다", app.register_pdf(["/tmp/x.pdf"], request)[1])


class AnswerCacheTest(unittest.TestCase):
    """#24: 새 PDF DB에서 이전 PDF의 캐시 답변이 반환되면 안 됨"""

    def setUp(self):
        app._answer_cache.clear()

    def test_cache_survives_id_reuse(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            app, "OpenAIEmbeddings", return_value=LocalEmbeddings()
        ):
            old_pdf, new_pdf = Path(folder) / "old.pdf", Path(folder) / "new.pdf"
            write_pdf(old_pdf, ["apple orchard"])
            write_pdf(new_pdf, ["banana plantation"])
            old_db, _ = app.build_db([str(old_pdf)], "test")
            new_db, _ = app.build_db([str(new_pdf)], "test")
            # 이전 DB가 해제된 뒤 새 DB가 같은 id()를 받은 상황을 재현
            with patch.object(app, "id", create=True, side_effect=lambda obj: 1):
                app._cache_put(old_db, "q", "old answer")
                self.assertIsNone(app._cache_get(new_db, "q"))
                self.assertEqual(app._cache_get(old_db, "q"), "old answer")
            app._delete_db(old_db)
            app._delete_db(new_db)

    def test_delete_db_clears_its_cache_only(self):
        first, second = Mock(_collection_name="pdf_a"), Mock(_collection_name="pdf_b")
        app._cache_put(first, "q", "A")
        app._cache_put(second, "q", "B")
        app._delete_db(first)
        first.delete_collection.assert_called_once_with()
        self.assertIsNone(app._cache_get(first, "q"))
        self.assertEqual(app._cache_get(second, "q"), "B")


if __name__ == "__main__":
    unittest.main()
