"""Offline checks for #49: real JSONL, real callbacks, mocked provider boundaries."""
import json
import tempfile
import unittest
import uuid
from contextvars import Context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import openai

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import app
import request_log
from test_app import LocalEmbeddings, write_pdf


class RequestLogTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "requests.jsonl"
        self.log_patch = patch.object(app, "LOG_PATH", self.path)
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)
        self.db = Mock(_collection_name="pdf_test")
        app._answer_cache.clear()

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def fake_llm(self, **kwargs):
        return FakeMessagesListChatModel(
            responses=[AIMessage(content="offline answer", usage_metadata={
                "input_tokens": 7, "output_tokens": 3, "total_tokens": 10})],
            callbacks=kwargs["callbacks"],
        )

    def test_success_metadata_tokens_cache_and_no_results(self):
        doc = Document(page_content="PRIVATE DOCUMENT BODY", metadata={
            "source_filename": "report.pdf", "page": 2, "chunk_id": "chunk-1"})
        retriever = Mock()
        retriever.invoke.return_value = [doc]
        with patch.object(app, "ChatOpenAI", side_effect=self.fake_llm), \
             patch.object(app.MultiQueryRetriever, "from_llm", return_value=retriever):
            self.assertIn("offline answer", list(app.answer("q", "sk-test-secret", self.db))[-1])
            self.assertIn("offline answer", list(app.answer("q", "sk-test-secret", self.db))[-1])
            retriever.invoke.return_value = []
            list(app.answer("missing", "sk-test-secret", self.db))
        rows = self.records()
        finished = [r for r in rows if r["event"] == "request_finished"]
        self.assertEqual([r["status"] for r in finished], ["success", "cached", "no_results"])
        self.assertEqual(len({r["request_id"] for r in finished}), 3)
        self.assertEqual(finished[0]["token_usage"]["total_tokens"], 10)
        self.assertEqual(finished[1]["token_usage"]["total_tokens"], 0)
        usage = [r for r in rows if r["event"] == "token_usage"]
        self.assertTrue(any(r["stage"] == "answer_generation" and r["model"] == app.MODEL_NAME
                            and r["total_tokens"] == 10 for r in usage))
        self.assertTrue(any(r["usage_status"] == "unavailable" and r["total_tokens"] is None for r in usage))
        found = next(r for r in rows if r["event"] == "retrieval_result" and r["documents"])
        self.assertEqual(found["documents"], [{"filename": "report.pdf", "page": "3", "chunk_id": "chunk-1"}])
        self.assertEqual(found["collection"], "pdf_test")
        self.assertNotIn("PRIVATE DOCUMENT BODY", self.path.read_text(encoding="utf-8"))

    def test_retrieval_error_and_rerank_fallback_are_recorded_and_redacted(self):
        retriever = Mock()
        retriever.invoke.side_effect = RuntimeError("Bearer private-token sk-test-secret")
        with patch.object(app, "ChatOpenAI", side_effect=self.fake_llm), \
             patch.object(app.MultiQueryRetriever, "from_llm", return_value=retriever):
            with self.assertRaises(RuntimeError):
                list(app.answer("q", "sk-test-secret", self.db))
        self.assertEqual(self.records()[-1]["status"], "error")
        self.assertTrue(any(r["event"] == "error" and r["stage"] == "retrieval" and r["traceback"]
                            for r in self.records()))
        docs = [Document(page_content="x") for _ in range(app.TOP_K + 1)]
        with request_log.request(self.path, "qa"):
            with patch.object(app.LLMListwiseRerank, "from_llm", side_effect=RuntimeError("rerank failed")):
                self.assertEqual(app._rerank(None, docs, "q"), docs[:app.TOP_K])
        self.assertTrue(any(r["event"] == "rerank_fallback" for r in self.records()))
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn("sk-test-secret", text)
        self.assertNotIn("private-token", text)

    def test_pdf_partial_metadata_and_embedding_cleanup_errors(self):
        good = Path(self.folder.name) / "good.pdf"
        broken = Path(self.folder.name) / "broken.pdf"
        write_pdf(good, ["apple orchard"])
        broken.write_text("not a pdf")
        with patch.object(app, "OpenAIEmbeddings", return_value=LocalEmbeddings()):
            db, _ = app.build_db([str(good), str(broken)], "test")
        self.addCleanup(db.delete_collection)
        doc = db.similarity_search("apple", k=1)[0]
        self.assertTrue(doc.metadata["chunk_id"])
        self.assertEqual(db.get()["ids"], [doc.metadata["chunk_id"]])
        self.assertEqual(self.records()[-1]["status"], "partial")
        failed = Mock(_collection_name="pdf_failed")
        failed.add_documents.side_effect = RuntimeError("embed sk-secret-value")
        failed.delete_collection.side_effect = RuntimeError("cleanup sk-secret-value")
        with patch.object(app, "OpenAIEmbeddings", return_value=LocalEmbeddings()), \
             patch.object(app, "Chroma", return_value=failed), self.assertLogs("app", level="ERROR") as console:
            result, _ = app.build_db(str(good), "sk-secret-value")
        self.assertIsNone(result)
        self.assertTrue(any(r.get("cleanup") for r in self.records()))
        self.assertEqual(self.records()[-1]["status"], "error")
        self.assertNotIn("sk-secret-value", self.path.read_text(encoding="utf-8") + str(console.output))

    def test_missing_usage_file_failure_and_cross_context_streaming(self):
        with request_log.request(self.path, "qa") as trace:
            callback = request_log.TokenUsageCallback(trace, "fake-model")
            callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]]),
                                run_id=uuid.uuid4())
            trace.emit("metadata", api_key="not-a-standard-key")
        self.assertEqual(self.records()[-2]["api_key"], "[REDACTED]")
        self.assertEqual(self.records()[-1]["usage_status"], "partial")
        with patch.object(Path, "open", side_effect=OSError("disk full")), \
             self.assertLogs("request_log", level="ERROR") as console:
            trace.emit("test")
        self.assertIn("write failed", str(console.output))
        retriever = Mock()
        retriever.invoke.return_value = [Document(page_content="x", metadata={"page": 0})]
        with patch.object(app, "ChatOpenAI", side_effect=self.fake_llm), \
             patch.object(app.MultiQueryRetriever, "from_llm", return_value=retriever):
            iterator = app.answer("cross-context", "test", self.db)
            while True:
                try:
                    Context().run(next, iterator)
                except StopIteration:
                    break
        self.assertEqual(self.records()[-1]["status"], "success")
        self.assertIsNone(request_log.current())

    def test_ui_validation_is_logged_without_calling_provider(self):
        request = SimpleNamespace(session_hash="logging-test")
        with patch.object(app, "ChatOpenAI") as provider:
            list(app.ask_question("", [], None, request))
            list(app.ask_question("q", [], None, request))
        provider.assert_not_called()
        self.assertEqual([r["reason"] for r in self.records() if r["event"] == "validation_failed"],
                         ["empty_question", "missing_or_replaced_db"])
        app._session_locks.pop(request.session_hash, None)

    def test_real_retrieval_chain_counts_rewrite_query_and_answer_usage(self):
        pdf = Path(self.folder.name) / "report.pdf"
        write_pdf(pdf, ["apple orchard"])
        with patch.object(app, "OpenAIEmbeddings", return_value=LocalEmbeddings()):
            db, _ = app.build_db(str(pdf), "test")
        self.addCleanup(db.delete_collection)
        with patch.object(app, "ChatOpenAI", side_effect=self.fake_llm):
            result = list(app.answer("more", "test", db,
                                     [{"role": "user", "content": "apple"}]))
        self.assertIn("offline answer", result[-1])
        usage = [r for r in self.records() if r["event"] == "token_usage" and r["usage_status"] == "available"]
        self.assertEqual({r["stage"] for r in usage}, {"question_rewrite", "retrieval", "answer_generation"})
        self.assertEqual(self.records()[-1]["token_usage"]["total_tokens"], 30)

    def test_api_failure_preserves_user_error_and_logs_original_exception(self):
        retriever = Mock()
        retriever.invoke.side_effect = openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        with patch.object(app, "ChatOpenAI", side_effect=self.fake_llm), \
             patch.object(app.MultiQueryRetriever, "from_llm", return_value=retriever):
            with self.assertRaises(app.gr.Error):
                list(app.answer("q", "test", self.db))
        self.assertEqual(self.records()[-1]["status"], "error")
        self.assertTrue(any(r.get("error_type") == "APIConnectionError" for r in self.records()))


if __name__ == "__main__":
    unittest.main()
