"""Step 2 검색 부분 테스트 (소용우 담당). API 호출 없음.
실행: python -m unittest test_step2_search -v
- PDF와 Chroma는 실제로 쓰고, 임베딩과 GPT만 가짜로 바꿈
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import app
from test_app import LocalEmbeddings, write_pdf   # Step 1 테스트의 가짜 임베딩·PDF 생성 함수 재사용


def fake_llm(**kwargs):
    """ChatOpenAI 대신 쓰는 가짜 GPT. 항상 같은 글자를 돌려줌."""
    return FakeListChatModel(responses=["apple\nbanana\ncherry"])


class SearchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()                       # 테스트용 임시 폴더
        self.folder = Path(self.tmp.name)
        self.patches = [
            patch.object(app, "OpenAIEmbeddings", return_value=LocalEmbeddings()),  # 임베딩 → 가짜
            patch.object(app, "ChatOpenAI", side_effect=fake_llm),                    # GPT → 가짜
            patch.object(app, "LOG_PATH", self.folder / "qa_log.jsonl"),              # 로그 → 임시 폴더
        ]
        for p in self.patches:
            p.start()
        app._answer_cache.clear()                                       # 테스트마다 캐시 비우기
        self.dbs = []

    def tearDown(self):
        for db in self.dbs:
            db.delete_collection()                                      # 만든 DB 정리
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def make_db(self, name, pages):
        """PDF를 만들어 Step 1 build_db로 DB 생성."""
        path = self.folder / name
        write_pdf(path, pages)
        db, message = app.build_db([str(path)], "test")
        self.assertIsNotNone(db, message)
        self.dbs.append(db)
        return db

    def test_db_key_uses_collection_name(self):
        """캐시 키는 Chroma 컬렉션 이름이고, PDF마다 다르며, DB 삭제 후에도 오류 없이 같은 값."""
        a = self.make_db("a.pdf", ["apple orchard"])
        b = self.make_db("b.pdf", ["banana plantation"])
        key_a = app._db_key(a)
        self.assertEqual(key_a, a._collection.name)
        self.assertNotEqual(key_a, app._db_key(b))
        a.delete_collection()
        self.dbs.remove(a)
        self.assertEqual(app._db_key(a), key_a)

    def test_same_id_does_not_mix_cache(self):
        """[이슈 재현] 두 DB의 id가 같아져도(메모리 재사용) 이전 PDF 답변이 나오지 않음."""
        a = self.make_db("a.pdf", ["apple orchard"])
        b = self.make_db("b.pdf", ["banana plantation"])
        with patch("builtins.id", lambda obj: 12345):                 # 모든 객체 id를 같게 만들어 재사용 상황 재현
            app._cache_put(a, "q", "A 답변")
            self.assertEqual(app._cache_get(a, "q"), "A 답변")
            self.assertIsNone(app._cache_get(b, "q"))                  # B에서는 A 답변이 나오면 안 됨

    def test_clear_answer_cache(self):
        """PDF 교체 시 이전 PDF 캐시만 지우고, 다른 PDF 캐시는 남김. 인자 없으면 전체 삭제."""
        a = self.make_db("a.pdf", ["apple orchard"])
        b = self.make_db("b.pdf", ["banana plantation"])
        app._cache_put(a, "q1", "A1")
        app._cache_put(a, "q2", "A2")
        app._cache_put(b, "q1", "B1")
        self.assertEqual(app.clear_answer_cache(a), 2)
        self.assertIsNone(app._cache_get(a, "q1"))
        self.assertEqual(app._cache_get(b, "q1"), "B1")
        self.assertEqual(app.clear_answer_cache(), 1)
        self.assertEqual(len(app._answer_cache), 0)

    def test_cache_hit_same_pdf(self):
        """같은 PDF에 같은 질문 두 번 → 두 번째는 캐시에서 한 번에 나옴."""
        db = self.make_db("a.pdf", ["apple orchard", "apple harvest"])
        first = list(app.answer("apple?", "test", db))
        second = list(app.answer("apple?", "test", db))
        self.assertGreater(len(first), 1)                               # 처음엔 조금씩 여러 번 출력
        self.assertEqual(second, [first[-1]])                           # 두 번째는 저장된 최종 답변 1번

    def test_cache_isolated_between_pdfs(self):
        """PDF A에 질문 → PDF B로 교체 → 같은 질문 → B 기준 출처가 나와야 함."""
        a = self.make_db("a.pdf", ["apple orchard"])
        out_a = list(app.answer("fruit?", "test", a))[-1]
        b = self.make_db("b.pdf", ["banana plantation"])
        out_b = list(app.answer("fruit?", "test", b))[-1]
        self.assertIn("a.pdf", out_a)
        self.assertIn("b.pdf", out_b)
        self.assertNotIn("a.pdf", out_b)

    def test_retrieve_every_mode(self):
        """4가지 검색 방식 모두 k개 이하를 돌려주고, 재정렬 여부를 올바르게 표시."""
        db = self.make_db("a.pdf", ["apple orchard", "apple harvest", "banana", "cherry"])
        llm = fake_llm()
        expected_flag = {"similarity": None, "mmr": None, "multiquery": None}
        for mode in app.SEARCH_MODES:
            docs, rerank_ok = app._retrieve(db, llm, "apple", mode=mode, k=2)
            self.assertLessEqual(len(docs), 2, mode)
            self.assertGreater(len(docs), 0, mode)
            if mode in expected_flag:
                self.assertIsNone(rerank_ok, mode)
            else:
                self.assertIsInstance(rerank_ok, bool, mode)            # 재정렬 방식은 True/False

    def test_rerank_failure_is_recorded(self):
        """재정렬이 실패해도 답변은 계속되고, 로그에 rerank_ok=false가 남음."""
        db = self.make_db("a.pdf", ["apple orchard", "apple harvest", "banana", "cherry", "grape"])
        with patch.object(app, "TOP_K", 2), patch.object(app, "SEARCH_MODE", "multiquery_rerank"), \
                patch.object(app.LLMListwiseRerank, "from_llm", side_effect=RuntimeError("boom")):
            out = list(app.answer("apple?", "test", db))
        self.assertIn("**참고**", out[-1])                               # 답변은 정상 완료
        log = [json.loads(line) for line in app.LOG_PATH.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(log[-1]["mode"], "multiquery_rerank")
        self.assertIs(log[-1]["rerank_ok"], False)

    def test_unknown_mode_raises(self):
        """없는 검색 방식 이름을 쓰면 바로 오류로 알려줌."""
        db = self.make_db("a.pdf", ["apple"])
        with self.assertRaises(ValueError):
            app._retrieve(db, fake_llm(), "apple", mode="wrong")


if __name__ == "__main__":
    unittest.main()
