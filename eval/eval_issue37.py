"""#37 답변 품질 평가 (표 비교 · 목록 완전성 · 공통성 판단 · 거절)

무엇을 하나요?
  eval/issue37_questions.jsonl의 질문을 PDF별로 실제 앱의 answer()로 답하게 하고,
  GPT 채점기가 정답 요약(gold)과 비교해 ✅(1) / △(0.5) / ❌(0)으로 채점합니다.
  질문마다 응답 시간, 토큰 수, 비용(USD)도 함께 기록합니다.

실행 방법 (프로젝트 폴더에서, .venv 켠 상태):
  python eval/eval_issue37.py --label before
  python eval/eval_issue37.py --label after

준비물:
  - .env의 OPENAI_API_KEY
  - PDF: samples/test.pdf, samples/unseen/hri_26-14_price.pdf, samples/unseen/hri_26-15_nk_china_trade.pdf
    (출처는 docs/sources.md, PDF는 저장소에 올리지 않음)

결과:
  - eval/results_issue37_<label>.csv (질문별 답변·점수·근거·시간·토큰·비용)
  - 화면에 유형별·세트별 요약 출력
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import app

PDF_DIRS = [ROOT / "samples", ROOT / "samples" / "unseen"]
JUDGE_MODEL = "gpt-4o"
SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
MARKS = {"correct": "✅", "partial": "△", "wrong": "❌"}

JUDGE_PROMPT = """당신은 문서 기반 Q&A 챗봇의 답변을 채점합니다.
질문, 정답 요약(gold), 챗봇 답변을 보고 판정하세요.

- correct: gold의 핵심 내용을 모두 맞게 담음. 목록 질문은 항목을 빠짐없이, 공통성 질문은 공통이 아닌 것을 공통이라 하지 않음.
  '문서에 없음' 유형은 확인할 수 없다고 답하고 다른 값을 지어내지 않음.
- partial: 일부만 맞음(목록 일부 누락, 수치 일부 누락, 해석은 맞지만 근거 수치 부족 등).
- wrong: 거절하면 안 되는데 거절, 틀린 수치·명칭, 문서에 없는 항목을 만들어냄, 잘못된 일반화, 문서에 없는 내용을 지어냄.
답변 끝의 '참고: 파일 (p.N)' 표시는 채점 대상이 아닙니다.

[질문 유형] {qtype}
[질문] {question}
[정답 요약] {gold}
[챗봇 답변]
{answer}"""


class Verdict(BaseModel):
    verdict: str = Field(description="correct / partial / wrong 중 하나")
    reason: str = Field(description="판정 이유 한두 문장")


def find_pdf(name):
    for folder in PDF_DIRS:
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {name} (samples/ 또는 samples/unseen/)")


def run_answer(question, key, db):
    app._answer_cache.clear()                     # 캐시 없이 매번 실제로 답변
    start = time.time()
    with get_openai_callback() as cb:
        final = ""
        for part in app.answer(question, key, db):
            final = part
    return final, time.time() - start, cb.total_tokens, cb.total_cost


def judge_answer(judge, q, answer, attempts=3):
    """채점기가 가끔 출력을 반복해 길이 제한에 걸리므로 몇 번 다시 시도하고, 그래도 실패하면 wrong 처리."""
    prompt = JUDGE_PROMPT.format(qtype=q["type"], question=q["question"], gold=q["gold"], answer=answer)
    for _ in range(attempts):
        try:
            return judge.invoke(prompt)
        except Exception as error:
            last = error
    return Verdict(verdict="wrong", reason=f"채점 실패: {type(last).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="결과 파일 이름에 붙일 이름 (예: before, after)")
    parser.add_argument("--only", nargs="*", help="특정 질문 id만 실행 (예: Q44 P1)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ["OPENAI_API_KEY"]
    judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0, max_tokens=400,
                       api_key=key).with_structured_output(Verdict)
    questions = [json.loads(line) for line in (ROOT / "eval" / "issue37_questions.jsonl").read_text().splitlines()
                 if line.strip()]
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]

    by_pdf = defaultdict(list)
    for q in questions:
        by_pdf[q["pdf"]].append(q)

    rows = []
    for pdf_name, qs in by_pdf.items():
        db, message = app.build_db([str(find_pdf(pdf_name))], key)
        print(f"\n[{pdf_name}] {message}")
        for q in qs:
            answer, seconds, tokens, cost = run_answer(q["question"], key, db)
            verdict = judge_answer(judge, q, answer)
            mark = verdict.verdict if verdict.verdict in SCORES else "wrong"
            rows.append(dict(id=q["id"], set=q["set"], pdf=pdf_name, type=q["type"], question=q["question"],
                             verdict=MARKS[mark], score=SCORES[mark], reason=verdict.reason,
                             seconds=round(seconds, 1), tokens=tokens, cost_usd=round(cost, 5),
                             answer=answer))
            print(f"  {q['id']:>4} {MARKS[mark]} {seconds:5.1f}s {tokens:>6}tok  {verdict.reason[:70]}")
        db.delete_collection()

    out = ROOT / "eval" / f"results_issue37_{args.label}.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== 요약 ({args.label}) → {out.relative_to(ROOT)}")
    for key_name in ("set", "type"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key_name]].append(r)
        for name, rs in groups.items():
            total = sum(r["score"] for r in rs)
            print(f"  {name:<8} {total:4.1f}/{len(rs)}  "
                  f"평균 {sum(r['seconds'] for r in rs) / len(rs):4.1f}s  "
                  f"평균 {sum(r['tokens'] for r in rs) / len(rs):6.0f}tok  "
                  f"합계 ${sum(r['cost_usd'] for r in rs):.4f}")


if __name__ == "__main__":
    main()
