"""검색 방식 비교 실험 (Step 2 · 소용우)

무엇을 하나요?
  같은 평가 질문으로 검색 방식(SEARCH_MODES)과 k값을 바꿔 가며 검색만 실행하고,
  "정답 페이지를 찾았는지(적중률)"와 "얼마나 걸렸는지(시간)"를 표로 비교합니다.
  답변 생성은 하지 않으므로 비용이 적게 듭니다.

실행 방법 (프로젝트 폴더에서, .venv 켠 상태):
  python eval/compare_retrievers.py 보고서.pdf
  python eval/compare_retrievers.py 보고서1.pdf 보고서2.pdf --k 3 4 6

준비물:
  - .env 파일의 OPENAI_API_KEY (임베딩·멀티쿼리·재정렬에 실제 API를 씀 → 소액 비용 발생)
  - eval/qa_set.jsonl (평가 질문, 형식은 아래 load_questions 참고)

결과:
  - 화면에 비교표 출력
  - eval/results_retriever.csv 저장 (README 표 작성에 사용)
"""
import argparse                                   # 명령줄 옵션(--k 등)을 읽는 도구
import csv                                        # 결과를 CSV(엑셀로 열 수 있는 표) 파일로 저장
import json                                       # 평가 질문 파일(JSON 한 줄씩)을 읽는 도구
import os                                         # 환경변수(OPENAI_API_KEY)를 읽는 도구
import sys                                        # 파이썬이 app.py를 찾을 수 있게 경로를 추가할 때 사용
import time                                       # 검색에 걸린 시간을 재는 도구
from pathlib import Path                          # 파일 경로를 다루는 도구

ROOT = Path(__file__).resolve().parent.parent     # 프로젝트 폴더 (이 파일은 eval/ 안에 있으므로 한 단계 위)
sys.path.insert(0, str(ROOT))                     # 프로젝트 폴더의 app.py를 import할 수 있게 경로 추가

from dotenv import load_dotenv                    # .env 파일을 읽어 환경변수로 등록하는 도구  # noqa: E402
from langchain_openai import ChatOpenAI           # 멀티쿼리·재정렬에 쓸 GPT  # noqa: E402

import app                                        # 우리 앱 (Step 1의 build_db, Step 2의 _retrieve 사용)  # noqa: E402

QA_PATH = ROOT / "eval" / "qa_set.jsonl"          # 평가 질문 파일 위치
OUT_PATH = ROOT / "eval" / "results_retriever.csv"  # 결과 저장 위치


def load_questions(path):
    """평가 질문 파일을 읽음. 한 줄에 질문 하나(JSON).
    예: {"id": 1, "type": "수치", "question": "2026년 성장률 전망은?",
         "source_filename": "보고서.pdf", "answer_pages": [3]}
    - answer_pages: 정답이 있는 페이지 (PDF 뷰어 기준, 1부터)
    - 문서에 없는 질문은 answer_pages를 [] 로 두면 적중률 계산에서 빠짐
    """
    questions = []                                                    # 읽은 질문을 담을 목록
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):  # 한 줄씩 읽기 (줄 번호 포함)
        if not line.strip():                                          # 빈 줄은
            continue                                                  #   건너뜀
        try:
            q = json.loads(line)                                      # JSON 한 줄을 파이썬 사전으로 변환
        except json.JSONDecodeError as e:                             # 형식이 틀리면
            sys.exit(f"[오류] {path.name} {line_no}번째 줄 JSON 형식 오류: {e}")  # 몇 번째 줄인지 알려주고 종료
        if "question" not in q or not isinstance(q.get("answer_pages"), list):  # 필수 항목이 없으면
            sys.exit(f"[오류] {path.name} {line_no}번째 줄에 question 또는 answer_pages(목록)가 없습니다.")
        questions.append(q)                                           # 정상이면 목록에 추가
    return questions


def is_hit(docs, q):
    """검색된 조각 중 하나라도 정답 페이지(같은 파일)에 있으면 True."""
    want_file = q.get("source_filename")                              # 정답 파일 이름 (없으면 파일은 안 따짐)
    for d in docs:                                                    # 검색된 조각을 하나씩 확인
        page = d.metadata.get("page")                                 # 0부터 시작하는 페이지 번호
        if not isinstance(page, int):                                 # 페이지 정보가 없으면
            continue                                                  #   건너뜀
        same_file = not want_file or d.metadata.get("source_filename") == want_file  # 파일이 맞는지
        if same_file and (page + 1) in q["answer_pages"]:             # 파일도 맞고 페이지(1부터)도 맞으면
            return True                                               #   적중
    return False                                                      # 끝까지 없으면 실패


def main():
    parser = argparse.ArgumentParser(description="검색 방식 비교 실험")      # 명령줄 옵션 정의
    parser.add_argument("pdfs", nargs="+", help="실험에 쓸 PDF 경로 (여러 개 가능)")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 4, 6], help="비교할 k값 (기본: 3 4 6)")
    parser.add_argument("--modes", nargs="+", default=list(app.SEARCH_MODES), help="비교할 검색 방식")
    args = parser.parse_args()                                        # 실제 입력된 옵션 읽기

    load_dotenv(ROOT / ".env")                                        # .env의 OPENAI_API_KEY를 환경변수로 등록
    key = os.getenv("OPENAI_API_KEY")                                 # API 키 꺼내기
    if not key:                                                       # 키가 없으면
        sys.exit("[오류] .env에 OPENAI_API_KEY가 없습니다.")             #   안내 후 종료
    if not QA_PATH.exists():                                          # 평가 질문 파일이 없으면
        sys.exit(f"[오류] 평가 질문 파일이 없습니다: {QA_PATH}")          #   안내 후 종료

    questions = load_questions(QA_PATH)                               # 평가 질문 읽기
    scored = [q for q in questions if q["answer_pages"]]              # 정답 페이지가 있는 질문만 적중률 계산에 사용
    print(f"평가 질문 {len(questions)}개 (적중률 계산 대상 {len(scored)}개)")

    db, message = app.build_db(args.pdfs, key)                        # Step 1 함수로 PDF를 DB로 변환 (실제 임베딩 API 사용)
    print(message)                                                    # 처리 결과 출력 (파일·페이지·청크 수)
    if db is None:                                                    # DB 생성에 실패하면
        sys.exit(1)                                                   #   종료

    llm = ChatOpenAI(model=app.MODEL_NAME, temperature=0, api_key=key)  # 멀티쿼리·재정렬에 쓸 GPT (앱과 같은 설정)
    rows = []                                                         # 결과 표의 줄들
    try:
        for mode in args.modes:                                       # 검색 방식마다
            for k in args.k:                                          #   k값마다
                hits, seconds, rerank_fail = 0, [], 0                 #   적중 수, 걸린 시간 목록, 재정렬 실패 수
                for q in scored:                                      #   평가 질문마다
                    start = time.perf_counter()                       #     시간 측정 시작
                    docs, rerank_ok = app._retrieve(db, llm, q["question"], mode=mode, k=k)  # 검색만 실행
                    seconds.append(time.perf_counter() - start)       #     걸린 시간 기록
                    hits += is_hit(docs, q)                           #     적중이면 1 더함 (True=1, False=0)
                    rerank_fail += rerank_ok is False                 #     재정렬 실패면 1 더함
                row = {                                               #   이 설정의 결과 한 줄
                    "mode": mode,
                    "k": k,
                    "hit_rate": round(hits / len(scored), 2) if scored else 0.0,         # 적중률 (0~1)
                    "avg_seconds": round(sum(seconds) / len(seconds), 2) if seconds else 0.0,  # 평균 시간(초)
                    "rerank_fail": rerank_fail,                       # 재정렬 실패 횟수 (재정렬 방식만 의미 있음)
                }
                rows.append(row)                                      #   결과 표에 추가
                print(f"{mode:<18} k={k}  적중률 {row['hit_rate']:.0%}  평균 {row['avg_seconds']}초  재정렬 실패 {rerank_fail}")
    finally:
        db.delete_collection()                                        # 실험이 끝나거나 중간에 멈춰도 임시 DB 삭제

    with OUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 한글이 안 깨지게
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))   # 열 이름: mode, k, hit_rate, ...
        writer.writeheader()                                          # 첫 줄에 열 이름 쓰기
        writer.writerows(rows)                                        # 결과 줄들 쓰기
    print(f"\n결과 저장: {OUT_PATH}")

    best = max(rows, key=lambda r: (r["hit_rate"], -r["avg_seconds"]))  # 적중률이 가장 높고, 같으면 더 빠른 설정
    print(f"추천 설정: SEARCH_MODE = \"{best['mode']}\", TOP_K = {best['k']}")


if __name__ == "__main__":                                            # 이 파일을 직접 실행했을 때만
    main()                                                            #   실험 시작
