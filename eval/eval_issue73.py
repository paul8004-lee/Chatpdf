"""Small live A/B smoke evaluation. Explicit --live is required; charges API usage."""
import argparse
import json
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import app
from report_library import ReportLibrary

QUESTIONS = [
    ("catalog", "2026년 5월 수출 보고서 찾아줘", "수출 보고서 제목과 2026-05-22, 다른 보고서 제외"),
    ("fact", "2025년 방한 외래관광객 수는 몇 명인가요?", "약 1,894만 명, 관광 보고서 출처"),
    ("comparison", "관광정책 보고서와 이란 전쟁 기업 경영 보고서에서 제시하는 정책 과제는 어떻게 다른가?",
     "관광: 체류·소비 중심 / 이란: 유가 기준 현실화·조달 및 공급망 전략, 각 보고서 근거"),
    ("refusal", "세 보고서 중 2027년 한국 경제성장률 전망치를 제시한 보고서는 어디인가요?",
     "없음 또는 확인 불가. 2026년 수치로 대체하지 않음"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--baseline-ref", default="0df5630")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results_issue73.jsonl")
    args = parser.parse_args()
    if not args.live:
        parser.error("실제 API 비용이 발생합니다. 실행하려면 --live를 지정하세요.")
    key = app.OPENAI_API_KEY
    if not key:
        parser.error(".env에 OPENAI_API_KEY가 필요합니다.")
    reports = ReportLibrary().list_reports()
    if not reports or not all(r["available"] for r in reports):
        parser.error("report_library.py로 카탈로그를 동기화하고 PDF를 설치하세요.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".events.jsonl")
    source = subprocess.check_output(["git", "show", f"{args.baseline_ref}:app.py"], cwd=ROOT).decode("utf-8")
    with tempfile.TemporaryDirectory() as folder:
        baseline_file = Path(folder) / "baseline_app.py"
        baseline_file.write_text(source, encoding="utf-8")
        before = runpy.run_path(str(baseline_file))
        app.LOG_PATH = before["answer"].__globals__["LOG_PATH"] = log_path
        # Same text-only index for both versions: isolate routing / evidence selection.
        with patch.object(app, "fitz", None):
            db, message = app.build_db([r["path"] for r in reports], key, report_records=reports)
        if db is None:
            raise RuntimeError(message)
        try:
            for phase, answer_fn, cache in [("before", before["answer"], before["_answer_cache"]),
                                             ("after", app.answer, app._answer_cache)]:
                for kind, question, expected in QUESTIONS:
                    cache.clear()
                    previous = log_path.stat().st_size if log_path.exists() else 0
                    started = time.perf_counter()
                    result = ""
                    error = None
                    try:
                        for result in answer_fn(question, key, db):
                            pass
                    except Exception as exc:
                        # Full sanitized traceback remains in events; keep report free of credentials.
                        error = type(exc).__name__
                    seconds = round(time.perf_counter() - started, 4)
                    with log_path.open("rb") as stream:
                        stream.seek(previous)
                        events = [json.loads(line) for line in stream.read().decode("utf-8").splitlines()]
                    finish = next((e for e in reversed(events) if e["event"] == "request_finished"), {})
                    tokens = finish.get("token_usage", {})
                    record = dict(phase=phase, kind=kind, question=question, expected=expected, answer=result,
                                  error=error, seconds=seconds, tokens=tokens,
                                  llm_usd_regular_input_estimate=round((tokens.get("input_tokens", 0)*0.15 +
                                                                      tokens.get("output_tokens", 0)*0.60)/1e6, 8),
                                  usage_status=finish.get("usage_status"), text_only_index=True,
                                  baseline_ref=args.baseline_ref, review_status="manual_review_required")
                    with args.output.open("a", encoding="utf-8") as out:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(phase, kind, seconds, tokens.get("total_tokens"), error or "OK", flush=True)
        finally:
            app._delete_db(db)


if __name__ == "__main__":
    main()
