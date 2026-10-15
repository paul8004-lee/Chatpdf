"""Small SQLite report catalog. PDF bodies remain in the existing Chroma workflow."""
import argparse
import json
import re
import sqlite3
import hashlib
import shutil
from contextlib import closing
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "library" / "catalog.sqlite3"
DEFAULT_MANIFEST = ROOT / "library" / "reports.json"


def classify_question(question):
    if re.search(r"찾|목록|어떤\s*보고서|무슨\s*보고서|보고서.*추천|발행.*보고서", question):
        return "documents"
    if re.search(r"비교|차이|공통|각각|연계|연결", question):
        return "comparison"
    if re.search(r"종합|제안|대응\s*방안|전략.*세워", question):
        return "synthesis"
    return "fact"


def _terms(text):
    return set(re.findall(r"[가-힣A-Za-z0-9]+", text.lower()))


def rank_reports(question, reports, filter_dates=True):
    """Match catalog fields, including partial titles and YYYY-MM / Korean dates."""
    normalized = re.sub(r"(\d{4})년\s*(\d{1,2})월", lambda m: f"{m[1]}-{int(m[2]):02}", question)
    dates = re.findall(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", normalized)
    generic = {"보고서", "주요", "시사점", "분석", "현대경제연구원", "따른", "대한", "미치는"}
    def anchors(report):
        return {term for term in _terms(report["title"] + " " + " ".join(report.get("topics", [])))
                if len(term) >= 2 and term not in generic and not term.isdigit()}
    topical = any(term in question.lower() for report in reports for term in anchors(report))
    scored = []
    for report in reports:
        if filter_dates and dates and not all(d in report.get("published_date", "") or d in report["title"] for d in dates):
            continue
        score = sum(1 for term in anchors(report) if term in question.lower())
        score += sum(3 for topic in report.get("topics", []) if topic.lower() in question.lower())
        if report.get("filename", "") and report["filename"].lower() in question.lower():
            score += 20
        if filter_dates and dates and not topical:
            score += 1
        if score:
            scored.append((score, report))
    return [r for _, r in sorted(scored, key=lambda item: (-item[0], item[1]["title"]))]


def select_reports(question, reports, intent):
    found = rank_reports(question, reports, filter_dates=intent == "documents")
    # Generic cross-report questions mean the active report set, not one topical hit.
    if re.search(r"모든|전체|세\s*(?:문서|보고서)|각\s*보고서", question):
        return list(reports)
    if intent == "documents" and re.fullmatch(r"\s*(?:보고서\s*)?(?:목록|리스트)(?:\s*(?:보여줘|알려줘))?\s*", question):
        return list(reports)
    return found or (list(reports) if intent != "documents" else [])


class ReportLibrary:
    def __init__(self, db_path=DEFAULT_DB):
        self.path = Path(db_path)

    def save_uploaded(self, record):
        """Keep uploaded PDFs and their generated metadata across manifest refreshes."""
        original = Path(record["path"])
        with original.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        target = self.path.parent / "uploads" / digest / original.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.resolve() != target.resolve():
            shutil.copyfile(original, target)
        stored = dict(record, path=str(target.resolve()))
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS uploaded_reports (digest TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conn.execute("INSERT OR REPLACE INTO uploaded_reports VALUES (?, ?)",
                         (digest, json.dumps(stored, ensure_ascii=False)))
        return stored

    def sync(self, manifest=DEFAULT_MANIFEST):
        manifest = Path(manifest).resolve()
        records = json.loads(manifest.read_text(encoding="utf-8"))
        prepared = []
        seen = set()
        for record in records:
            path = (manifest.parent / record["path"]).resolve()
            date.fromisoformat(record["published_date"])
            if record["id"] in seen:
                raise ValueError("보고서 id가 중복되었습니다.")
            seen.add(record["id"])
            prepared.append((record["id"], record["title"], record["published_date"],
                             json.dumps(record["topics"], ensure_ascii=False), record["summary"],
                             str(path), record.get("source_url", "")))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY, title TEXT NOT NULL, published_date TEXT NOT NULL,
                topics TEXT NOT NULL, summary TEXT NOT NULL, path TEXT NOT NULL, source_url TEXT NOT NULL)""")
            conn.executemany("INSERT OR REPLACE INTO reports VALUES (?, ?, ?, ?, ?, ?, ?)", prepared)
            ids = [r[0] for r in prepared]
            if ids:
                conn.execute(f"DELETE FROM reports WHERE report_id NOT IN ({','.join('?' for _ in ids)})", ids)
            else:
                conn.execute("DELETE FROM reports")
        return self.list_reports()

    def list_reports(self):
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(self.path)) as conn:
            conn.row_factory = sqlite3.Row
            has_reports = conn.execute("SELECT 1 FROM sqlite_master WHERE name='reports'").fetchone()
            rows = conn.execute("SELECT * FROM reports ORDER BY published_date DESC, title").fetchall() if has_reports else []
            has_uploads = conn.execute("SELECT 1 FROM sqlite_master WHERE name='uploaded_reports'").fetchone()
            uploads = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM uploaded_reports")] if has_uploads else []
        records = []
        for row in list(rows) + uploads:
            record = dict(row)
            if isinstance(record["topics"], str):
                record["topics"] = json.loads(record["topics"])
            record["filename"] = Path(record["path"]).name
            record["available"] = Path(record["path"]).is_file()
            records.append(record)
        return records

    def search(self, question):
        records = self.list_reports()
        return rank_reports(question, records) if question.strip() else records


def format_catalog(records):
    if not records:
        return "조건에 맞는 보고서를 찾지 못했습니다. 제목·주제·발행일을 바꿔 검색해 주세요."
    return "\n\n".join(
        f"**{r['title']}** ({r['published_date']})\n\n"
        f"주제: {', '.join(r['topics'])}  \n{r['summary']}  \n"
        f"파일: `{r['filename']}` · {'본문 사용 가능' if r['available'] else 'PDF 미설치'}"
        + (f" · [원문]({r['source_url']})" if r.get('source_url') else "")
        for r in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="보고서 JSON 목록을 SQLite 카탈로그에 반영")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    records = ReportLibrary(args.database).sync(args.manifest)
    print(f"보고서 {len(records)}개 동기화, PDF 준비 {sum(r['available'] for r in records)}개")
