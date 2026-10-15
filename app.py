# ===== Step 1. PDF 처리 (feature/pdf-processing) =====
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def build_db(pdf_paths, openai_key):
    """단일 경로 또는 경로 목록을 받아 (새 DB, 처리 메시지)를 반환한다.

    실패 시 DB는 None이다. 기존 DB 해제와 질문 상태 제어는 호출자가 담당한다.
    """
    if not pdf_paths:
        return None, ""
    if not openai_key or not openai_key.strip():
        return None, "⚠️ OpenAI API 키를 입력한 뒤 PDF를 다시 올려주세요."
    if isinstance(pdf_paths, (str, Path)):
        pdf_paths = [pdf_paths]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100,
        length_function=len, is_separator_regex=False,
    )
    documents = []
    page_count = 0
    db = None
    filename = Path(pdf_paths[0]).name
    try:
        for pdf_path in pdf_paths:
            filename = Path(pdf_path).name
            pages = PyPDFLoader(str(pdf_path)).load()
            for page in pages:
                page.metadata["source_filename"] = filename
            chunks = splitter.split_documents(pages)
            if not chunks:
                raise ValueError("검색 가능한 텍스트가 없습니다.")
            documents.append((filename, chunks))
            page_count += len(pages)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_key)
        db = Chroma(collection_name=f"pdf_{uuid.uuid4().hex}", embedding_function=embeddings)
        for filename, chunks in documents:
            db.add_documents(chunks)
    except Exception:
        if db is not None:
            db.delete_collection()
        return None, f"❌ 처리 실패: {filename}. PDF 파일과 API 키/연결을 확인하세요."
    chunk_count = sum(len(chunks) for _, chunks in documents)
    return db, f"✅ 처리 완료: {len(documents)}개 파일 · {page_count}페이지 · {chunk_count}개 청크"


# ===== Step 2. 질문 처리 (feature/qa-chain) =====
# 이 구역의 역할: 사용자가 질문을 입력하면
#   (1) 질문을 다듬고 → (2) PDF에서 관련 내용을 찾고 → (3) 가장 관련 있는 내용만 고른 뒤
#   → (4) GPT가 그 내용을 근거로 답을 만들어 화면에 조금씩 보여준다.
# Step 2에서 쓰는 import는 이 구역에 모아 둡니다. (다른 Step과 merge 충돌 방지)

# ---------- 파이썬 기본 라이브러리 ----------
import json                      # 성능 기록을 JSON 형식(한 줄짜리 텍스트)으로 저장할 때 사용
import re                        # 정규표현식: 파일 경로에서 파일 이름만 잘라낼 때 사용
import time                      # 답변에 걸린 시간(초)을 잴 때 사용
import weakref                   # DB 객체가 사라지는 순간을 감지해서 그 DB의 캐시를 지울 때 사용
from datetime import datetime    # 성능 기록에 "언제 질문했는지" 날짜·시간을 남길 때 사용


# ---------- 외부 패키지 (requirements.txt로 설치됨) ----------
import gradio as gr              # 웹 화면 라이브러리. 여기서는 gr.Error로 화면에 오류 메시지를 띄우는 데 사용
import openai                    # OpenAI 공식 라이브러리. API 키 오류 등 '오류 종류'를 구분하는 데 사용
from langchain.retrievers.document_compressors import LLMListwiseRerank  # GPT가 검색 결과의 순위를 다시 매겨주는 도구
from langchain.retrievers.multi_query import MultiQueryRetriever         # 질문 하나를 여러 표현으로 바꿔 여러 번 검색하는 도구
from langchain_core.output_parsers import StrOutputParser                        # GPT 응답 객체에서 '글자(문자열)'만 꺼내주는 도구
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate            # GPT에게 보낼 지시문(프롬프트) 틀을 만드는 도구
from langchain_openai import ChatOpenAI                                          # LangChain에서 GPT 모델을 부르는 도구

# ---------- 설정값 (숫자만 바꾸면 동작이 바뀜) ----------
MODEL_NAME = "gpt-4o-mini"       # 사용할 GPT 모델 이름
TOP_K = 4                        # 최종적으로 GPT에게 넘겨줄 PDF 조각 개수
MAX_QUESTION_LEN = 500           # 질문 최대 글자 수 (너무 긴 질문은 비용이 커서 막음)
MAX_HISTORY_TURNS = 3            # 후속 질문을 이해하려고 참고할 '최근 대화' 개수 (질문+답변 한 쌍이 1개)
CACHE_SIZE = 100                 # 답변 캐시(임시 저장소)에 보관할 최대 질문 수
LOG_PATH = Path("eval/qa_log.jsonl")  # 질문별 성능 기록을 저장할 파일 위치

# ---------- 검색 방식 설정 ----------
# 이 값 하나만 바꾸면 검색 방식이 바뀜
#   "similarity"        : 질문과 가장 비슷한 조각 TOP_K개 (기본 방식)
#   "mmr"               : 비슷하면서도 내용이 겹치지 않는 조각 TOP_K개
#   "multiquery"        : 질문을 3개로 바꿔 각각 검색 (GPT 1회 추가)
#   "multiquery_rerank" : multiquery + GPT가 순위 재정렬 (GPT 2회 추가)
SEARCH_MODES = ("similarity", "mmr", "multiquery", "multiquery_rerank")
SEARCH_MODE = "multiquery_rerank"  # 비교 실험 후 최종 결정

_answer_cache = {}               # 답변 캐시: {(PDF 고유 이름, 질문): 답변} 형태로 저장. 앱을 끄면 사라짐
_watched_db_keys = set()         # 사라지면 캐시를 지우도록 등록해 둔 DB 이름 목록 (중복 등록 방지)
_dead_db_keys = set()            # 이미 사라진 DB 이름 목록. 다음 캐시 사용 때 그 DB의 캐시를 지움

# ---------- 프롬프트 1: 멀티쿼리 (검색어 늘리기) ----------
# 질문 하나로만 검색하면 표현이 조금만 달라도 관련 내용을 놓칠 수 있음.
# 그래서 GPT에게 같은 뜻의 다른 질문 3개를 만들게 해서 여러 번 검색함.
MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    "당신은 문서 검색을 돕는 도우미입니다.\n"                                       # GPT의 역할 지정
    "아래 질문을 벡터 검색에 잘 걸리도록 서로 다른 관점의 한국어 질문 3개로 바꾸세요.\n"  # 할 일: 질문 3개 만들기
    "한 줄에 하나씩, 번호 없이 질문만 출력하세요.\n\n"                               # 출력 형식 (한 줄씩 나눠서 읽기 때문)
    "원래 질문: {question}"                                                       # {question} 자리에 실제 질문이 들어감
)

# ---------- 프롬프트 2: 후속 질문 재작성 ----------
# "그거 더 자세히 알려줘"처럼 앞 대화를 모르면 이해 안 되는 질문을
# "2026년 경제성장률 전망을 더 자세히 알려줘"처럼 혼자서도 이해되는 질문으로 바꿈.
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",                                                                    # system: GPT에게 주는 규칙
     "이전 대화를 참고해서 마지막 질문을 그 자체로 이해되는 하나의 질문으로 다시 쓰세요. "
     "답하지 말고 다시 쓴 질문만 출력하세요."),
    ("human", "이전 대화:\n{history}\n\n마지막 질문: {question}"),                 # human: 실제 입력 (대화 기록 + 새 질문)
])

# ---------- 프롬프트 3: 최종 답변 ----------
# 검색한 PDF 내용({context})만 근거로 답하게 해서, GPT가 지어내는 것(환각)을 줄임.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 현대경제연구원 보고서를 바탕으로 답하는 어시스턴트입니다.\n"
     "규칙:\n"
     "1. 아래 [문서] 내용만 근거로 한국어로 답하세요.\n"                            # 근거 제한
     "2. 문서에 없는 내용은 추측하지 말고 '문서에서 확인할 수 없습니다'라고 답하세요.\n"  # 모르면 모른다고
     "3. 수치는 문서 표현 그대로 쓰고, 근거 문장 끝에 (p.페이지) 형식으로 표시하세요.\n"  # 출처 표시
     "4. 핵심부터 간결하게 답하세요.\n\n"                                         # 답변 스타일
     "[문서]\n{context}"),                                                        # {context}에 검색된 PDF 내용이 들어감
    ("human", "{question}"),                                                      # {question}에 사용자 질문이 들어감
])


def _page_label(meta):
    """PDF 조각의 정보(meta)에서 사람이 보는 페이지 번호를 꺼냄."""
    if meta.get("page_label"):              # PDF에 인쇄된 페이지 번호(예: 'iv', '12')가 있으면
        return str(meta["page_label"])      #   그 번호를 그대로 사용
    if isinstance(meta.get("page"), int):   # 없으면 컴퓨터식 페이지 번호(0부터 시작)를 확인해서
        return str(meta["page"] + 1)        #   사람식(1부터 시작)으로 바꾸려고 1을 더함
    return "?"                              # 페이지 정보가 아예 없으면 물음표 표시


def _format_docs(docs):
    """검색된 PDF 조각들을 GPT에게 보낼 하나의 글로 합침. 조각마다 [p.페이지]를 앞에 붙임."""
    return "\n\n".join(                                         # 조각 사이를 빈 줄로 구분해서 이어 붙임
        f"[p.{_page_label(d.metadata)}] {d.page_content}"       # 예: "[p.3] 2026년 성장률은 ..."
        for d in docs                                           # 검색된 조각을 하나씩 처리
    )


def _format_sources(docs):
    """답변 아래에 붙일 '참고 페이지' 문구를 만듦. 예: '참고: report.pdf (p.2, 5, 7)'"""
    pages = {}                                                  # {파일이름: [페이지 목록]}을 담을 빈 상자
    for d in docs:                                              # 조각을 하나씩 보면서
        # 경로 'C:\data\report.pdf' 또는 '/tmp/report.pdf'에서 'report.pdf'만 꺼냄 (\ 와 / 모두 처리)
        name = re.split(r"[\\/]", d.metadata.get("source", ""))[-1] or "문서"
        label = _page_label(d.metadata)                         # 이 조각의 페이지 번호
        if label not in pages.setdefault(name, []):             # 같은 파일·같은 페이지가 이미 있으면 건너뜀 (중복 제거)
            pages[name].append(label)                           # 처음 나온 페이지만 목록에 추가
    sort_key = lambda p: (0, int(p)) if p.isdigit() else (1, p)  # 숫자 페이지는 숫자 순서로, 'iv' 같은 건 뒤로 정렬
    parts = [f"{name} (p.{', '.join(sorted(ps, key=sort_key))})"  # 파일마다 "report.pdf (p.2, 5)" 형태로 만듦
             for name, ps in pages.items()]
    return "\n\n---\n**참고**: " + " / ".join(parts)              # 구분선(---) 아래에 굵은 글씨 '참고'로 표시


def _rerank(llm, docs, query, k=None):
    """검색된 조각 중 질문과 가장 관련 있는 k개를 GPT가 골라 순서를 다시 매김.
    반환: (조각 목록, 재정렬 성공 여부)
    """
    # k를 안 주면 설정값 TOP_K 사용
    k = k or TOP_K

    # 조각이 이미 k개 이하면 고를 필요 없음 (API 호출 절약)
    if len(docs) <= k:
        return docs, True

    try:
        reranker = LLMListwiseRerank.from_llm(llm=llm, top_n=k)
        ranked = list(reranker.compress_documents(docs, query))
        if ranked:
            return ranked, True
        # 결과가 비면 원래 순서의 앞 k개 사용 (실패로 기록)
        return docs[:k], False
    except Exception:
        # 오류가 나도 답변은 계속되도록 원래 순서의 앞 k개 사용 (실패로 기록)
        return docs[:k], False


def _retrieve(db, llm, query, mode=None, k=None):
    """[검색 방식 변경] SEARCH_MODE에 맞는 방식으로 PDF 조각을 찾음.
    반환: (조각 목록, 재정렬 성공 여부). 재정렬을 안 하는 방식이면 None.
    answer()와 비교 실험 스크립트(eval/compare_retrievers.py)가 같이 씀.
    """
    # 값을 안 주면 설정값 사용
    mode = mode or SEARCH_MODE
    k = k or TOP_K
    if mode not in SEARCH_MODES:
        raise ValueError(f"알 수 없는 검색 방식: {mode} (가능: {SEARCH_MODES})")

    # 1) 기본 검색기 만들기
    if mode == "similarity":
        # 질문과 가장 비슷한 k개
        base = db.as_retriever(search_kwargs={"k": k})
    else:
        # MMR: 후보 k×5개 중 서로 겹치지 않게 k개
        base = db.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 5})

    # 2) similarity, mmr은 여기서 끝
    if mode in ("similarity", "mmr"):
        return base.invoke(query), None

    # 3) 멀티쿼리: 질문을 3개로 바꿔 원래 질문까지 총 4번 검색 (중복은 자동 제거)
    retriever = MultiQueryRetriever.from_llm(
        retriever=base,
        llm=llm,
        prompt=MULTI_QUERY_PROMPT,
        include_original=True,
    )
    docs = retriever.invoke(query)

    if mode == "multiquery":
        return docs[:k], None

    # 4) 재정렬: 후보 k×2개 중 GPT가 k개를 고름
    return _rerank(llm, docs[: k * 2], query, k)


def _db_key(db):
    """[버그 수정] 캐시에서 PDF DB를 구분하는 고유 이름.
    예전에는 id(db)(메모리 주소)를 썼는데, DB를 버리고 새 DB를 만들면
    파이썬이 같은 주소를 다시 줄 수 있어서 이전 PDF의 답변이 나오는 문제가 있었음.
    → Step 1이 업로드마다 새로 만드는 고유 컬렉션 이름(pdf_<uuid>)을 사용.
    (db._collection은 DB 삭제 후 오류가 나므로, 삭제 후에도 남는 _collection_name을 씀)
    """
    return getattr(db, "_collection_name", None) or id(db)    # 이름이 없는 DB만 예전처럼 id 사용


def _remove_cache_of(key):
    """key(DB 이름)에 해당하는 캐시를 모두 지우고, 지운 개수를 돌려줌."""
    old = [k for k in list(_answer_cache) if k[0] == key]        # 이 DB로 저장된 (DB 이름, 질문) 목록
    for k in old:
        _answer_cache.pop(k, None)                               # 하나씩 삭제 (이미 없으면 무시)
    return len(old)


def _purge_dead_caches():
    """사라진 DB들의 캐시를 정리함. 캐시를 읽거나 쓰기 직전에 매번 호출."""
    while _dead_db_keys:                                         # 정리할 DB가 남아 있는 동안
        key = _dead_db_keys.pop()                                #   하나 꺼내서
        _remove_cache_of(key)                                    #   그 DB의 캐시 삭제
        _watched_db_keys.discard(key)                            #   감시 목록에서도 제거


def _watch_db(db, key):
    """DB 객체가 사라지면(새 PDF 등록으로 버려지는 경우 등) 그 DB 이름을 '정리 대상'에 넣도록 등록.
    여기서 바로 지우지 않는 이유: 객체가 사라지는 시점은 예측할 수 없어서,
    캐시를 사용하는 도중에 지우면 오류가 날 수 있음 → 다음 캐시 사용 때 안전하게 정리."""
    if key in _watched_db_keys:                                  # 이미 등록했으면
        return                                                   #   다시 등록하지 않음
    try:
        weakref.finalize(db, _dead_db_keys.add, key)             # db가 사라지는 순간 key를 정리 대상에 추가
        _watched_db_keys.add(key)
    except TypeError:                                            # 감시할 수 없는 객체면 (드묾)
        pass                                                     #   자동 정리만 생략 (캐시 구분은 정상 동작)


def _cache_get(db, query):
    """[업그레이드 2] 같은 PDF(db)에 같은 질문(query)을 한 적이 있으면 저장해 둔 답변을 꺼냄. 없으면 None."""
    _purge_dead_caches()                                         # 사라진 DB의 캐시부터 정리
    return _answer_cache.get((_db_key(db), query))               # (PDF 고유 이름, 질문)으로 저장된 답변 찾기


def _cache_put(db, query, text):
    """[업그레이드 2] 새로 만든 답변을 캐시에 저장함."""
    _purge_dead_caches()                                         # 사라진 DB의 캐시부터 정리
    if len(_answer_cache) >= CACHE_SIZE:                         # 캐시가 가득 찼으면
        _answer_cache.pop(next(iter(_answer_cache)))             #   가장 먼저 저장된(가장 오래된) 답변 1개를 지움
    key = _db_key(db)                                            # 이 PDF의 고유 이름
    _answer_cache[(key, query)] = text                           # (PDF 고유 이름, 질문)을 열쇠로 답변을 저장
    _watch_db(db, key)                                           # 이 DB가 사라지면 캐시가 자동 정리되도록 등록


def clear_answer_cache(db=None):
    """[버그 수정 · 화면 팀 호출용] 캐시를 직접 정리함.
    db를 주면 그 PDF의 캐시만, 안 주면 전체 캐시를 지움. 반환: 지운 개수.
    DB를 직접 삭제할 때(old_db.delete_collection()) 그 전에 호출하면 됨.
    """
    _purge_dead_caches()                                         # 사라진 DB의 캐시부터 정리
    if db is None:                                               # 전체 삭제
        count = len(_answer_cache)
        _answer_cache.clear()
        return count
    return _remove_cache_of(_db_key(db))                         # 이 PDF의 캐시만 삭제


def _log(**record):
    """[업그레이드 3] 질문 1개마다 걸린 시간 등을 eval/qa_log.jsonl 파일에 한 줄씩 추가 기록."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)      # eval 폴더가 없으면 만듦 (있으면 그냥 넘어감)
        record["time"] = datetime.now().isoformat(timespec="seconds")  # 기록 시각 추가. 예: '2026-10-14T17:30:05'
        with LOG_PATH.open("a", encoding="utf-8") as f:         # 파일을 '이어쓰기(a)' 모드로 열어서
            f.write(json.dumps(record, ensure_ascii=False) + "\n")  # 한글이 깨지지 않게 JSON 한 줄로 저장
    except Exception:                                           # 기록에 실패해도 (권한 문제 등)
        pass                                                    #   답변에는 영향 없도록 조용히 넘어감


def _history_to_text(history):
    """Gradio가 넘겨주는 대화 기록을 GPT가 읽을 수 있는 글로 바꿈. 대화가 없으면 빈 글자('')."""
    if not history:                                             # 대화 기록이 없으면 (첫 질문이면)
        return ""                                               #   빈 글자를 돌려줌
    lines = []                                                  # 바꾼 대화를 한 줄씩 담을 목록
    for item in history[-MAX_HISTORY_TURNS * 2:]:               # 최근 대화만 사용 (질문+답변이 각각 1개씩이라 ×2)
        if isinstance(item, dict):                              # 최신 Gradio 형식: {"role": "user", "content": "..."}
            role = "사용자" if item.get("role") == "user" else "AI"  # 누가 한 말인지 한글로 표시
            content = item.get("content", "")                   # 실제 대화 내용
            if isinstance(content, list):                       # 이미지 등이 섞인 형식이면 (내용이 목록으로 옴)
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))  # 글자 부분만 이어 붙임
            lines.append(f"{role}: {content}")                  # 예: "사용자: 성장률은?"
        elif isinstance(item, (list, tuple)) and len(item) == 2:  # 예전 Gradio 형식: [질문, 답변]
            lines.append(f"사용자: {item[0]}")                   # 첫 번째는 사용자 질문
            lines.append(f"AI: {item[1]}")                      # 두 번째는 AI 답변
    return "\n".join(lines)                                     # 줄바꿈으로 이어서 하나의 글로 만듦


def answer(question, openai_key, db, history=None):
    """질문을 받아 답변을 만드는 메인 함수.
    question  : 사용자가 입력한 질문
    openai_key: 사용자가 입력한 OpenAI API 키
    db        : Step 1에서 PDF를 잘라 저장한 벡터 데이터베이스
    history   : (선택) 이전 대화 기록. 넘기지 않아도 동작함
    yield를 쓰기 때문에 답변이 한 번에 나오지 않고 타자 치듯 조금씩 화면에 표시됨.
    """
    # ===== 1단계: 입력 검사 =====
    if db is None:                                              # PDF가 아직 업로드되지 않았으면
        raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")  #   화면에 오류 창을 띄우고 멈춤
    question = (question or "").strip()                         # 질문 앞뒤의 공백·줄바꿈 제거 (None이면 빈 글자로)
    if not question:                                            # 공백만 입력했거나 비어 있으면
        raise gr.Error("질문을 입력하세요.")                       #   오류 창 띄우고 멈춤
    if len(question) > MAX_QUESTION_LEN:                        # 질문이 너무 길면
        raise gr.Error(f"질문은 {MAX_QUESTION_LEN}자 이내로 입력하세요.")  # 오류 창 띄우고 멈춤

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=openai_key)  # GPT 준비. temperature=0: 같은 질문엔 최대한 같은 답
    start = time.perf_counter()                                 # [업그레이드 3] 시간 측정 시작 (스톱워치 누르기)

    try:                                                        # 아래에서 OpenAI 오류가 나면 맨 아래 except에서 처리
        # ===== 2단계: 후속 질문이면 혼자서도 이해되는 질문으로 바꾸기 =====
        history_text = _history_to_text(history)                # 대화 기록을 글로 변환
        search_question = question                              # 기본값: 사용자가 쓴 질문 그대로 검색
        if history_text:                                        # 이전 대화가 있으면
            search_question = (CONDENSE_PROMPT | llm | StrOutputParser()).invoke(  # 프롬프트 → GPT → 글자만 꺼내기 순서로 실행
                {"history": history_text, "question": question}  # 프롬프트의 {history}, {question} 자리에 넣을 값
            ).strip() or question                               # 결과 앞뒤 공백 제거. 결과가 비면 원래 질문 사용

        # ===== 3단계: [업그레이드 2] 캐시 확인 =====
        cached = _cache_get(db, search_question)                # 같은 질문에 대한 저장된 답변이 있는지 확인
        if cached:                                              # 있으면
            _log(question=question, search_question=search_question, cached=True,
                 mode=SEARCH_MODE,
                 seconds=round(time.perf_counter() - start, 2))
            yield cached                                        # 저장된 답변을 바로 화면에 보여주고
            return                                              # 함수 끝 (검색·GPT 호출 생략 → 빠르고 비용 0)

        # ===== 4단계: PDF에서 관련 내용 검색 (방식은 SEARCH_MODE 설정을 따름) =====
        docs, rerank_ok = _retrieve(db, llm, search_question)

        if not docs:                                            # 관련 내용을 하나도 못 찾았으면
            yield "업로드한 문서에서 관련 내용을 찾지 못했습니다. 질문을 바꿔서 다시 시도해 보세요."  # 안내 문구 표시
            return                                              # 함수 끝

        # ===== 5단계: GPT로 답변 만들기 (조금씩 화면에 표시) =====
        chain = ANSWER_PROMPT | llm | StrOutputParser()         # 프롬프트 → GPT → 글자만 꺼내기를 하나로 연결
        text = ""                                               # 지금까지 받은 답변을 모아 둘 변수
        for chunk in chain.stream(                              # stream: 답변을 한 번에 받지 않고 조각조각 받음
            {"context": _format_docs(docs), "question": search_question}  # {context}=검색된 PDF 내용, {question}=질문
        ):
            text += chunk                                       # 새로 받은 조각을 이어 붙이고
            yield text                                          # 지금까지의 답변을 화면에 갱신 (타자 치는 효과)

        # ===== 6단계: 참고 페이지 붙이기 =====
        final = text + _format_sources(docs)                    # 완성된 답변 아래에 '참고: report.pdf (p.2, 5)' 추가
        yield final                                             # 최종 답변을 화면에 표시

        # ===== 7단계: [업그레이드 2, 3] 캐시 저장 + 성능 기록 =====
        _cache_put(db, search_question, final)                  # 다음에 같은 질문이 오면 바로 답하도록 저장
        _log(question=question, search_question=search_question, cached=False,
             mode=SEARCH_MODE, rerank_ok=rerank_ok,
             seconds=round(time.perf_counter() - start, 2),
             pages=[_page_label(d.metadata) for d in docs])

    # ===== 오류 처리: OpenAI 오류를 사용자가 이해할 수 있는 한국어로 안내 =====
    except openai.AuthenticationError:                          # API 키가 틀렸거나 만료된 경우
        raise gr.Error("API 키가 올바르지 않습니다. 키를 다시 확인하세요.")
    except openai.RateLimitError:                               # 너무 자주 요청했거나 크레딧(사용 한도)을 다 쓴 경우
        raise gr.Error("OpenAI 사용 한도를 초과했습니다. 잠시 후 다시 시도하세요.")
    except openai.APIConnectionError:                           # 인터넷 연결 문제 등으로 OpenAI 서버에 접속 못 한 경우
        raise gr.Error("OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인하세요.")

# ===== Step 3. 화면 (feature/chat-ui) =====
# ===== Step 3. 화면 (feature/chat-ui) =====

import os
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

# .env의 OPENAI_API_KEY를 환경변수로 읽어옴 (Spaces에서는 Secrets에 등록)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



# ==========================================
# 1. CSS 디자인 + 다크모드
# ==========================================

CSS = """
.gradio-container {
    --page: #FFFFFF;
    --card: #FFFFFF;
    --field: #FFFFFF;
    --answer: #F7F9FF;
    --fg: #202638;
    --muted: #738096;
    --line: #E3E8F1;

    background: var(--page) !important;
    color: var(--fg) !important;
    max-width: 720px !important;
    margin: 0 auto !important;
    padding: 30px 18px !important;
    min-height: 100vh;
    font-family: 'Pretendard', Arial, sans-serif !important;
}

/* 기본 배경 */

html, body {
    background: #FFFFFF !important;
}

/* 다크모드 */

html:has(#mode-select input[type="checkbox"]:checked),
body:has(#mode-select input[type="checkbox"]:checked) {
    background: #11141C !important;
}

.gradio-container:has(
    #mode-select input[type="checkbox"]:checked
) {
    --page: #11141C;
    --card: #1A1E2B;
    --field: #252B3A;
    --answer: #252B3A;
    --fg: #F3F5FA;
    --muted: #ACB5C8;
    --line: #394256;

    color-scheme: dark;
}

footer {
    display: none !important;
}

/* 메인 카드 */

#main-card {
    background: var(--card) !important;
    color: var(--fg) !important;
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px rgba(40,60,100,.05);
    gap: 14px !important;
}

/* 제목 */

#title-row {
    align-items: center !important;
}

.page-title {
    font-size: 21px;
    font-weight: 800;
    color: var(--fg);
    margin: 0;
    line-height: 1.4;
}

/* 다크모드 체크박스 */

#mode-select {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    min-width: 0 !important;
}

#mode-select label,
#mode-select span {
    color: var(--fg) !important;
    font-size: 13px !important;
}

/* 그라데이션 구분선 */

.gradient-line {
    width: 100%;
    height: 3px;
    border-radius: 99px;
    background: linear-gradient(
        90deg,
        #329AFF,
        #8578FF,
        #FF65AF
    );
    margin: 3px 0 9px;
}

/* 영역 제목 */

.section-label {
    font-size: 14px;
    font-weight: 700;
    color: var(--fg);
    margin: 0 0 5px;
}

/* PDF 파일 업로드 */

#file-row {
    align-items: stretch !important;
    gap: 10px !important;
}

#pdf-select {
    border-radius: 11px !important;
    background: var(--field) !important;
    color: var(--fg) !important;
    border: 1px solid var(--line) !important;
    min-height: 45px !important;
    box-shadow: none !important;
}

#pdf-select,
#pdf-select * {
    color: var(--fg) !important;
}

#upload-btn {
    background: var(--answer) !important;
    color: var(--fg) !important;
    border: 1px solid var(--line) !important;
    border-radius: 11px !important;
    min-height: 45px !important;
}

/* 업로드 상태 */

#upload-status,
#upload-status * {
    color: var(--muted) !important;
    font-size: 12px !important;
}

/* 질문 입력창 */

#question-box {
    background: var(--field) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: none !important;
    padding: 0 !important;
}

#question-box textarea {
    display: block !important;
    background: transparent !important;
    color: var(--fg) !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    min-height: 100px !important;
}

#question-box textarea::placeholder {
    color: var(--muted) !important;
}

#question-box:focus-within {
    border-color: #827AFF !important;
}

/* 질문하기 버튼 */

#ask-btn {
    background: linear-gradient(
        110deg,
        #329AFF 0%,
        #827AFF 50%,
        #FF65AF 100%
    ) !important;
    border: none !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    min-height: 47px !important;
    font-size: 14px !important;
    font-weight: 750 !important;
    box-shadow: 0 5px 14px rgba(130,122,255,.15);
}

#ask-btn:hover {
    filter: brightness(1.06);
}

/* 답변 영역 */

#answer-box {
    background: var(--answer) !important;
    border: 1px solid var(--line) !important;
    border-left: 4px solid #827AFF !important;
    border-radius: 12px !important;
    padding: 16px !important;
    min-height: 90px;
    box-shadow: none !important;
}

#answer-box,
#answer-box * {
    color: var(--fg) !important;
}

/* 모바일 대응 */

@media (max-width: 600px) {
    .gradio-container {
        padding: 12px 8px !important;
    }

    #main-card {
        padding: 16px !important;
        border-radius: 16px !important;
    }

    .page-title {
        font-size: 18px;
    }

    #file-row {
        flex-wrap: nowrap !important;
    }
}
"""



# ==========================================
# 2. UI 이벤트 함수
# ==========================================

def select_pdf(file):
    if file is None:
        return None, "파일을 선택해주세요."

    path = file if isinstance(file, str) else file.name

    return path, f"선택한 파일: {Path(path).name}"



def register_pdf(pdf_path):
    if not pdf_path:
        return None, "PDF 파일을 선택해주세요."

    if not OPENAI_API_KEY:
        return None, ":warning: OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요."

    return build_db([pdf_path], OPENAI_API_KEY)



def ask_question(question, db):
    # answer()가 yield로 답변을 조금씩 내보내므로 그대로 화면에 전달
    if not question or not question.strip():
        yield ":warning: 질문을 입력해주세요."
        return

    if db is None:
        yield "먼저 PDF 파일을 등록해주세요."
        return

    try:
        yield from answer(question, OPENAI_API_KEY, db)
    except gr.Error:
        raise
    except Exception as e:
        yield f":warning: 답변 처리 오류: {e}"



# ==========================================
# 3. Gradio Blocks UI
# ==========================================

with gr.Blocks(title="문서 질의응답 서비스") as demo:

    db_state = gr.State(value=None)
    selected_file = gr.State(value=None)

    with gr.Column(elem_id="main-card"):

        # 제목 + 다크모드
        with gr.Row(elem_id="title-row"):

            with gr.Column(scale=5, min_width=180):
                gr.HTML("""
                <div class="page-title">
                    문서 질의응답 서비스
                </div>
                """)

            with gr.Column(scale=1, min_width=110):
                dark_mode = gr.Checkbox(
                    label=":crescent_moon: 다크모드",
                    value=False,
                    container=False,
                    elem_id="mode-select"
                )

        gr.HTML('<div class="gradient-line"></div>')

        # PDF 업로드
        gr.HTML(
            '<div class="section-label">:page_facing_up: PDF 파일</div>'
        )

        with gr.Row(elem_id="file-row"):

            pdf_file = gr.UploadButton(
                ":paperclip: PDF 파일 선택",
                file_types=[".pdf"],
                file_count="single",
                variant="secondary",
                scale=5,
                min_width=160,
                elem_id="pdf-select"
            )

            upload_btn = gr.Button(
                "파일 등록",
                variant="secondary",
                scale=2,
                min_width=95,
                elem_id="upload-btn"
            )

        status = gr.Markdown(
            "파일을 선택해주세요.",
            elem_id="upload-status"
        )

        # 질문
        gr.HTML(
            '<div class="section-label">:speech_balloon: 질문</div>'
        )

        question = gr.Textbox(
            placeholder="문서에 대해 궁금한 내용을 입력하세요...",
            show_label=False,
            container=False,
            lines=3,
            max_lines=5,
            elem_id="question-box"
        )

        ask_btn = gr.Button(
            "질문하기 →",
            variant="primary",
            elem_id="ask-btn"
        )

        # 답변
        gr.HTML(
            '<div class="section-label">:sparkles: 답변</div>'
        )

        output = gr.Markdown(
            "질문을 입력하면 이곳에 답변이 표시됩니다.",
            elem_id="answer-box"
        )

    # ======================================
    # 4. 이벤트 핸들러
    # ======================================

    pdf_file.upload(
        fn=select_pdf,
        inputs=[pdf_file],
        outputs=[selected_file, status]
    )

    upload_btn.click(
        fn=register_pdf,
        inputs=[selected_file],
        outputs=[db_state, status]
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question, db_state],
        outputs=output
    )

    question.submit(
        fn=ask_question,
        inputs=[question, db_state],
        outputs=output
    )



# ==========================================
# 5. 서버 실행
# ==========================================

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=gr.themes.Soft()
    )