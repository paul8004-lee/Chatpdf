# ===== Step 1. PDF 처리 (feature/pdf-processing) =====
import logging
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import request_log
import base64
import os
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 표·이미지 청킹용 PyMuPDF가 없어도 앱은 켜지고 텍스트 PDF 등록은 되도록 함 (#65)
try:
    import pymupdf as fitz
except ImportError:
    fitz = None
    logging.getLogger(__name__).warning("pymupdf가 없어 표·이미지 청킹을 건너뜁니다. pip install -r requirements.txt")

NO_TEXT_ERROR = "검색 가능한 텍스트가 없습니다."

# ==========================================
# 표 / 이미지 전담 청킹 클래스 정의
# ==========================================
class VisualElementChunker:
    """PDF 내에서 일반 텍스트는 건너뛰고, '표(Table)'와 '이미지/차트(Image)'만 전담으로 추출 및 요약 청킹"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        image_save_dir: str = "./extracted_images",
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0, api_key=api_key)
        self.image_save_dir = image_save_dir
        os.makedirs(self.image_save_dir, exist_ok=True)

    def _summarize_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = """
        당신은 데이터 시각화 및 문서 분석 전문가입니다.
        제공된 이미지(차트, 그래프, 다이어그램 등)를 면밀히 분석하여 핵심 정보를 기술하세요.
        - 차트/그래프 종류 (예: 꺾은선, 막대, 파이차트 등)
        - 축 레이블, 범례, 주요 수치 데이터 및 증감 트렌드
        - 이미지가 전달하고자 하는 핵심 결론
        사용자가 자연어로 질문했을 때 벡터 검색에 잘 걸리도록 핵심 키워드와 숫자를 포함해 요약하세요.
        """
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
            ]
        )
        response = self.llm.invoke([message])
        return response.content

    def _summarize_table(self, table_markdown: str) -> str:
        prompt = f"""
        다음 표의 구조와 수치 데이터를 분석하여 핵심 요약문을 작성하세요.
        사용자가 자연어로 검색할 때 이 표가 잘 검색될 수 있도록 주요 항목명, 수치, 결론을 포함하세요.

        [표 데이터]
        {table_markdown}
        """
        response = self.llm.invoke(prompt)
        return response.content

    def extract_visual_chunks(self, pdf_path: str, filename: str) -> list[Document]:
        doc = fitz.open(pdf_path)
        visual_chunks: list[Document] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx  # 0-based page index

            # 1. 표(Table) 추출
            tabs = page.find_tables()
            for t_idx, table in enumerate(tabs):
                df = table.to_pandas()
                if df.empty or df.shape[0] < 1:
                    continue

                table_md = df.to_markdown(index=False)
                table_summary = self._summarize_table(table_md)

                visual_chunks.append(
                    Document(
                        page_content=f"[표 데이터 요약]\n{table_summary}\n\n[원본 표]\n{table_md}",
                        metadata={
                            "element_type": "table",
                            "page": page_num,
                            "source_filename": filename,
                            "source": str(pdf_path),
                            "table_index": t_idx,
                            "raw_table_md": table_md,
                        },
                    )
                )

            # 2. 이미지/차트(Image) 추출
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if base_image["width"] < 120 or base_image["height"] < 120:
                    continue

                img_filename = f"page_{page_num + 1}_img_{img_idx}_{uuid.uuid4().hex[:6]}.{base_image['ext']}"
                img_filepath = os.path.join(self.image_save_dir, img_filename)
                with open(img_filepath, "wb") as f:
                    f.write(base_image["image"])

                img_summary = self._summarize_image(img_filepath)

                visual_chunks.append(
                    Document(
                        page_content=f"[이미지/차트 분석]\n{img_summary}",
                        metadata={
                            "element_type": "image",
                            "page": page_num,
                            "source_filename": filename,
                            "source": str(pdf_path),
                            "image_path": img_filepath,
                        },
                    )
                )

        doc.close()
        return visual_chunks

def _layout_page_texts(pdf_path, pages):
    """페이지별 레이아웃 보존 텍스트 {page: text}. 표의 열(대상·연도) 배치가 글자 위치로 남아
    여러 열짜리 표를 GPT가 열별로 읽을 수 있게 함 (#37). 실패하면 일반 추출 텍스트를 사용."""
    try:
        reader = PdfReader(str(pdf_path))
        texts = {}
        for index, page in enumerate(reader.pages):
            lines = (line.rstrip() for line in page.extract_text(extraction_mode="layout").splitlines())
            texts[index] = "\n".join(line for line in lines if line.strip())
        return texts
    except Exception:
        return {page.metadata.get("page"): page.page_content for page in pages}


def _load_failure_reason(error):
    """파일 하나를 읽거나 나누다 실패한 이유를 사용자에게 보여줄 문구로 바꾼다."""
    if isinstance(error, ValueError) and str(error) == NO_TEXT_ERROR:
        return "텍스트가 없는 PDF(이미지·스캔본)"
    return "PDF를 읽을 수 없음(손상되었거나 PDF 형식이 아님)"


def build_db(pdf_paths, openai_key):
    """Process PDF paths and append request diagnostics without storing credentials."""
    with request_log.request(LOG_PATH, "pdf_processing", secrets=(openai_key,)) as trace:
        db, message = _build_db(pdf_paths, openai_key)
        if db is None:
            trace.status = "error" if pdf_paths else "empty"
        trace.emit("pdf_result", status=trace.status, message=message,
                   collection=getattr(db, "_collection_name", None) if db is not None else None)
        return db, message


def _build_db(pdf_paths, openai_key):
    """단일 경로 또는 경로 목록을 받아 (새 DB, 처리 메시지)를 반환한다.

    파일별로 읽기·분할하고, 실패한 파일은 제외한 채 성공한 파일만으로 DB를 만든다 (#42).
    모든 파일이 실패하거나 임베딩·저장이 실패하면 DB는 None이다.
    기존 DB 해제와 질문 상태 제어는 호출자가 담당한다.
    """
    if not pdf_paths:
        return None, ""
    if not openai_key or not openai_key.strip():
        request_log.current().emit("validation_failed", reason="missing_api_key")
        return None, "⚠️ OpenAI API 키를 입력한 뒤 PDF를 다시 올려주세요."
    if isinstance(pdf_paths, (str, Path)):
        pdf_paths = [pdf_paths]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100,
        length_function=len, is_separator_regex=False,
    )
    documents = []
    failures = []
    page_count = 0
    # 시각 청킹 준비가 실패해도 텍스트 청크만으로 등록을 이어감 (#65)
    visual_chunker = None
    if fitz is not None:
        try:
            visual_chunker = VisualElementChunker(api_key=openai_key)
        except Exception as error:
            logging.getLogger(__name__).warning("시각 청킹 준비 실패: %s", error)
    for pdf_path in pdf_paths:
        filename = Path(pdf_path).name
        try:
            with request_log.stage("pdf_load_split"):
                pages = PyPDFLoader(str(pdf_path)).load()
                for page in pages:
                    page.metadata["source_filename"] = filename
                text_chunks = splitter.split_documents(pages)
                # 시각 요소(표, 이미지) 청크 추출
                visual_chunks = []
                if visual_chunker is not None:
                    try:
                        visual_chunks = visual_chunker.extract_visual_chunks(str(pdf_path), filename)
                    except Exception as ve:
                        logging.getLogger(__name__).warning("시각 청크 추출 실패(%s): %s", filename, ve)

                chunks = text_chunks + visual_chunks
                if not chunks:
                    raise ValueError(NO_TEXT_ERROR)
                page_texts = _layout_page_texts(pdf_path, pages)
                for chunk in chunks:
                    chunk.metadata["chunk_id"] = uuid.uuid4().hex
                    # 답변할 때 이 조각이 속한 페이지 전체(표의 열 배치 유지)를 근거로 쓰기 위해 저장 (#37)
                    chunk.metadata["page_text"] = page_texts.get(chunk.metadata.get("page"), "")
                request_log.current().emit("pdf_loaded", filename=filename,
                                           pages=len(pages), chunks=len(chunks))
        except Exception as error:
            request_log.current().emit("pdf_excluded", filename=filename,
                                       reason=_load_failure_reason(error))
            failures.append((filename, _load_failure_reason(error)))
            continue
        documents.append((filename, chunks))
        page_count += len(pages)

    failure_lines = "\n".join(f"- {name}: {reason}" for name, reason in failures)
    if not documents:
        return None, f"❌ 처리 실패: 등록할 수 있는 PDF가 없습니다.\n{failure_lines}"

    # 임베딩·저장 실패는 API 키·연결 문제라 전체 실패로 처리
    db = None
    try:
        with request_log.stage("embedding_storage"):
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_key)
            db = Chroma(collection_name=f"pdf_{uuid.uuid4().hex}", embedding_function=embeddings)
            for filename, chunks in documents:
                db.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    except Exception:
        _delete_db(db)
        return None, f"❌ 처리 실패: {filename}. API 키/연결을 확인하세요."
    finally:
        # LangChain embeddings do not expose response usage to callbacks.
        request_log.current().usage("text-embedding-3-large", reason="embedding_adapter_usage_unavailable",
                                    stage="embedding_storage")
    chunk_count = sum(len(chunks) for _, chunks in documents)
    message = f"✅ 처리 완료: {len(documents)}개 파일 · {page_count}페이지 · {chunk_count}개 청크"
    if failures:
        request_log.current().status = "partial"
        message += f"\n\n⚠️ 제외된 파일 {len(failures)}개\n{failure_lines}"
    return db, message


# ===== Step 2. 질문 처리 (feature/qa-chain) =====
# 이 구역의 역할: 사용자가 질문을 입력하면
#   (1) 질문을 다듬고 → (2) PDF에서 관련 내용을 찾고 → (3) 가장 관련 있는 내용만 고른 뒤
#   → (4) GPT가 그 내용을 근거로 답을 만들어 화면에 조금씩 보여준다.
# Step 2에서 쓰는 import는 이 구역에 모아 둡니다. (다른 Step과 merge 충돌 방지)

# ---------- 파이썬 기본 라이브러리 ----------
import re                        # 정규표현식: 파일 경로에서 파일 이름만 잘라낼 때 사용
import time                      # 답변에 걸린 시간(초)을 잴 때 사용
from pathlib import Path         # 파일 경로를 운영체제(Windows/Linux)에 상관없이 다루는 도구

# ---------- 외부 패키지 (requirements.txt로 설치됨) ----------
import gradio as gr              # 웹 화면 라이브러리. 여기서는 gr.Error로 화면에 오류 메시지를 띄우는 데 사용
import openai                    # OpenAI 공식 라이브러리. API 키 오류 등 '오류 종류'를 구분하는 데 사용
from langchain.retrievers.document_compressors import LLMListwiseRerank  # GPT가 검색 결과의 순위를 다시 매겨주는 도구
from langchain.retrievers.multi_query import MultiQueryRetriever         # 질문 하나를 여러 표현으로 바꿔 여러 번 검색하는 도구
from langchain_core.documents import Document                                     # 페이지 단위로 이어 붙인 근거를 담는 문서 객체
from langchain_core.output_parsers import StrOutputParser                        # GPT 응답 객체에서 '글자(문자열)'만 꺼내주는 도구
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate            # GPT에게 보낼 지시문(프롬프트) 틀을 만드는 도구
from langchain_openai import ChatOpenAI                                          # LangChain에서 GPT 모델을 부르는 도구

# ---------- 설정값 (숫자만 바꾸면 동작이 바뀜) ----------
MODEL_NAME = "gpt-4o-mini"       # 사용할 GPT 모델 이름
TOP_K = 4                        # 최종적으로 GPT에게 넘겨줄 PDF 조각 개수
MAX_CONTEXT_CHARS = 10000        # 페이지 단위로 넓힌 근거 글자 수 상한 (비용·속도 보호)
MAX_QUESTION_LEN = 500           # 질문 최대 글자 수 (너무 긴 질문은 비용이 커서 막음)
MAX_HISTORY_TURNS = 3            # 후속 질문을 이해하려고 참고할 '최근 대화' 개수 (질문+답변 한 쌍이 1개)
CACHE_SIZE = 100                 # 답변 캐시(임시 저장소)에 보관할 최대 질문 수
LOG_PATH = Path("eval/qa_log.jsonl")  # 질문별 성능 기록을 저장할 파일 위치

_answer_cache = {}               # 답변 캐시: {(PDF 컬렉션 이름, 질문): 답변} 형태로 저장. 앱을 끄면 사라짐
_db_files_cache = {}             # DB별 등록 파일 이름 목록: {PDF 컬렉션 이름: [파일명, ...]}

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
     "3. 수치는 문서 표현 그대로 쓰고, 근거 문장 끝에 (p.페이지) 형식으로 표시하세요. "
     "[문서]에 파일이 여럿이면 (파일명 p.페이지)로 표시하세요.\n"                    # 출처 표시
     "4. 핵심부터 간결하게 답하세요.\n"                                           # 답변 스타일
     "5. 표: 행(항목)과 열(연도·기간)을 맞춰 읽고, 비교·증감 질문은 해당 수치를 직접 비교해 답하세요. "
     "표에 필요한 수치가 있으면 '확인할 수 없다'고 하지 마세요. 일부만 확인되면 확인되는 부분은 답하고 "
     "확인되지 않는 부분만 그렇다고 밝히세요.\n"                                    # 표 비교 (#37)
     "6. 목록: '모두', '구성', '3대', 'N개'처럼 전체를 묻거나 문서가 개수를 제시하면 문서의 항목을 하나도 빠뜨리지 말고 "
     "원문 명칭 그대로 나열하세요. 문서에 없는 항목은 만들지 마세요.\n"               # 목록 완전성 (#37)
     "7. 공통점: 여러 대상의 '공통' 특징을 물으면 모든 대상에 실제로 나타나는 것만 공통이라고 하세요. "
     "일부 대상에만 해당하면 대상별로 구분해 쓰고, 공통 요소가 없으면 없다고 답하세요.\n"  # 잘못된 일반화 방지 (#37)
     "8. 표가 글자로 풀려 열(대상) 구분이 불분명하면 단정하지 말고 그 한계를 밝히세요.\n"
     "9. 여러 문서: 문서·보고서끼리 비교하거나 '각각' 묻는 질문은 [문서]의 [파일명 p.N] 표시로 문서를 구분해 "
     "문서별 근거를 먼저 정리한 뒤 비교하세요. 질문한 문서마다 답하고, 특정 문서에 근거가 없으면 그 문서만 "
     "확인할 수 없다고 밝히세요. 대상·기간·기준이 다른 수치는 직접 비교의 한계를 밝히고, 문서에 없는 "
     "인과관계나 연결은 만들지 마세요.\n\n"                                         # 여러 문서 비교·종합 (#59)
     "[문서]\n{context}"),                                                        # {context}에 검색된 PDF 내용이 들어감
    ("human", "{question}"),                                                      # {question}에 사용자 질문이 들어감
])


def _page_label(meta):
    """PDF 조각의 정보(meta)에서 사람이 보는 페이지 번호를 꺼냄.
    README 기준대로 실제 PDF 페이지 번호(page + 1)를 우선함 (#28). 인쇄된 라벨('iv' 등)은 page가 없을 때만 사용."""
    if isinstance(meta.get("page"), int):   # 컴퓨터식 페이지 번호(0부터 시작)가 있으면
        return str(meta["page"] + 1)        #   사람식(1부터 시작)으로 바꾸려고 1을 더함
    if meta.get("page_label"):              # 없으면 PDF에 인쇄된 페이지 번호(예: 'iv', '12')를 사용
        return str(meta["page_label"])
    return "?"                              # 페이지 정보가 아예 없으면 물음표 표시


def _source_name(meta):
    """PDF 조각의 원본 파일 이름. build_db가 저장한 source_filename을 우선하고, 없으면 source 경로에서 꺼냄 (#28)."""
    if meta.get("source_filename"):
        return str(meta["source_filename"])
    # 경로 'C:\data\report.pdf' 또는 '/tmp/report.pdf'에서 'report.pdf'만 꺼냄 (\ 와 / 모두 처리)
    return re.split(r"[\\/]", meta.get("source", ""))[-1] or "문서"


def _format_docs(docs):
    """검색된 PDF 조각들을 GPT에게 보낼 하나의 글로 합침. 조각마다 [파일명 p.페이지]를 앞에 붙임.
    여러 PDF를 함께 등록했을 때 GPT가 어느 내용이 어느 문서인지 구분할 수 있게 파일명을 넣음 (#59)."""
    return "\n\n".join(                                         # 조각 사이를 빈 줄로 구분해서 이어 붙임
        f"[{_source_name(d.metadata)} p.{_page_label(d.metadata)}] {d.page_content}"  # 예: "[report.pdf p.3] 2026년 성장률은 ..."
        for d in docs                                           # 검색된 조각을 하나씩 처리
    )


def _format_sources(docs):
    """답변 아래에 붙일 '참고 페이지' 문구를 만듦. 예: '참고: report.pdf (p.2, 5, 7)'"""
    pages = {}                                                  # {파일이름: [페이지 목록]}을 담을 빈 상자
    for d in docs:                                              # 조각을 하나씩 보면서
        name = _source_name(d.metadata)                         # 원본 파일 이름 (예: report.pdf)
        label = _page_label(d.metadata)                         # 이 조각의 페이지 번호
        if label not in pages.setdefault(name, []):             # 같은 파일·같은 페이지가 이미 있으면 건너뜀 (중복 제거)
            pages[name].append(label)                           # 처음 나온 페이지만 목록에 추가
    sort_key = lambda p: (0, int(p)) if p.isdigit() else (1, p)  # 숫자 페이지는 숫자 순서로, 'iv' 같은 건 뒤로 정렬
    parts = [f"{name} (p.{', '.join(sorted(ps, key=sort_key))})"  # 파일마다 "report.pdf (p.2, 5)" 형태로 만듦
             for name, ps in pages.items()]
    return "\n\n---\n**참고**: " + " / ".join(parts)              # 구분선(---) 아래에 굵은 글씨 '참고'로 표시


def _rerank(llm, docs, query, top_n=TOP_K):
    """[업그레이드 1] 검색된 조각들 중 질문과 가장 관련 있는 top_n개를 GPT가 골라 순서를 다시 매김."""
    if len(docs) <= top_n:                                      # 조각이 이미 top_n개 이하면 고를 필요가 없으니
        return docs                                             #   그대로 돌려줌 (API 호출 절약)
    try:
        with request_log.stage("rerank"):
            reranker = LLMListwiseRerank.from_llm(llm=llm, top_n=top_n)
            return list(reranker.compress_documents(docs, query)) or docs[:top_n]
    except Exception:                                           # 재정렬 중 어떤 오류가 나도
        if request_log.current():
            request_log.current().emit("rerank_fallback", status="fallback")
        return docs[:top_n]                                     #   답변은 계속되도록 원래 순서의 앞 top_n개를 사용


def _db_files(db):
    """DB에 등록된 원본 파일 이름 목록 (컬렉션별로 한 번만 조회)."""
    key = _cache_key(db)
    if key not in _db_files_cache:
        try:
            metadatas = db.get(include=["metadatas"])["metadatas"]
            _db_files_cache[key] = sorted({_source_name(m) for m in metadatas if m})
        except Exception:
            return []
    return _db_files_cache[key]


def _chunk_key(doc):
    return doc.metadata.get("chunk_id") or (_source_name(doc.metadata), doc.metadata.get("page"), doc.page_content)


def _add_file_candidates(db, docs, query, files):
    """[#59] 여러 PDF가 등록돼 있으면 파일마다 관련 조각을 2개씩 더 찾아 후보에 넣음.
    전체 검색만 하면 내용이 많은 한 문서가 후보를 독차지해 다른 문서가 비교·종합에서 빠지기 때문."""
    seen = {_chunk_key(d) for d in docs}
    candidates = list(docs)
    for name in files:
        try:
            found = db.max_marginal_relevance_search(query, k=2, fetch_k=8, filter={"source_filename": name})
        except Exception:
            continue
        for doc in found:
            if _chunk_key(doc) not in seen:
                seen.add(_chunk_key(doc))
                candidates.append(doc)
    return candidates


def _cover_files(ranked, candidates, files):
    """[#59] 재정렬 상위 TOP_K개에 없는 파일이 있으면, 그 파일에서 가장 순위가 높은 조각을 1개씩 보충함.
    한 문서만 근거로 쓰여 '각각'·'비교' 질문에서 다른 문서가 빠지는 문제를 막음."""
    ordered = list(ranked) + [d for d in candidates if d not in ranked]
    picked = list(ranked[:TOP_K])
    covered = {_source_name(d.metadata) for d in picked}
    for doc in ordered[TOP_K:]:
        name = _source_name(doc.metadata)
        if name in files and name not in covered:
            picked.append(doc)
            covered.add(name)
    return picked


def _cache_key(db):
    """캐시에서 PDF DB를 구분하는 열쇠. id(db)는 DB 해제 후 재사용될 수 있어 컬렉션 이름(UUID)을 사용함 (#24)"""
    return getattr(db, "_collection_name", None) or id(db)       # build_db가 만든 컬렉션 이름: pdf_<uuid>


def _cache_get(db, query):
    """[업그레이드 2] 같은 PDF(db)에 같은 질문(query)을 한 적이 있으면 저장해 둔 답변을 꺼냄. 없으면 None."""
    return _answer_cache.get((_cache_key(db), query))


def _cache_put(db, query, text):
    """[업그레이드 2] 새로 만든 답변을 캐시에 저장함."""
    if len(_answer_cache) >= CACHE_SIZE:                        # 캐시가 가득 찼으면
        _answer_cache.pop(next(iter(_answer_cache)))            #   가장 먼저 저장된(가장 오래된) 답변 1개를 지움
    _answer_cache[(_cache_key(db), query)] = text               # (PDF 컬렉션 이름, 질문)을 열쇠로 답변을 저장


def _cache_clear(db):
    """DB를 삭제·교체할 때 그 DB로 만든 답변을 캐시에서 지움 (#24)"""
    key = _cache_key(db)
    for cache_key in [k for k in list(_answer_cache) if k[0] == key]:
        _answer_cache.pop(cache_key, None)
    _db_files_cache.pop(key, None)                              # 등록 파일 목록 캐시도 정리 (#59)


def _log(**record):
    """Keep the original QA fields alongside request events."""
    trace = request_log.current()
    if trace:
        trace.emit("qa_result", **record)


def _expand_to_pages(docs):
    """[#37] 고른 조각을 그 조각이 속한 페이지 전체(레이아웃 보존 텍스트)로 바꿈.
    표·목록이 600자 청크 경계에서 잘려 일부만 GPT에 전달되는 문제를 막음.
    같은 페이지는 한 번만 넣고, 재정렬 순서를 유지하며, 전체 글자 수는 MAX_CONTEXT_CHARS를 넘지 않게 함."""
    expanded, seen, total = [], set(), 0
    for doc in docs:
        key = (_source_name(doc.metadata), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        page_text = doc.metadata.get("page_text") or doc.page_content   # 예전 DB처럼 page_text가 없으면 조각 그대로
        if expanded and total + len(page_text) > MAX_CONTEXT_CHARS:
            page_text = doc.page_content                                 # 상한을 넘으면 그 페이지는 원래 조각만
            if total + len(page_text) > MAX_CONTEXT_CHARS:
                break
        expanded.append(Document(page_content=page_text, metadata=doc.metadata))
        total += len(page_text)
    return expanded


def _log_documents(db, docs):
    trace = request_log.current()
    if trace:
        trace.emit("retrieval_result", collection=getattr(db, "_collection_name", None),
                   documents=[dict(filename=_source_name(d.metadata),
                                   page=_page_label(d.metadata),
                                   chunk_id=d.metadata.get("chunk_id") or getattr(d, "id", None))
                              for d in docs])


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
    """Stream an answer while logging one request across Gradio worker contexts."""
    if request_log.current():
        yield from _answer(question, openai_key, db, history)
    else:
        with request_log.request(LOG_PATH, "qa", secrets=(openai_key,), activate=False) as trace:
            yield from request_log.iterate(trace, _answer(question, openai_key, db, history))


def _answer(question, openai_key, db, history=None):
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

    trace = request_log.current()
    trace.emit("qa_context", collection=getattr(db, "_collection_name", None), model=MODEL_NAME)
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=openai_key,
                     stream_usage=True, callbacks=[request_log.TokenUsageCallback(trace, MODEL_NAME)])
    start = time.perf_counter()                                 # [업그레이드 3] 시간 측정 시작 (스톱워치 누르기)

    try:                                                        # 아래에서 OpenAI 오류가 나면 맨 아래 except에서 처리
        # ===== 2단계: 후속 질문이면 혼자서도 이해되는 질문으로 바꾸기 =====
        history_text = _history_to_text(history)                # 대화 기록을 글로 변환
        search_question = question                              # 기본값: 사용자가 쓴 질문 그대로 검색
        if history_text:                                        # 이전 대화가 있으면
            with request_log.stage("question_rewrite"):
                search_question = (CONDENSE_PROMPT | llm | StrOutputParser()).invoke(
                    {"history": history_text, "question": question}
                ).strip() or question

        # ===== 3단계: [업그레이드 2] 캐시 확인 =====
        cached = _cache_get(db, search_question)                # 같은 질문에 대한 저장된 답변이 있는지 확인
        if cached:                                              # 있으면
            trace.status = "cached"
            _log(question=question, search_question=search_question, cached=True,  # 캐시로 답했다고 기록
                 seconds=round(time.perf_counter() - start, 2))  # 걸린 시간(초, 소수점 둘째 자리까지)
            yield cached                                        # 저장된 답변을 바로 화면에 보여주고
            return                                              # 함수 끝 (검색·GPT 호출 생략 → 빠르고 비용 0)

        # ===== 4단계: PDF에서 관련 내용 검색 =====
        base_retriever = db.as_retriever(                       # 벡터 DB를 '검색기'로 변환
            search_type="mmr",                                  # MMR: 비슷한 내용만 몰리지 않게 다양한 조각을 고르는 방식
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 5}    # 후보 20개를 먼저 찾고 그중 다양하게 4개 선택
        )
        retriever = MultiQueryRetriever.from_llm(               # 멀티쿼리 검색기: 질문을 3개로 바꿔 각각 검색
            retriever=base_retriever,                           # 실제 검색은 위의 MMR 검색기가 담당
            llm=llm,                                            # 질문 3개를 만드는 데 GPT 사용
            prompt=MULTI_QUERY_PROMPT,                          # 한국어 질문을 만들도록 직접 만든 프롬프트 사용
            include_original=True,                              # 바꾼 질문 3개 + 원래 질문까지 총 4번 검색
        )
        try:
            with request_log.stage("retrieval"):
                docs = retriever.invoke(search_question)[: TOP_K * 2]
                _log_documents(db, docs)
        finally:
            trace.usage("text-embedding-3-large", reason="embedding_adapter_usage_unavailable",
                        stage="retrieval_embeddings")
        files = _db_files(db)
        if len(files) > 1:                                      # [#59] 여러 PDF: 파일별 후보 추가 → 전체 순위 → 빠진 파일 보충
            candidates = _add_file_candidates(db, docs, search_question, files)
            # top_n을 후보 수보다 작게 해야 실제로 재정렬됨 (같거나 크면 _rerank가 재정렬을 건너뜀)
            ranked = _rerank(llm, candidates, search_question, top_n=TOP_K + len(files))
            docs = _cover_files(ranked, candidates, files)
        else:
            docs = _rerank(llm, docs, search_question)          # [업그레이드 1] 8개 중 가장 관련 있는 4개로 추림
        with request_log.stage("selected_context"):
            _log_documents(db, docs)
        with request_log.stage("page_context"):                 # [#37] 고른 조각을 페이지 전체로 넓힘
            docs = _expand_to_pages(docs)
            if trace:
                trace.emit("page_context", pages=[_page_label(d.metadata) for d in docs],
                           chars=sum(len(d.page_content) for d in docs))

        if not docs:                                            # 관련 내용을 하나도 못 찾았으면
            trace.status = "no_results"
            yield "업로드한 문서에서 관련 내용을 찾지 못했습니다. 질문을 바꿔서 다시 시도해 보세요."  # 안내 문구 표시
            return                                              # 함수 끝

        # ===== 5단계: GPT로 답변 만들기 (조금씩 화면에 표시) =====
        chain = ANSWER_PROMPT | llm | StrOutputParser()         # 프롬프트 → GPT → 글자만 꺼내기를 하나로 연결
        text = ""                                               # 지금까지 받은 답변을 모아 둘 변수
        with request_log.stage("answer_generation"):
            for chunk in chain.stream(
                {"context": _format_docs(docs), "question": search_question}
            ):
                text += chunk
                yield text

        # ===== 6단계: 참고 페이지 붙이기 =====
        final = text + _format_sources(docs)                    # 완성된 답변 아래에 '참고: report.pdf (p.2, 5)' 추가
        yield final                                             # 최종 답변을 화면에 표시

        # ===== 7단계: [업그레이드 2, 3] 캐시 저장 + 성능 기록 =====
        _cache_put(db, search_question, final)                  # 다음에 같은 질문이 오면 바로 답하도록 저장
        _log(question=question, search_question=search_question, cached=False,  # 새로 답변했다고 기록
             seconds=round(time.perf_counter() - start, 2),     # 전체 걸린 시간(초)
             pages=[_page_label(d.metadata) for d in docs])     # 답변 근거로 쓴 페이지 목록

    # ===== 오류 처리: OpenAI 오류를 사용자가 이해할 수 있는 한국어로 안내 =====
    except openai.AuthenticationError as error:                          # API 키가 틀렸거나 만료된 경우
        trace.error(error)
        raise gr.Error("API 키가 올바르지 않습니다. 키를 다시 확인하세요.")
    except openai.RateLimitError as error:                               # 너무 자주 요청했거나 크레딧(사용 한도)을 다 쓴 경우
        trace.error(error)
        raise gr.Error("OpenAI 사용 한도를 초과했습니다. 잠시 후 다시 시도하세요.")
    except openai.APIConnectionError as error:                           # 인터넷 연결 문제 등으로 OpenAI 서버에 접속 못 한 경우
        trace.error(error)
        raise gr.Error("OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인하세요.")


# ===== Step 3. 화면 (feature/chat-ui) =====

import os
import threading
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv


# ==========================================
# 1. 환경변수 및 세션 관리
# ==========================================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_session_dbs = {}
_session_locks = {}
_sessions_guard = threading.Lock()


# ==========================================
# 2. CSS 디자인 + 다크모드
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

html, body {
    background: #FFFFFF !important;
}

/* 다크모드 */

html:has(#mode-select input[type="checkbox"]:checked),
body:has(#mode-select input[type="checkbox"]:checked) {
    background: #11141C !important;
}

.gradio-container:has(#mode-select input[type="checkbox"]:checked) {
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
    word-break: keep-all;   /* 한글 단어 중간에서 줄바꿈하지 않음 (#40) */
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
        90deg, #329AFF, #8578FF, #FF65AF
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

/* PDF 업로드 */

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

/* 질문창 안내 문구도 단어 단위로 줄바꿈 (#40) */
#question-box textarea {
    word-break: keep-all;
}

/* 버튼 글자는 한 줄로 표시 (#40) */
#pdf-select,
#upload-btn,
#reset-select-btn {
    white-space: nowrap !important;
    word-break: keep-all !important;
}

#upload-status,
#upload-status * {
    color: var(--muted) !important;
    font-size: 12px !important;
}

#reset-select-btn {
    align-self: flex-start !important;
    width: auto !important;
    min-width: 0 !important;
    background: var(--field) !important;
    color: var(--muted) !important;
    border: 1px solid var(--line) !important;
    border-radius: 9px !important;
    box-shadow: none !important;
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
    min-height: 60px !important;
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

/* 질문할 수 없는 상태(PDF 미등록·처리 중·등록 실패)를 눈에 띄게 표시 (#55) */
#question-box:has(textarea:disabled) {
    border-style: dashed !important;
    border-color: var(--muted) !important;
}

#question-box textarea:disabled::placeholder {
    color: var(--muted) !important;
    opacity: 1 !important;
}

#question-box textarea:disabled {
    cursor: not-allowed !important;
}

#ask-btn:disabled,
#ask-btn[disabled] {
    background: var(--line) !important;
    color: var(--fg) !important;
    opacity: 0.85 !important;
    box-shadow: none !important;
    filter: none !important;
    cursor: not-allowed !important;
}

/* 채팅 영역 */

#chat-box {
    background: var(--answer) !important;
    border: 1px solid var(--line) !important;
    border-left: 4px solid #827AFF !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

#chat-box .message {
    border: 1px solid var(--line) !important;
}

#chat-box .message.user {
    background: var(--field) !important;
}

#chat-box .message.bot,
#chat-box .message.assistant {
    background: var(--card) !important;
}

#chat-box,
#chat-box * {
    color: var(--fg) !important;
}

/* 초기화 버튼 */

#clear-btn {
    background: var(--answer) !important;
    color: var(--fg) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    min-height: 40px !important;
    box-shadow: none !important;
}

/* 모바일 */

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

    /* 좁은 화면에서는 버튼 글자·여백·최소 폭을 줄여 한 줄 유지, 카드 밖으로 넘치지 않게 (#40) */
    #pdf-select,
    #upload-btn {
        font-size: 13px !important;
        padding-left: 8px !important;
        padding-right: 8px !important;
        min-width: 0 !important;
    }
}

/* 아주 좁은 폰(400px 이하): 카드 안쪽 여백 축소 (#40) */
@media (max-width: 400px) {
    #main-card {
        padding: 14px 12px !important;
    }
}
"""

# Gradio는 css= 규칙 앞에 '.gradio-container ... .contain'을 자동으로 붙여서
# 카드 바깥 요소(.main)에는 적용되지 않음 → head로 전역 스타일을 넣어
# 아주 좁은 폰에서 Gradio 기본 좌우 여백(32px)을 없앰 (#40)
HEAD_STYLE = """
<style>
@media (max-width: 400px) {
    .gradio-container .main,
    .gradio-container .fillable {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
}
</style>
"""


# ==========================================
# 3. 세션 및 DB 관리 함수
# ==========================================

def _session_lock(request):
    with _sessions_guard:
        return _session_locks.setdefault(
            request.session_hash,
            threading.Lock()
        )


def _delete_db(db):
    if db is None:
        return

    _cache_clear(db)

    try:
        db.delete_collection()
    except Exception as error:
        trace = request_log.current()
        if trace:
            trace.error(error, cleanup=True)
        else:
            with request_log.request(LOG_PATH, "db_cleanup", secrets=(OPENAI_API_KEY,)) as trace:
                trace.status = "error"
                trace.error(error)
        logging.getLogger(__name__).error("Chroma 컬렉션 삭제 실패: %s", trace.redact(str(error)))


# ==========================================
# 4. PDF 다중 업로드
# ==========================================

def _selection_status(paths):
    if not paths:
        return "📁 선택한 PDF가 없습니다."

    return (
        f"✅ 총 {len(paths)}개 PDF 선택 완료\n\n"
        + "\n".join(
            f"- 📄 {Path(path).name}"
            for path in paths
        )
        + "\n\n다시 선택하면 목록에 추가됩니다. 비우려면 '선택 초기화'를 누르세요."
    )


def select_pdf(files, current_paths=None):
    # 다시 선택하면 기존 목록에 추가 (#41). 같은 이름의 파일은 새로 고른 것으로 교체
    if not files:
        return current_paths, _selection_status(current_paths)

    if not isinstance(files, list):
        files = [files]

    selected = {
        Path(path).name: path
        for path in (current_paths or [])
    }

    for f in files:
        path = f if isinstance(f, str) else f.name
        selected[Path(path).name] = path

    paths = list(selected.values())

    return paths, _selection_status(paths)


def clear_selection():
    # 선택한 PDF 목록 비우기 (#41). 이미 등록된 DB는 그대로 두고 다음 등록 대상만 초기화
    return None, _selection_status(None)


# 질문창 안내 문구: 질문할 수 없는 이유를 상태별로 보여줌 (#55)
QUESTION_READY = "💬 여러 PDF 문서에 대해 질문하세요..."
QUESTION_NO_PDF = "🔒 먼저 PDF를 선택하고 '파일 등록'을 눌러야 질문할 수 있습니다."
QUESTION_PROCESSING = "⏳ PDF 처리 중입니다. 완료되면 질문할 수 있습니다."
QUESTION_NOT_REGISTERED = "🔒 등록된 PDF가 없어 질문할 수 없습니다. 위 안내를 확인하고 PDF를 다시 등록해 주세요."
REGISTER_FAILED_NOTE = "\n\n🔒 PDF 등록에 성공해야 질문할 수 있습니다."


def start_register(request: gr.Request):

    with _session_lock(request):
        _delete_db(
            _session_dbs.pop(request.session_hash, None)
        )

    locked = gr.update(interactive=False)

    return (
        None,
        QUESTION_PROCESSING,
        gr.update(interactive=False, placeholder=QUESTION_PROCESSING),
        locked,
        locked,
        locked,
        locked
    )


def register_pdf(pdf_paths, request: gr.Request):

    if not pdf_paths:
        with request_log.request(LOG_PATH, "pdf_registration", secrets=(OPENAI_API_KEY,)) as trace:
            trace.status = "empty"
        return None, "⚠️ PDF 파일을 선택해주세요." + REGISTER_FAILED_NOTE

    if not OPENAI_API_KEY:
        with request_log.request(LOG_PATH, "pdf_registration") as trace:
            trace.status = "error"
            trace.emit("validation_failed", reason="missing_api_key")
        return (
            None,
            "⚠️ OPENAI_API_KEY가 없습니다. .env 파일을 확인해주세요." + REGISTER_FAILED_NOTE
        )

    # 여러 PDF를 하나의 DB로 처리
    db, message = build_db(pdf_paths, OPENAI_API_KEY)

    if db is not None:
        with _session_lock(request):
            _session_dbs[request.session_hash] = db
    else:
        message += REGISTER_FAILED_NOTE                          # 질문이 막힌 이유 안내 (#55)

    return db, message


def finish_register(db, message):
    # 결과 문구와 질문창 활성화를 한 번에 바꿔 '완료됐는데 못 쓰는' 순간이 없게 함 (#55)
    can_ask = db is not None
    placeholder = QUESTION_READY if can_ask else QUESTION_NOT_REGISTERED

    return (
        message,
        gr.update(interactive=can_ask, placeholder=placeholder),
        gr.update(interactive=can_ask),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True)
    )


# ==========================================
# 5. 질문 및 스트리밍 답변
# ==========================================

def ask_question(question, history, db, request: gr.Request):
    with request_log.request(LOG_PATH, "qa", secrets=(OPENAI_API_KEY,), activate=False) as trace:
        yield from request_log.iterate(trace, _ask_question(question, history, db, request))


def _ask_question(question, history, db, request):

    history = list(history or [])
    prior_history = list(history)

    if not question or not question.strip():
        request_log.current().status = "error"
        request_log.current().emit("validation_failed", reason="empty_question")
        history.append({
            "role": "assistant",
            "content": "⚠️ 질문을 입력해주세요."
        })
        yield history, ""
        return

    history.append({
        "role": "user",
        "content": question
    })

    with _session_lock(request):

        if db is None or db is not _session_dbs.get(request.session_hash):
            request_log.current().status = "error"
            request_log.current().emit("validation_failed", reason="missing_or_replaced_db")
            history.append({
                "role": "assistant",
                "content": "⚠️ PDF를 선택한 뒤 파일 등록 버튼을 눌러주세요."
            })
            yield history, ""
            return

        history.append({
            "role": "assistant",
            "content": "🤖 답변을 생성하는 중..."
        })
        yield history, ""

        try:
            result = answer(
                question,
                OPENAI_API_KEY,
                db,
                prior_history
            )

            if isinstance(result, str):
                history[-1]["content"] = result
                yield history, ""

            else:
                for text in result:
                    history[-1]["content"] = str(text)
                    yield history, ""

        except Exception as e:
            request_log.current().status = "error"
            request_log.current().error(e)
            history[-1]["content"] = f"⚠️ 답변 처리 오류: {e}"
            yield history, ""


def clear_chat():
    return []


def cleanup_session(request: gr.Request):

    with _session_lock(request):
        _delete_db(
            _session_dbs.pop(request.session_hash, None)
        )

    with _sessions_guard:
        _session_locks.pop(request.session_hash, None)


# ==========================================
# 6. Gradio Blocks UI
# ==========================================

with gr.Blocks(title="문서 질의응답 서비스") as demo:

    db_state = gr.State(value=None)
    selected_files = gr.State(value=None)
    register_message = gr.State(value="")

    with gr.Column(elem_id="main-card"):

        # 제목 + 다크모드
        with gr.Row(elem_id="title-row"):

            with gr.Column(scale=5, min_width=180):
                gr.HTML("""
                <div class="page-title">
                    📚 문서 질의응답 서비스
                </div>
                """)

            with gr.Column(scale=1, min_width=110):
                dark_mode = gr.Checkbox(
                    label="🌙 다크모드",
                    value=False,
                    container=False,
                    elem_id="mode-select"
                )

        gr.HTML('<div class="gradient-line"></div>')

        # PDF 다중 업로드
        gr.HTML(
            '<div class="section-label">📚 PDF 다중 문서 업로드</div>'
        )

        with gr.Row(elem_id="file-row"):

            pdf_file = gr.UploadButton(
                "📎 PDF 여러 개 선택",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath",
                variant="secondary",
                scale=5,
                min_width=160,
                elem_id="pdf-select"
            )

            upload_btn = gr.Button(
                "📂 파일 등록",
                variant="secondary",
                scale=2,
                min_width=110,
                elem_id="upload-btn"
            )

        status = gr.Markdown(
            "📁 등록된 문서가 없습니다. PDF를 선택하고 '파일 등록'을 누르면 질문할 수 있습니다.",
            elem_id="upload-status"
        )

        reset_select_btn = gr.Button(
            "🧹 선택 초기화",
            variant="secondary",
            size="sm",
            elem_id="reset-select-btn"
        )

        # 질문
        gr.HTML(
            '<div class="section-label">💬 질문</div>'
        )

        question = gr.Textbox(
            placeholder=QUESTION_NO_PDF,
            show_label=False,
            container=False,
            lines=1,
            max_lines=5,
            interactive=False,
            elem_id="question-box"
        )

        ask_btn = gr.Button(
            "🚀 질문하기 →",
            variant="primary",
            interactive=False,
            elem_id="ask-btn"
        )

        # AI 대화
        gr.HTML(
            '<div class="section-label">✨ AI 대화</div>'
        )

        chatbot = gr.Chatbot(
            placeholder="🤖 질문을 입력하면 이곳에 대화가 쌓입니다.",
            show_label=False,
            layout="bubble",
            height=400,
            elem_id="chat-box"
        )

        # 대화 초기화
        clear_btn = gr.Button(
            "🗑️ 대화 초기화",
            variant="secondary",
            elem_id="clear-btn"
        )

    # ======================================
    # 7. 이벤트 핸들러
    # ======================================

    pdf_file.upload(
        fn=select_pdf,
        inputs=[pdf_file, selected_files],
        outputs=[selected_files, status]
    )

    reset_select_btn.click(
        fn=clear_selection,
        outputs=[selected_files, status]
    )

    upload_btn.click(
        fn=start_register,
        outputs=[
            db_state,
            status,
            question,
            ask_btn,
            pdf_file,
            upload_btn,
            reset_select_btn
        ]
    ).then(
        fn=register_pdf,
        inputs=[selected_files],
        outputs=[db_state, register_message]
    ).then(
        fn=finish_register,
        inputs=[db_state, register_message],
        outputs=[
            status,
            question,
            ask_btn,
            pdf_file,
            upload_btn,
            reset_select_btn
        ]
    )

    ask_btn.click(
        fn=ask_question,
        inputs=[question, chatbot, db_state],
        outputs=[chatbot, question]
    )

    question.submit(
        fn=ask_question,
        inputs=[question, chatbot, db_state],
        outputs=[chatbot, question]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=chatbot
    )

    demo.unload(cleanup_session)


# ==========================================
# 8. 서버 실행
# ==========================================

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        head=HEAD_STYLE,
        theme=gr.themes.Soft()
    )
