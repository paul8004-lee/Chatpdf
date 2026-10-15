# ===== Step 1. PDF 처리 (feature/pdf-processing) =====
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. build_db 함수 정의
def build_db(pdf_path, openai_key):
    if not openai_key:
        return None, "⚠️ OpenAI API 키를 먼저 입력한 뒤 PDF를 다시 올려주세요."
    if pdf_path is None:
        return None, ""

    # PDF 원본 로드 (중복 split 방지를 위해 .load() 사용 권장)
    pages = PyPDFLoader(pdf_path).load()

    # 텍스트 분할 (청크 크기 및 오버랩)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # 임베딩 모델 생성
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_key,
    )

    # Chroma 벡터 DB 컬렉션 생성 및 청크 임베딩 저장
    collection_name = f"pdf_{uuid.uuid4().hex}"
    db = Chroma.from_documents(
        texts,
        embeddings_model,
        collection_name=collection_name
    )

    return db, f"✅ 처리 완료: {len(pages)}페이지 → {len(texts)}개 청크"

=======

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


# ===== Step 3. 화면 (feature/chat-ui) =====
