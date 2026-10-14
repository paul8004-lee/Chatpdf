# ===== Step 1. PDF 처리 (feature/pdf-processing) =====
import uuid
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


# ===== Step 2. 질문 처리 (feature/qa-chain) =====


# ===== Step 3. 화면 (feature/chat-ui) =====

