import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. PDF 로드 및 분할 함수 (파라미터 주입형)
def process_pdf_with_params(pdf_path: str, chunk_size: int, chunk_overlap: int):
    """지정된 chunk_size 및 chunk_overlap으로 문서를 분할하고 벡터 DB를 생성합니다."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY,
    )

    # 매 실험마다 컬렉션이 겹치지 않도록 임의의 이름 지정
    collection_name = f"test_{uuid.uuid4().hex[:8]}"
    
    # In-memory 형태의 임시 Chroma DB 생성 (실험 속도 및 용량 절약)
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings_model,
        collection_name=collection_name,
    )

    return db, len(pages), len(texts)


# 3. 실험 파이프라인 실행
def run_experiment(pdf_path: str, test_queries: list, configs: list):
    """
    여러 청킹 설정을 순회하며 검색 결과를 정량/정성 평가합니다.
    """
    print("=" * 60)
    print("🚀 청크 설정(chunk_size, chunk_overlap) 튜닝 실험 시작")
    print("=" * 60)

    results_summary = []

    for cfg in configs:
        c_size = cfg["size"]
        c_overlap = cfg["overlap"]
        label = cfg["label"]

        print(f"\n▶ [{label}] 실행 중... (Size: {c_size}, Overlap: {c_overlap})")
        db, page_count, chunk_count = process_pdf_with_params(pdf_path, c_size, c_overlap)
        print(f"   - 총 페이지 수: {page_count}장 | 생성된 청크 수: {chunk_count}개")

        hit_count = 0

        # 테스트 질문셋 평가
        for item in test_queries:
            query = item["question"]
            expected_keyword = item["keyword"]

            # Top-3 청크 검색
            retrieved_docs = db.similarity_search(query, k=3)
            retrieved_texts = " ".join([doc.page_content for doc in retrieved_docs])

            # 기대 키워드가 검색 결과 청크에 포함되어 있는지 확인
            is_hit = expected_keyword in retrieved_texts
            if is_hit:
                hit_count += 1

        hit_rate = (hit_count / len(test_queries)) * 100
        print(f"   - Top-3 정답 키워드 적중률(Hit Rate): {hit_rate:.1f}% ({hit_count}/{len(test_queries)})")

        results_summary.append({
            "label": label,
            "size": c_size,
            "overlap": c_overlap,
            "chunks": chunk_count,
            "hit_rate": hit_rate,
        })

    # 최종 결과 비교 출력
    print("\n" + "=" * 60)
    print("📊 [실험 최종 결과 비교 표]")
    print("=" * 60)
    print(f"{'설정':<12} | {'Size':<6} | {'Overlap':<8} | {'청크 수':<8} | {'Hit Rate':<10}")
    print("-" * 60)
    for res in results_summary:
        print(f"{res['label']:<12} | {res['size']:<6} | {res['overlap']:<8} | {res['chunks']:<8} | {res['hit_rate']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    # [1] 테스트할 PDF 경로 입력
    PDF_FILE_PATH = "1781500795407_e24e06c0fe42.pdf"  # <-- 가지고 계신 PDF 파일 이름/경로로 변경

    # [2] 실험할 하이퍼파라미터 조합 (보통 overlap은 size의 10~20%)
    TEST_CONFIGS = [
        {"label": "Small",  "size": 300,  "overlap": 50},
        {"label": "Medium", "size": 600,  "overlap": 100},
        {"label": "Large",  "size": 1000, "overlap": 150},
    ]

    # [3] 평가용 질문 및 정답 포함 키워드 (실제 문서 내용에 맞게 3~5개 설정)
    TEST_QUERIES = [
        {"question": "2025년 한국을 방문한 외국인 관광객은 몇 명인가요?", "keyword": "관광객"},
        {"question": "외국인 관광객 수가 증가했는데도 한국 관광산업의 질적 회복이 제한적이라고 평가하는 이유는 무엇인가요?", "keyword": "질적 회복"},
        {"question": "정부가 추진하는 ‘관광 혁신 3대 전략’의 핵심 목표와 세부 추진 방향은 무엇인가요?", "keyword": "관광 혁신 3대 전략"},
    ]

    # 실행
    run_experiment(PDF_FILE_PATH, TEST_QUERIES, TEST_CONFIGS)