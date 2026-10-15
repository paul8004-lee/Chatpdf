import base64
import os
import uuid
import pymupdf as fitz  # 최신 권장 방식 (fitz 이름 유지)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class VisualElementChunker:
    """PDF 내에서 일반 텍스트는 건너뛰고,
    '표(Table)'와 '이미지/차트(Image)'만 전담으로 추출 및 요약하여 청킹하는 모듈
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        image_save_dir: str = "./extracted_images",
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.image_save_dir = image_save_dir
        os.makedirs(self.image_save_dir, exist_ok=True)

    def _summarize_image(self, image_path: str) -> str:
        """추출된 이미지/차트를 GPT-4o-mini로 분석해 검색용 요약 생성"""
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
        """표 데이터를 자연어 검색용 요약문으로 변환"""
        prompt = f"""
        다음 표의 구조와 수치 데이터를 분석하여 핵심 요약문을 작성하세요.
        사용자가 자연어로 검색할 때 이 표가 잘 검색될 수 있도록 주요 항목명, 수치, 결론을 포함하세요.

        [표 데이터]
        {table_markdown}
        """
        response = self.llm.invoke(prompt)
        return response.content

    def extract_visual_chunks(self, pdf_path: str) -> list[Document]:
        """PDF에서 '표'와 '이미지'만 추출하여 Document 리스트로 반환"""
        doc = fitz.open(pdf_path)
        visual_chunks: list[Document] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. 표(Table) 추출
            tabs = page.find_tables()
            for t_idx, table in enumerate(tabs):
                df = table.to_pandas()
                # 데이터가 빈 표이거나 행/열이 너무 작은 경우 제외
                if df.empty or df.shape[0] < 1:
                    continue

                table_md = df.to_markdown(index=False)
                table_summary = self._summarize_table(table_md)

                visual_chunks.append(
                    Document(
                        # 본문에는 요약문과 원본 표를 함께 두어 검색 및 답변 생성에 모두 활용
                        page_content=f"[표 데이터 요약]\n{table_summary}\n\n[원본 표]\n{table_md}",
                        metadata={
                            "element_type": "table",
                            "page": page_num,
                            "source": pdf_path,
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
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # 단순 아이콘, 구분선, 로고 등 노이즈 이미지 제거 (가로/세로 120px 미만 무시)
                if base_image["width"] < 120 or base_image["height"] < 120:
                    continue

                # 이미지 파일 저장
                img_filename = f"page_{page_num}_img_{img_idx}_{uuid.uuid4().hex[:6]}.{image_ext}"
                img_filepath = os.path.join(self.image_save_dir, img_filename)
                with open(img_filepath, "wb") as f:
                    f.write(image_bytes)

                # GPT Vision 요약
                img_summary = self._summarize_image(img_filepath)

                visual_chunks.append(
                    Document(
                        page_content=f"[이미지/차트 분석]\n{img_summary}",
                        metadata={
                            "element_type": "image",
                            "page": page_num,
                            "source": pdf_path,
                            "image_path": img_filepath,
                        },
                    )
                )

        doc.close()
        return visual_chunks