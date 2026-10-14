# ===== Step 1. PDF 처리 (feature/pdf-processing) =====


# ===== Step 2. 질문 처리 (feature/qa-chain) =====
# Step 2에서 쓰는 import는 이 구역에 모아 둡니다. (다른 Step과 merge 충돌 방지)
import re

import gradio as gr
import openai
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-4o-mini"
TOP_K = 4                # 검색할 문서 조각 수
MAX_QUESTION_LEN = 500   # 질문 최대 길이
MAX_HISTORY_TURNS = 3    # 맥락으로 쓸 최근 대화 수

# 1) 멀티쿼리: 질문을 한국어 검색어 3개로 바꿔서 검색 누락을 줄임
MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    "당신은 문서 검색을 돕는 도우미입니다.\n"
    "아래 질문을 벡터 검색에 잘 걸리도록 서로 다른 관점의 한국어 질문 3개로 바꾸세요.\n"
    "한 줄에 하나씩, 번호 없이 질문만 출력하세요.\n\n"
    "원래 질문: {question}"
)

# 2) 후속 질문 재작성: "그거 더 자세히" 같은 질문을 혼자서도 이해되는 질문으로 바꿈
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "이전 대화를 참고해서 마지막 질문을 그 자체로 이해되는 하나의 질문으로 다시 쓰세요. "
     "답하지 말고 다시 쓴 질문만 출력하세요."),
    ("human", "이전 대화:\n{history}\n\n마지막 질문: {question}"),
])

# 3) 답변: 문서 근거로만 답하고, 근거가 없으면 모른다고 말하게 함 (hub.pull 대신 직접 정의)
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 현대경제연구원 보고서를 바탕으로 답하는 어시스턴트입니다.\n"
     "규칙:\n"
     "1. 아래 [문서] 내용만 근거로 한국어로 답하세요.\n"
     "2. 문서에 없는 내용은 추측하지 말고 '문서에서 확인할 수 없습니다'라고 답하세요.\n"
     "3. 수치는 문서 표현 그대로 쓰고, 근거 문장 끝에 (p.페이지) 형식으로 표시하세요.\n"
     "4. 핵심부터 간결하게 답하세요.\n\n"
     "[문서]\n{context}"),
    ("human", "{question}"),
])


def _page_label(meta):
    """PDF 메타데이터에서 사람이 보는 페이지 번호를 꺼냄 (PyPDFLoader의 page는 0부터 시작)."""
    if meta.get("page_label"):
        return str(meta["page_label"])
    if isinstance(meta.get("page"), int):
        return str(meta["page"] + 1)
    return "?"


def _format_docs(docs):
    """검색된 조각마다 출처를 붙여서 LLM에 넘길 문맥으로 만듦."""
    return "\n\n".join(
        f"[p.{_page_label(d.metadata)}] {d.page_content}" for d in docs
    )


def _format_sources(docs):
    """답변 아래에 붙일 참고 페이지 목록. 파일별로 묶고 페이지 순으로 정렬."""
    pages = {}
    for d in docs:
        # Windows(\\)와 Linux(/) 경로 모두에서 파일 이름만 꺼냄
        name = re.split(r"[\\/]", d.metadata.get("source", ""))[-1] or "문서"
        label = _page_label(d.metadata)
        if label not in pages.setdefault(name, []):
            pages[name].append(label)
    sort_key = lambda p: (0, int(p)) if p.isdigit() else (1, p)
    parts = [f"{name} (p.{', '.join(sorted(ps, key=sort_key))})" for name, ps in pages.items()]
    return "\n\n---\n**참고**: " + " / ".join(parts)


def _history_to_text(history):
    """Gradio 대화 기록(messages 형식/튜플 형식 모두)을 텍스트로 변환."""
    if not history:
        return ""
    lines = []
    for item in history[-MAX_HISTORY_TURNS * 2:]:
        if isinstance(item, dict):  # {"role": "user", "content": "..."}
            role = "사용자" if item.get("role") == "user" else "AI"
            content = item.get("content", "")
            if isinstance(content, list):  # 멀티모달 형식이면 텍스트만
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            lines.append(f"{role}: {content}")
        elif isinstance(item, (list, tuple)) and len(item) == 2:  # [user, bot]
            lines.append(f"사용자: {item[0]}")
            lines.append(f"AI: {item[1]}")
    return "\n".join(lines)


def answer(question, openai_key, db, history=None):
    # --- 입력 검사 ---
    if db is None:
        raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")
    question = (question or "").strip()
    if not question:
        raise gr.Error("질문을 입력하세요.")
    if len(question) > MAX_QUESTION_LEN:
        raise gr.Error(f"질문은 {MAX_QUESTION_LEN}자 이내로 입력하세요.")

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=openai_key)

    try:
        # --- 후속 질문이면 대화 맥락을 반영해 질문 재작성 ---
        history_text = _history_to_text(history)
        search_question = question
        if history_text:
            search_question = (CONDENSE_PROMPT | llm | StrOutputParser()).invoke(
                {"history": history_text, "question": question}
            ).strip() or question

        # --- 검색: MMR(비슷한 조각만 몰리지 않게) + 멀티쿼리(원래 질문 포함) ---
        base_retriever = db.as_retriever(
            search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 5}
        )
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm,
            prompt=MULTI_QUERY_PROMPT, include_original=True,
        )
        docs = retriever.invoke(search_question)[: TOP_K * 2]

        if not docs:
            yield "업로드한 문서에서 관련 내용을 찾지 못했습니다. 질문을 바꿔서 다시 시도해 보세요."
            return

        # --- 답변 생성 (스트리밍) ---
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        text = ""
        for chunk in chain.stream({"context": _format_docs(docs), "question": search_question}):
            text += chunk
            yield text

        # --- 참고 페이지 표시 ---
        yield text + _format_sources(docs)

    except openai.AuthenticationError:
        raise gr.Error("API 키가 올바르지 않습니다. 키를 다시 확인하세요.")
    except openai.RateLimitError:
        raise gr.Error("OpenAI 사용 한도를 초과했습니다. 잠시 후 다시 시도하세요.")
    except openai.APIConnectionError:
        raise gr.Error("OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인하세요.")

# ===== Step 3. 화면 (feature/chat-ui) =====

