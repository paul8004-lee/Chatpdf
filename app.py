


# ===== Step 0. 공통 import (Step 2 answer 함수에서 사용) =====
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# langchain 버전에 따라 위치가 달라서 둘 다 시도
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain import hub
except Exception:  # 새 버전/파이썬 3.14에서는 ImportError 외 오류도 나서 전체를 잡음
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    from langchain_classic import hub


# ===== Step 1. PDF 처리 (feature/pdf-processing) =====


# ===== Step 2. 질문 처리 (feature/qa-chain) =====
def answer(question, openai_key, db):
    if db is None:
        raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")
    if not question:
        raise gr.Error("질문을 입력하세요.")

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), llm=llm
    )
    prompt = hub.pull("rlm/rag-prompt")
    generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
        | prompt
        | generate_llm
        | StrOutputParser()
    )

    text = ""
    for chunk in rag_chain.stream(question):
        text += chunk
        yield text

# ===== Step 3. 화면 (feature/chat-ui) =====

import gradio as gr  # [삭제 대상] Step 0에 이미 있음 (중복)
from pathlib import Path  # [유지] select_pdf 등에서 사용


# ==========================================
# 1. 테스트용 함수
# 실제 Step 1, Step 2 완성 후 삭제
# ==========================================

# [삭제 대상] 이 구역(1. 테스트용 함수) 전체: build_db, answer 두 함수 모두
# Step 1 담당자의 build_db(pdf_path)와 Step 2 담당자의 answer로 대체됨
def build_db(pdf_path):  # [삭제 대상] Step 1 담당자의 build_db로 대체
    if not pdf_path:
        return None, "PDF 파일을 선택해주세요."  # 올릴때는 지우고 (build_db, answer 둘 다)

    filename = Path(pdf_path).name

    return (
        {"filename": filename},
        f"✅ {filename} 등록 완료 (테스트 모드)"
    )


def answer(question, openai_key, db, history=None):  # [삭제 대상] Step 2의 answer로 대체
    if db is None:
        return "먼저 PDF 파일을 등록해주세요."

    return (
        f"**질문:** {question}\n\n"
        "현재는 UI 테스트 모드입니다. "
        "실제 RAG가 연결되면 PDF 내용을 기반으로 답변합니다."
    )


# ==========================================
# 2. CSS 디자인 + 라이트/다크 모드
# ==========================================

CSS = """
/* 라이트 모드 기본 색상 */

.gradio-container {
    --page: #ffffff;
    --card: #ffffff;
    --field: #ffffff;
    --answer: #f7f9ff;
    --fg: #202638;
    --muted: #738096;
    --line: #e3e8f1;

    background: var(--page) !important;
    color: var(--fg) !important;
    max-width: 720px !important;
    margin: 0 auto !important;
    padding: 30px 18px !important;
    min-height: 100vh;
    font-family: 'Pretendard', Arial, sans-serif !important;
}

html, body {
    background: #ffffff !important;
}

/* 다크모드 활성화 */

html:has(#mode-select input[type="checkbox"]:checked),
body:has(#mode-select input[type="checkbox"]:checked) {
    background: #11141c !important;
}

.gradio-container:has(#mode-select input[type="checkbox"]:checked) {
    --page: #11141c;
    --card: #1a1e2b;
    --field: #252b3a;
    --answer: #252b3a;
    --fg: #f3f5fa;
    --muted: #acb5c8;
    --line: #394256;

    color-scheme: dark;
}

/* 공통 */

footer {
    display: none !important;
}

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
        #329aff,
        #8578ff,
        #ff65af
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

/* PDF 파일 선택 */

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
    box-shadow: none !important;
}

/* 업로드 상태 */

#upload-status,
#upload-status * {
    color: var(--muted) !important;
    font-size: 12px !important;
}

/* API 키 입력창 */

#api-key-box {
    background: var(--field) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

#api-key-box input {
    background: transparent !important;
    color: var(--fg) !important;
    border: none !important;
    box-shadow: none !important;
}

#api-key-box input::placeholder {
    color: var(--muted) !important;
}

#api-key-box:focus-within {
    border-color: #827aff !important;
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
    border-color: #827aff !important;
}

/* 질문하기 버튼 */

#ask-btn {
    background: linear-gradient(
        110deg,
        #329aff 0%,
        #827aff 50%,
        #ff65af 100%
    ) !important;
    border: none !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    min-height: 47px !important;
    font-size: 14px !important;
    font-weight: 750 !important;
    box-shadow: 0 5px 14px rgba(130,122,255,.15);
}

#ask-btn:hover {
    filter: brightness(1.06);
}

/* 대화 영역 (Chatbot) */

#chat-box {
    background: var(--answer) !important;
    border: 1px solid var(--line) !important;
    border-left: 4px solid #827aff !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

#chat-box,
#chat-box * {
    color: var(--fg) !important;
}

/* 말풍선: 사용자는 입력창 색, 답변은 카드 색 */

#chat-box .message {
    border: 1px solid var(--line) !important;
}

#chat-box .message.user {
    background: var(--field) !important;
}

#chat-box .message.bot {
    background: var(--card) !important;
}

/* 대화 초기화 버튼 */

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
}
"""


# ==========================================
# 3. 이벤트 함수
# ==========================================

def select_pdf(file):
    if file is None:
        return None, "대상 파일을 선택해주세요."

    path = file if isinstance(file, str) else file.name

    return path, f"선택한 파일: {Path(path).name}"


def register_pdf(pdf_path):
    return build_db(pdf_path)


def ask_question(question, openai_key, history, db): # 리스트 안에 이전 대화 기록 적재
    history = list(history or [])
    prior_history = list(history)  # 이번 질문을 넣기 전의 기록 (answer에 맥락으로 전달)

    # 빈 질문: 챗봇 안에 안내 메시지만 추가
    if not question or not question.strip():
        history.append({"role": "assistant", "content": "⚠️질문을 입력해주세요."})
        yield history, ""
        return

    # 질문을 먼저 말풍선으로 보여주고 입력창은 비움
    history.append({"role": "user", "content": question})

    # PDF 미등록: 안내 메시지 표시 (answer가 안내해 주지만 여기서도 미리 막음)
    if db is None:
        history.append({"role": "assistant", "content": "⚠️ PDF를 선택한 뒤 '파일 등록' 버튼을 눌러주세요."})
        yield history, ""
        return

    history.append({"role": "assistant", "content": "답변을 생성하는 중..."})
    yield history, ""

    try:
        result = answer(question, openai_key, db, prior_history)

        # answer가 문자열이면 한 번에, yield(스트리밍)이면 조각마다 갱신
        if isinstance(result, str):
            history[-1]["content"] = result
            yield history, ""
        else:
            for text in result:
                history[-1]["content"] = str(text)
                yield history, ""
    except Exception as e:
        history[-1]["content"] = f"⚠️ 답변 처리 오류: {e}"
        yield history, ""


def clear_chat():
    return []


# ==========================================
# 4. Gradio Blocks UI
# ==========================================

with gr.Blocks(title="[현답] 문서 질의응답 서비스") as demo:

    db_state = gr.State(value=None)
    selected_file = gr.State(value=None)

    with gr.Column(elem_id="main-card"):

        # 제목 + 다크모드
        with gr.Row(elem_id="title-row"):

            with gr.Column(scale=5, min_width=180):
                gr.HTML("""
                <div class="page-title">
                    [현답] 문서 질의응답 서비스
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

        # PDF 업로드
        gr.HTML(
            '<div class="section-label">📄 PDF 파일</div>'
        )

        with gr.Row(elem_id="file-row"):

            pdf_file = gr.UploadButton(
                "📎 PDF 파일 선택",
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

        # OpenAI API 키 (테스트 모드에서는 비워 둬도 됨)
        gr.HTML(
            '<div class="section-label">🔑 OpenAI API 키</div>'
        )

        openai_key = gr.Textbox(
            placeholder="sk-... (테스트 모드에서는 비워 두세요)",
            type="password",
            show_label=False,
            container=False,
            lines=1,
            elem_id="api-key-box"
        )

        # 질문
        gr.HTML(
            '<div class="section-label">💬 질문</div>'
        )

        question = gr.Textbox(
            placeholder="문서에 대해 궁금한 내용을 입력하세요. 줄바꿈은 (Shift+Enter를 활용하세요.)",
            show_label=False,
            container=False,
            lines=1,  # lines=1이어야 Enter로 제출됨 (줄바꿈은 Shift+Enter)
            max_lines=5,
            elem_id="question-box"
        )

        ask_btn = gr.Button(
            "질문하기 →",
            variant="primary",
            elem_id="ask-btn"
        )

        # 대화 기록
        gr.HTML(
            '<div class="section-label">✨ 채팅 기록</div>'
        )

        chatbot = gr.Chatbot(
            placeholder="질문을 입력하면 이곳에 채팅 기록이 쌓입니다.",
            show_label=False,
            layout="bubble",
            height=400,
            elem_id="chat-box"
        )

        clear_btn = gr.Button(
            "채팅 초기화",
            variant="secondary",
            elem_id="clear-btn"
        )

    # ======================================
    # 5. 이벤트 핸들러
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
        inputs=[question, openai_key, chatbot, db_state],
        outputs=[chatbot, question]
    )

    question.submit(
        fn=ask_question,
        inputs=[question, openai_key, chatbot, db_state],
        outputs=[chatbot, question]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=chatbot
    )


# ==========================================
# 6. 실행
# ==========================================

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=gr.themes.Soft(),
        inbrowser=True
    )
