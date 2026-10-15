# ===== Step 1. PDF 처리 (feature/pdf-processing) =====


# ===== Step 2. 질문 처리 (feature/qa-chain) =====



# ===== Step 3. 화면 (feature/chat-ui) =====

import gradio as gr
from pathlib import Path


# ==========================================
# 1. 테스트용 함수
# 실제 Step 1, Step 2 완성 후 삭제
# ==========================================

def build_db(pdf_path):
    if not pdf_path:
        return None, "PDF 파일을 선택해주세요."

    filename = Path(pdf_path).name

    return (
        {"filename": filename},
        f"✅ {filename} 등록 완료 (테스트 모드)"
    )


def answer(question, db):
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

/* 답변 영역 */

#answer-box {
    background: var(--answer) !important;
    border: 1px solid var(--line) !important;
    border-left: 4px solid #827aff !important;
    border-radius: 12px !important;
    padding: 16px !important;
    min-height: 90px;
    box-shadow: none !important;
}

#answer-box,
#answer-box * {
    color: var(--fg) !important;
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
        return None, "파일을 선택해주세요."

    path = file if isinstance(file, str) else file.name

    return path, f"선택한 파일: {Path(path).name}"


def register_pdf(pdf_path):
    return build_db(pdf_path)


def ask_question(question, db):
    if not question or not question.strip():
        return "⚠️ 질문을 입력해주세요."

    try:
        return str(answer(question, db))
    except Exception as e:
        return f"⚠️ 답변 처리 오류: {e}"


# ==========================================
# 4. Gradio Blocks UI
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

        # 질문
        gr.HTML(
            '<div class="section-label">💬 질문</div>'
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
            '<div class="section-label">✨ 답변</div>'
        )

        output = gr.Markdown(
            "질문을 입력하면 이곳에 답변이 표시됩니다.",
            elem_id="answer-box"
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
        inputs=[question, db_state],
        outputs=output
    )

    question.submit(
        fn=ask_question,
        inputs=[question, db_state],
        outputs=output
    )


# ==========================================
# 6. 실행 (Gradio 6.x)
# ==========================================

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=gr.themes.Soft(),
        inbrowser=True
    )
