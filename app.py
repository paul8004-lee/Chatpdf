# ===== Step 1. PDF 처리 (feature/pdf-processing) =====


# ===== Step 2. 질문 처리 (feature/qa-chain) =====
def answer(question, openai_key, db):
    if db is None:
        raise gr.Error("먼저 API 키를 입력하고 PDF를 업로드하세요.")
    if not openai_key:  # 추가: API 키 확인
        raise gr.Error("OpenAI API 키를 입력하세요.")
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

    try:
        # 추가: 검색을 한 번만 실행해 문서를 재사용 (출처 표시용)
        docs = retriever_from_llm.invoke(question)

        rag_chain = (
            {"context": lambda _: format_docs(docs), "question": RunnablePassthrough()}
            | prompt
            | generate_llm
            | StrOutputParser()
        )

        text = ""
        for chunk in rag_chain.stream(question):
            text += chunk
            yield text

        # 추가: 참고한 페이지 표시 (PyPDFLoader의 page는 0부터 시작)
        pages = sorted({d.metadata["page"] + 1 for d in docs if "page" in d.metadata})
        if pages:
            text += "\n\n📄 참고 페이지: " + ", ".join(f"p.{p}" for p in pages)
            yield text
    except Exception as e:  # 추가: API 오류 안내
        raise gr.Error(f"답변 생성 중 오류가 발생했습니다: {e}")

# ===== Step 3. 화면 (feature/chat-ui) =====

