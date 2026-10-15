---
title: Team ChatPDF
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Team ChatPDF

현대경제연구원 pdf 문서 기반 Q&A 챗봇

## 팀원과 역할
GitHub PR 작성 기록 기준으로 정리했습니다. 공식 역할명은 팀 확인 후 보완합니다.

| 이름 (GitHub) | 담당 (PR 기록 기준) | 담당 브랜치 |
|---|---|---|
| 이성민 (paul8004-lee) | 팀장, 통합·버그 수정 | `bugfix`, `fix/*` |
| 이준영 (wljycjdi2001-ljy) | Step 1 PDF 처리, 청크 튜닝, 표·이미지 청킹 | `feature/step1_answer`, `feature/STEP1_answer_JY` |
| TODO 이름 확인 (wjsworud1025) | Step 1 다중 PDF 처리, 요청 로그(#49), DB 정리(#25) | `feat]여러_PDF_업로드_및_통합_검색지원_#8`, `feature/logging_#49` |
| 소용우 (nasaro-pro) | Step 2 재정렬·캐시·성능 기록, 검색 방식 선택 | `feature/step2_answer-nasaro`, `feature/step2_answer` |
| 박영준 (joon0724) | Step 2 답변 기능, 근거 원문 표시 | `feature/step2_answer_joon` |
| 박경환 (qkrrud141) | Step 3 Gradio 화면, 다중 PDF 업로드 화면 | `feature/chat-ui`, `featurechatui` |
| 기메리 (doxoba) | Step 3 대화 기록 유지·이전 대화 기록 화면 | `feature/chat_ui/HYEL` |
| TODO (teamone-kim) | TODO: PR 기록 없음, 담당 확인 필요 | TODO |

## 실행 방법
```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
python app.py
```

## 해결하려는 문제와 대상 사용자

### 1. 해결하려는 문제 (PDF 데이터 처리 및 청킹 관점)

- **문맥 단절 및 검색 실패 문제**: 
  - 경제 연구 보고서 특성상 전문 용어, 수치 지표, 분석 문맥이 긴밀하게 연결되어 있어 청크 크기(`chunk_size`)가 너무 작으면 문맥이 끊겨 검색 적중률이 급감함.
- **정보 노이즈 및 토큰 낭비 문제**: 
  - 반대로 청크가 너무 크면 질문과 상관없는 주변 정보(노이즈)까지 과도하게 포함되어 LLM 응답 정확도가 떨어지고 불필요한 토큰 비용(Latency/비용)이 증가함.
-
### 2. 대상 사용자 (Target Audience)

- **연구원 및 경제 분석가**: 방대한 보고서 속에서 원하는 핵심 경제 지표와 근거 문장을 누락 없이 정확하게 검색해야 하는 사용자.




## 개선 내역과 성능 비교

### 🚀 청크 설정(chunk_size, chunk_overlap) 튜닝 실험 및 최적화
현대경제연구원 PDF 보고서(35페이지)를 대상으로 정답 키워드 Top-3 적중률(Hit Rate)을 평가하여 최적의 RAG 청크 파라미터를 도출했습니다.

| 실험군 | chunk_size | chunk_overlap | 생성된 청크 수 | Top-3 적중률 (Hit Rate) | 결과 및 선정 이유 |
| :---: | :---: | :---: | :---: | :---: | :---|
| **Small** | 300 | 50 | 162개 | 33.3% (1/3) | 청크가 지나치게 파편화되어 핵심 문맥 누락 발생 |
| **Medium (선정)** | **600** | **100** | **84개** | **66.7% (2/3)** | **높은 검색 적중률과 경제적인 토큰 사용량 간의 최적 균형점** |
| **Large** | 1000 | 150 | 61개 | 66.7% (2/3) | 적중률은 동일하나 청크 크기가 커 LLM 입력 토큰 낭비 및 불필요한 노이즈 증가 우려 |

#### 💡 Medium(600 / 100)을 최종 채택한 이유
1. **검색 정확도 향상**: Small(33.3%) 대비 적중률을 **66.7%로 2배 향상**시켜 안정적인 정보 검색 성능을 확보했습니다.
2. **토큰 비용 및 응답 속도 최적화 (vs Large)**:
   - Large(1000)와 Medium(600)의 검색 적중률은 66.7%로 동일합니다.
   - 하지만 Top-k(3개) 청크를 LLM에 프롬프트로 주입할 때, Large는 청크당 최대 1,000자(총 약 3,000자 이상)가 전달되어 **API 비용(토큰 소모)과 LLM 생성 지연 시간(Latency)이 크게 증가**합니다.
   - 또한 청크가 과도하게 크면 관련 없는 문맥(Noise)이 함께 전달되어 LLM의 답변 집중도를 떨어뜨릴 수 있습니다.
3. **결론**: 따라서 검색 정확도를 최대치로 유지하면서도 LLM 컨텍스트 노이즈와 비용 효율성을 고려해 **Medium (`chunk_size=600`, `chunk_overlap=100`)**을 최종 설정값으로 선정했습니다.

## PDF 처리 팀: 다중 PDF 처리

`build_db(pdf_paths, openai_key)`는 파일 경로 목록을 받아 `(db, status)`를 반환합니다. 단일 경로도 호환됩니다. 모든 PDF의 청크를 하나의 새 Chroma 컬렉션에 저장하며, 완료 메시지에 파일·전체 페이지·전체 청크 수를 포함합니다.

청크의 `source_filename`은 원본 파일명이며, `source`와 0부터 시작하는 `page` 등 로더 메타데이터는 유지됩니다. 파일 로딩·분할은 파일별로 처리해, 실패한 PDF(손상 파일, 텍스트 없는 이미지·스캔본)는 제외하고 성공한 PDF만으로 DB를 만들며 제외된 파일명과 원인을 메시지에 함께 표시합니다. 모든 파일이 실패하거나 임베딩·저장이 실패하면 `None`을 반환하고, 부분 저장된 새 컬렉션은 삭제합니다. OCR은 지원하지 않습니다.

화면 팀 연결 사항:

1. `gr.File(file_count="multiple", type="filepath")`에서 경로 목록을 전달합니다.
2. 새 업로드 시작 시 기존 DB 참조를 해제하고 질문을 차단합니다. 기존 컬렉션 삭제는 호출자가 담당합니다.
3. 반환된 DB로 상태를 교체하고, DB가 있을 때만 질문을 허용합니다. 실패 시 기존 DB를 복구하지 않습니다.
4. 검색 결과의 `source_filename`과 `page + 1`을 화면에 표시합니다.

검증: `python -m unittest test_app -v`. 실제 PDF와 Chroma를 사용하고, 임베딩만 로컬 대체값을 사용합니다. 다중 PDF 검색·메타데이터·단일 PDF·새 컬렉션 격리·손상된 PDF·텍스트 없는 PDF·임베딩 실패를 검증합니다. 실제 OpenAI API 및 화면 연동 검증은 별도입니다.

## Step 2 질문 처리: 검색 방식·근거 원문·캐시

- **검색 방식 선택**: `app.py`의 `SEARCH_MODE` 한 줄로 바꿉니다. `similarity`(가장 비슷한 조각), `mmr`(비슷하되 겹치지 않는 조각), `multiquery`(질문을 3개로 바꿔 MMR 검색, GPT 1회 추가), `multiquery_rerank`(multiquery + GPT 재정렬, GPT 2회 추가, **기본값**). `answer()`와 비교 실험이 같은 `_retrieve()`를 씁니다. 여러 PDF를 등록하면 방식과 관계없이 파일별 후보를 추가하고 빠진 파일을 보충합니다(#59). 재정렬이 실패하면 원래 순서로 답변을 이어가고 `rerank_fallback`을 기록합니다.
- **근거 원문**: 답변 아래 `참고: 파일명 (p.N)` 다음에 `근거 원문`을 최대 3개 표시합니다. 예: `> [보고서.pdf p.3] "2026년 경제성장률은 …"`. 검색이 고른 조각에서 질문 단어가 가장 많이 든 문장 1~2개(최대 150자)를 고르며, GPT를 추가로 호출하지 않습니다. 페이지 번호는 `참고`와 같은 실제 PDF 페이지(`page + 1`, #28)입니다.
- **답변 캐시**: 같은 PDF 묶음(Chroma 컬렉션 이름 기준, #24)에 같은 질문을 하면 저장된 답변(근거 원문 포함)을 바로 보여줍니다. 최대 100개까지 보관하며, PDF를 교체하면 이전 DB의 캐시를 지웁니다. DB 객체가 사라질 때도 다음 캐시 사용 시 자동으로 정리합니다.
- **기록**: `eval/qa_log.jsonl`의 `qa_result` 이벤트에 `mode`(검색 방식)와 `rerank_ok`(재정렬 성공 여부, 재정렬을 안 하는 방식이면 `null`)를 남깁니다.

### 검색 방식 비교 (`eval/compare_retrievers.py`)

`python eval/compare_retrievers.py samples/test.pdf` 실행 결과입니다. 검색만 실행하고 답변은 만들지 않습니다. 정답 페이지가 있는 `eval/qa_set.jsonl` 질문 5개 중 하나라도 정답 페이지를 찾은 비율을 적중률로 셉니다. 텍스트 청크만 사용했고, 2회 실행 결과입니다(`eval/results_retriever.csv`는 2회차).

| 검색 방식 | k=3 적중률 | k=4 적중률 | k=6 적중률 | 평균 시간 (k=4) |
|---|:---:|:---:|:---:|:---:|
| similarity | 60% / 60% | 60% / 60% | 100% / 100% | 0.2초 |
| mmr | 80% / 80% | 80% / 80% | 80% / 80% | 0.2초 |
| multiquery | 60% / 60% | 60% / 60% | 80% / 80% | 1.8~2.1초 |
| **multiquery_rerank (기본, k=4)** | 60% / 80% | **80% / 80%** | 80% / 80% | 2.8초 |

- 질문이 5개라 1문항이 20%p입니다. 방식 간 차이는 대부분 1문항 이내라 이 표만으로 우열을 정하기 어렵습니다.
- `similarity k=6`이 가장 높지만, k를 늘리면 GPT에 넘기는 페이지가 늘어 토큰·응답 시간이 커집니다. 답변 품질 평가(#37·#59 질문 47개)는 `multiquery_rerank`, `TOP_K=4` 기준으로 맞춰져 있어 기본값을 유지합니다. 바꾸려면 질문을 늘려 비교한 뒤 `eval/eval_issue37.py`로 답변 품질도 함께 확인해야 합니다.

검증: `python -m unittest test_step2_search -v` (외부 API 호출 없음). 검색 방식 4개, 재정렬 실패 기록, 캐시 키·PDF 교체 시 캐시 분리, 근거 원문 형식을 확인합니다.

## 요청별 누적 로그 (#49)

`eval/qa_log.jsonl`에 JSON 한 줄씩 추가합니다. 기존 로그는 유지되며 새 기록에는 `schema_version=1`, UTC `time`, `request_id`, `operation`, `stage`, `event`가 포함됩니다. 파일은 Git에서 제외합니다.

- PDF 읽기·분할, 임베딩·저장, 질문 재작성, 검색, 재정렬, 답변 생성의 단계 결과와 소요 시간을 기록합니다. 오류는 종류·메시지·traceback을 남기며, 재정렬 대체 처리와 일부 PDF 제외도 구분합니다.
- `request_finished.status`는 `success`, `partial`(일부 PDF 제외), `cached`, `no_results`, `empty`, `error`, `cancelled`로 요청 결과를 구분합니다. 같은 `request_id`로 단계 기록을 모으고, 서로 다른 PDF 등록·질문 요청은 `collection`으로 연결할 수 있습니다.
- `token_usage` 이벤트에 모델명과 제공자가 반환한 입력·출력·총 토큰을 기록합니다. 스트리밍도 사용량을 요청하며, 요청 종료 기록의 `token_usage`는 수집된 값의 합계입니다. 캐시 요청도 질문 재작성을 호출했다면 그 사용량이 포함됩니다.
- LangChain 임베딩 어댑터는 실제 사용량을 노출하지 않으므로 임베딩·검색 임베딩은 `usage_status=unavailable`, 토큰은 `null`, 원인은 `embedding_adapter_usage_unavailable`로 기록합니다. 실패한 모델 호출 등 사용량이 누락되면 합계의 `usage_status=partial`이며 **실제 전체 사용량이나 과금액으로 해석하면 안 됩니다**. 토큰을 추정하지 않습니다.
- 검색 후보와 최종 문맥에는 원본 파일명, 실제 PDF 페이지, 컬렉션 이름, 청크 ID를 기록합니다. 새 청크 ID는 Chroma 문서 ID와 동일합니다. API 키·Bearer 값·민감한 키 이름의 값은 가리고, 프롬프트·답변·문서 본문 전체는 저장하지 않습니다. 기존 QA 결과의 질문·검색 질문은 기록하므로 로그 파일 접근 권한과 공유 범위는 별도로 관리해야 합니다.

로그 쓰기 실패는 요청 ID와 예외 종류를 콘솔에 알리며 앱 동작을 중단하지 않습니다. 현재 단일 프로세스 내 쓰기를 잠금으로 보호합니다. 로그는 자동 삭제·회전하지 않으므로 보관·백업은 운영 환경에서 관리합니다.

오프라인 검증: `python -m unittest discover -v` (외부 API 호출 없음). 기존 PDF 테스트와 함께 정상·캐시·검색 결과 없음·오류·토큰 수집/누락·메타데이터·비밀값 가림·쓰기 실패·스트리밍 컨텍스트 전환을 확인합니다.
