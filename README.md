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

도메인 특화 문서 기반 Q&A 챗봇

## 팀원과 역할
| 이름 | 역할 | 담당 브랜치 |
|---|---|---|
|  | PM / 팀장 |  |
|  |  | feature/pdf-processing |
|  |  | feature/qa-chain |
|  |  | feature/chat-ui |

## 실행 방법
```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
python app.py
```

## 해결하려는 문제와 대상 사용자
(작성 예정)


## 개선 내역과 성능 비교
(작성 예정)

## PDF 처리 팀: 다중 PDF 처리

`build_db(pdf_paths, openai_key)`는 파일 경로 목록을 받아 `(db, status)`를 반환합니다. 단일 경로도 호환됩니다. 모든 PDF의 청크를 하나의 새 Chroma 컬렉션에 저장하며, 완료 메시지에 파일·전체 페이지·전체 청크 수를 포함합니다.

청크의 `source_filename`은 원본 파일명이며, `source`와 0부터 시작하는 `page` 등 로더 메타데이터는 유지됩니다. 파일 로딩·분할·저장 실패 시 실패한 파일명이 포함된 메시지와 `None`을 반환하고, 부분 저장된 새 컬렉션은 삭제합니다. 텍스트 없는 PDF는 실패 처리하며 OCR은 지원하지 않습니다.

화면 팀 연결 사항:

1. `gr.File(file_count="multiple", type="filepath")`에서 경로 목록을 전달합니다.
2. 새 업로드 시작 시 기존 DB 참조를 해제하고 질문을 차단합니다. 기존 컬렉션 삭제는 호출자가 담당합니다.
3. 반환된 DB로 상태를 교체하고, DB가 있을 때만 질문을 허용합니다. 실패 시 기존 DB를 복구하지 않습니다.
4. 검색 결과의 `source_filename`과 `page + 1`을 화면에 표시합니다.

검증: `python -m unittest test_app -v`. 실제 PDF와 Chroma를 사용하고, 임베딩만 로컬 대체값을 사용합니다. 다중 PDF 검색·메타데이터·단일 PDF·새 컬렉션 격리·손상된 PDF·텍스트 없는 PDF·임베딩 실패를 검증합니다. 실제 OpenAI API 및 화면 연동 검증은 별도입니다.
