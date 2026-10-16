# 파일 분석 데이터스키마 샘플

기존 로컬 PDF 분석 결과인 `library/schema.sqlite3`에서 추출한 `schema-v2-hybrid` 저장 구조와 실제 데이터 예시다. 현재 develop의 실행 코드가 사용하는 데이터 계약으로 확정한 것은 아니다.

## 파일

| 파일 | 용도 |
| --- | --- |
| `schema.sql` | 원본 DB의 documents, passages, passage_fts 생성 SQL |
| `sample.json` | 공개 보고서 local-1의 분석 결과 1건과 페이지·chunk_id 순 첫 3개 청크 |
| `sample.schema.json` | sample.json의 구조를 검증하는 JSON Schema 2020-12 |

## 저장 구조

`documents.report_id`는 문서 식별자이며 `passages.report_id`로 본문 청크와 연결한다. `passages.chunk_id`는 청크 식별자다. 원본 DDL에는 외래키 제약이 없으므로 참조 무결성은 별도 확인해야 한다. `passage_fts`는 chunk_id, report_id, text를 복제하는 SQLite FTS5 검색 테이블이다. 동기화 트리거는 없으며 검색 인덱스는 별도로 채워야 한다.

| documents 필드 | 의미 |
| --- | --- |
| report_id, path, filename | 문서 ID, 저장 경로, 원본 파일명 |
| title, published_date, category | 제목, 발행일 문자열, 분류 |
| content_hash, mtime_ns, size | 원본 파일 해시, 수정 시각(ns), 크기(bytes); mtime_ns와 size는 NULL 허용 |
| available, status, error, version | 파일 존재 여부(0/1), 처리 상태, 오류 문자열, 분석 버전 |
| payload | 분석 메타데이터 JSON 문자열 |

`payload`에는 문서 메타데이터 외에 `topics`(주제 배열), `summary`(생성 요약), `claims`(생성 주장 배열), `source_url`, `analysis_status`, `analysis_error`, `analysis_version` 등이 있다. JSON 샘플은 가독성을 위해 payload를 객체로 풀었다. SQLite 삽입 시에는 JSON 문자열로 직렬화한다. payload.available은 boolean이고 documents.available은 정수다.

`passages`의 page는 1부터 시작하는 PDF 페이지 번호이고 text는 추출한 본문이다. 생성 요약과 주장은 모델 분석 결과이므로 검증된 사실로 간주하지 않는다.

## 원본과 차이 및 한계

추출 시점 원본에는 문서 101건(ready 98, pending 2, processing 1), 본문 청크 2,504건이 있었다. 문서 99건에서 documents.status와 payload.analysis_status가 달랐다. 샘플의 local-1도 status=ready, payload.analysis_status=pending인 원래 값을 유지했다. 두 상태의 차이를 임의로 보정하지 않았으며, 소비자는 상태 기준을 별도로 정해야 한다.

샘플의 공개 원문 출처는 payload.source_url에 기록했다. 내용의 최신성이나 원문 대비 정확성을 이번 작업에서 재검증하지 않았다. 전체 본문·PDF·벡터·DB는 포함하지 않는다. documents.path와 payload.path의 사용자 절대경로만 `library/pdfs/<filename>`으로 바꿨다. 이 경로는 예시이며, available=1은 원본 분석 당시 값으로 샘플 배포 환경의 파일 존재를 보장하지 않는다.

JSON Schema는 이 ready 샘플의 관찰된 필드를 기술한다. 전체 101건의 모든 상태·선택 필드를 포괄하는 운영 스키마가 아니다. 예를 들어 일부 원본에는 auto_classified가 있고, 미완료 문서에는 claims 또는 analysis_version이 없다. SQL은 원본 DDL을 그대로 담으므로 status의 허용 값이나 JSON 내부 필드까지 제한하지 않는다.

## 검증

JSON Schema 검증, 빈 메모리 SQLite DB에서 DDL 실행·샘플 삽입·문서/청크 연결·FTS 조회를 확인한다. 샘플 값은 경로 치환을 제외하고 원본 DB와 비교한다. 원본 DB는 읽기 전용으로 열며 재분석 API를 호출하지 않는다.
