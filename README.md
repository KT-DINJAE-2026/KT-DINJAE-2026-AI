# AI

KT 디인재 프로젝트 AI 레포지토리

교통약자를 위한 QR기반 입석 위험 안내 시스템 — 입석여부(모델A)·입석시간(모델B) 예측 모델 개발

## 구조

```text
.
├── docs/
│   ├── 20260724.md      # 회의록 (교통약자 판별 / 버스 유형 필터링 기준 / 백엔드 요청사항)
│   └── TCD_학습용명부_컬럼설명.md          # 학습용 명부 스키마 컬럼 정의
├── schema/
│   └── TCD_학습용명부_예시스펙.parquet     # 예시 스키마(실제 데이터 아님)
├── src/
│   ├── train_models.py                    # 모델A(입석여부 분류)/모델B(입석시간 회귀) 학습
│   └── inference.py                       # 학습된 모델로 API 응답 형태 예측 결과 생성
└── requirements.txt
```

## 모델 개요

- **모델A**: LightGBM 분류 — 승차 시점에 입석하는지(Y/N) 예측
- **모델B**: LightGBM 회귀 — 모델A가 Y로 판단한 경우, 착석까지 걸리는 시간(초) 예측
- 학습 피처: 노선, 승하차 정류장, 요일/시각, 공휴일, 날씨, 배차간격, 버스 유형(좌석수), 사용자구분코드
- 정답 레이블(`is_standing`, `standing_seconds`)은 백엔드가 TCD 원본에서 재차인원 복원 + FIFO 좌석배정 시뮬레이션을 거쳐 산출한 값을 사용

자세한 데이터 스펙과 결정/미결정 사항은 `docs/` 참고.

## 실행

```bash
pip install -r requirements.txt

# 학습 (백엔드가 준 실제 parquet 파일 경로로 DATA_PATH 수정 후 실행)
python src/train_models.py

# 추론 예시
python src/inference.py
```

## 현재 미확정 사항

- `RISK_MEDIUM_SEC`, `STANDING_PROBA_THRESHOLD`, `MIN_SAMPLE_COUNT` 등 임계값 팀 확정 필요
- 모델 서빙 방식(백엔드가 `.txt` 모델을 직접 로드 vs AI가 별도 추론 서버 운영) 결정 필요
- 교통약자 판별 범위(국가유공자/일반인 포함 여부), 임산부 식별 방안 — `docs/TCD_meeting_notes_20260722.md` 액션 아이템 참고
