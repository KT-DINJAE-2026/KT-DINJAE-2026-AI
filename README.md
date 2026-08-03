# AI

KT 디인재 프로젝트 AI 레포지토리

교통약자를 위한 QR기반 입석 위험 안내 시스템 — 입석여부(모델A)·입석시간(모델B) 예측 모델 개발

## 구조

```text
.
├── docs/
│   ├── 20260724.md                        # 회의록 (교통약자 판별 / 버스 유형 필터링 기준 / 백엔드 요청사항)
│   └── TCD_학습용명부_컬럼설명.md          # 학습용 명부 스키마 컬럼 정의
├── schema/
│   └── TCD_학습용명부_예시스펙.parquet     # 예시 스키마(실제 데이터 아님)
├── src/
│   ├── train_models.py                    # 정식 학습 코드 (300트리)
│   ├── train_models_wandb.py              # 실험용 학습 코드 (wandb 연동, 결과 자동 정리+push)
│   └── inference.py                       # 학습된 모델로 API 응답 형태 예측 결과 생성
├── experiments/
│   └── {실행시각}/                        # 실험별 결과 폴더 (train_models_wandb.py 실행 시 자동 생성)
│       ├── train_log.txt
│       ├── model_a.txt
│       ├── model_b.txt
│       └── metrics.json
├── .gitignore
├── README.md
└── requirements.txt
```

## 모델 개요

- **모델A**: LightGBM 분류 — 승차 시점에 입석하는지(Y/N) 예측
- **모델B**: LightGBM 회귀 — 모델A가 Y로 판단한 경우, 착석까지 걸리는 시간(초) 예측
- 학습 피처: 노선, 승하차 정류장, 요일/시각, 공휴일, 날씨, 배차간격, 버스 유형(좌석수), 사용자구분코드
- 정답 레이블(`is_standing`, `standing_seconds`)은 TCD 원본에서 재차인원 복원 + FIFO 좌석배정 시뮬레이션을 거쳐 산출한 값을 사용

자세한 데이터 스펙과 결정/미결정 사항은 `docs/` 참고.

## 실험 결과 추이 (300 → 1000 → 2000 → 3000 → 4000트리)

서초구 4월 한 달치(938만 행) 동일 데이터 기준, 트리 수를 늘려가며 확인:

| | 300트리 | 1000트리 | 2000트리 | 3000트리 | 4000트리 |
| --- | --- | --- | --- | --- | --- |
| 모델A AUC | 0.9374 | 0.9447 | 0.9476 | 0.9489 | **0.9498** |
| 모델A Accuracy | 92.84% | 93.07% | 93.17% | 93.22% | **93.27%** |
| 모델B MAE | 127.2초 | 121.4초 | 119.1초 | 117.8초 | **116.8초** |
| 모델B RMSE | 186.5초 | 178.7초 | 175.5초 | 173.6초 | **172.3초** |

트리 수를 늘릴수록 전 지표가 계속 개선되고 있지만, 구간별 개선 폭은 갈수록 줄어드는 추세입니다(300→1000 구간 대비 3000→4000 구간의 개선폭이 훨씬 작음 — 수확체감이 뚜렷해짐). 트리 수를 더 늘리는 것보다 `learning_rate`, `num_leaves` 등 다른 하이퍼파라미터 조정으로 넘어갈 계획입니다.

- 피처 중요도: 승하차 정류장 > 노선·시각 > 배차간격·요일 순. 공휴일·좌석수는 기여도 거의 없음(4월 데이터에 공휴일 부재로 추정)
- wandb 대시보드: `https://wandb.ai/newhaneul-inha-university/bus-standing-prediction`
- 실험 결과 히스토리는 `experiments/` 폴더에 타임스탬프별로 누적 (최신: `experiments/26.08.03.17-01-08`)

## 모델 파이프라인 상세

### 흐름 요약

1. **`train_models.py`** (정식본) 또는 **`train_models_wandb.py`** (실험용, 하이퍼파라미터는 스크립트 상단 `N_ESTIMATORS`/`LEARNING_RATE` 변수로 조정) — parquet을 그대로 넣고 실행
   - 모델A: `LGBMClassifier`로 `is_standing`(Y/N) 분류
   - 모델B: 모델A 정답이 `Y`인 행만 필터링해서 `standing_seconds` 회귀
   - 저장 포맷은 `.txt`(LightGBM 네이티브) — `.pkl`/`joblib`은 파이썬 전용이라 백엔드가 Spring(Java)이면 못 읽음. `.txt`는 LightGBM 공식 포맷이라 Java 쪽 LightGBM 바인딩으로도 로드 가능
   - ⚠️ 백엔드가 Java에서 이 `.txt`를 직접 로드할지, 아니면 AI가 별도 Python 추론 서버를 띄우고 백엔드가 API로 호출할지는 **아직 미정 — 다음 회의 안건**
2. **`inference.py`** — 학습된 모델A/B를 합쳐서 API 응답 JSON을 생성

### 입력

`predict_journey()` 함수에 아래 피처 딕셔너리와 표본 수를 전달:

| 피처 | 설명 |
| --- | --- |
| `route_id`, `board_stop_id`, `alight_stop_id` | 노선 및 승하차 정류장 (국토부 표준 ID) |
| `weekday`, `hour`, `is_holiday` | 요일/시각/공휴일 여부 |
| `weather` | 승차 시점 날씨 |
| `headway_sec` | 배차간격(초) |
| `bus_type_code`, `seat_capacity` | 버스 유형 및 좌석수 |
| `usertype_code` | 사용자구분코드 |
| `sample_count` | 해당 (노선,구간,시간대) 조합의 학습 표본 수 — `INSUFFICIENT_DATA` 판정용 |

### 상수 (팀 확정 필요)

| 상수 | 현재 기본값 | 의미 |
| --- | --- | --- |
| `STANDING_PROBA_THRESHOLD` | 0.5 | 모델A 확률이 이 이상이면 "입석"으로 확정 |
| `RISK_MEDIUM_SEC` | 300초 | 입석시간이 이하면 "보통", 초과면 "높음" (기획서 "5분 이하/초과" 기준 그대로 적용. 팀에서 논의된 "전체 탑승시간 n%" 기준은 아직 미정이라 시간 기준으로 임시 적용) |
| `MIN_SAMPLE_COUNT` | 30 | 표본이 이보다 적으면 `INSUFFICIENT_DATA` |

### 출력

```json
{
  "dataStatus": "SUCCESS",       // or "INSUFFICIENT_DATA"
  "isStanding": "Y",
  "standingSeconds": 423,
  "riskLevel": "MEDIUM",         // LOW/MEDIUM/HIGH = 기획서의 낮음/보통/높음
  "confidence": 0.812
}
```

### 최종적으로 백엔드에 넘길 것

1. `model_a.txt`, `model_b.txt` — 학습된 모델 파일
2. `predict_journey()` 함수 스펙 — 위 입력/출력 구조
3. 임계값 확정본 — 위 3개 상수를 팀 회의로 정한 뒤 문서화

## 실행

```bash
pip install -r requirements.txt

# 정식 학습 (300트리)
python src/train_models.py

# 실험용 학습 (하이퍼파라미터는 스크립트 상단에서 조정, 결과 자동으로 experiments/ 저장 및 push)
python src/train_models_wandb.py

# 추론 예시
python src/inference.py
```

## 현재 미확정 사항

- `RISK_MEDIUM_SEC`, `STANDING_PROBA_THRESHOLD`, `MIN_SAMPLE_COUNT` 등 임계값 팀 확정 필요
- 모델 서빙 방식(백엔드가 `.txt` 모델을 직접 로드 vs AI가 별도 추론 서버 운영) 결정 필요
- 교통약자 판별 범위(국가유공자/일반인 포함 여부), 임산부 식별 방안 — `docs/20260724.md` 액션 아이템 참고
- 정류장/노선 ID가 국토부 표준 ID 체계인지 백엔드와 최종 확인 필요