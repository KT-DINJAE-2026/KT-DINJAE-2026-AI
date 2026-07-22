"""
학습된 모델A/B를 로드해서, 백엔드 API(/api/v1/journeys/predictions)가
그대로 받아 쓸 수 있는 JSON 형태로 예측 결과를 출력한다.

임계값(THRESHOLD_*)은 아직 팀 확정 전이라 기본값만 넣어둠 —
회의록 "미결정 사항"과 함께 최종 확정 필요.
"""

import lightgbm as lgb
import pandas as pd

# ── 설정값 (팀 확정 필요) ────────────────────────────────
STANDING_PROBA_THRESHOLD = 0.5   # 모델A 확률이 이 값 이상이면 "입석"으로 판단
RISK_MEDIUM_SEC = 300            # 입석시간이 이 값 이하면 "보통", 초과면 "높음"
MIN_SAMPLE_COUNT = 30            # 이보다 표본이 적으면 INSUFFICIENT_DATA

CATEGORICAL_COLS = [
    "route_id", "board_stop_id", "alight_stop_id",
    "weekday", "weather", "bus_type_code", "usertype_code",
]
NUMERIC_COLS = ["hour", "is_holiday", "headway_sec", "seat_capacity"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

model_a = lgb.Booster(model_file="model_a.txt")
model_b = lgb.Booster(model_file="model_b.txt")


def predict_journey(features: dict, sample_count: int) -> dict:
    """
    features: FEATURE_COLS를 키로 갖는 단일 요청 dict
    sample_count: 해당 (노선,구간,시간대) 조합의 학습 표본 수
                  — 백엔드가 명부와 함께 준 sample_count_route_segment_hour
    """
    if sample_count < MIN_SAMPLE_COUNT:
        return {
            "dataStatus": "INSUFFICIENT_DATA",
            "isStanding": None,
            "standingSeconds": None,
            "riskLevel": None,
            "confidence": None,
        }

    X = pd.DataFrame([features])
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")

    proba = model_a.predict(X)[0]
    is_standing = proba >= STANDING_PROBA_THRESHOLD

    if is_standing:
        standing_seconds = int(model_b.predict(X)[0])
        risk_level = "MEDIUM" if standing_seconds <= RISK_MEDIUM_SEC else "HIGH"
    else:
        standing_seconds = 0
        risk_level = "LOW"

    return {
        "dataStatus": "SUCCESS",
        "isStanding": "Y" if is_standing else "N",
        "standingSeconds": standing_seconds,
        "riskLevel": risk_level,          # LOW/MEDIUM/HIGH = 낮음/보통/높음
        "confidence": round(float(proba if is_standing else 1 - proba), 3),
    }


if __name__ == "__main__":
    example_input = {
        "route_id": "121900005",
        "board_stop_id": "121900210",
        "alight_stop_id": "121900203",
        "weekday": "화",
        "weather": "맑음",
        "bus_type_code": "105",
        "usertype_code": "04",
        "hour": 8,
        "is_holiday": False,
        "headway_sec": 520,
        "seat_capacity": 20,
    }
    result = predict_journey(example_input, sample_count=842)
    print(result)
