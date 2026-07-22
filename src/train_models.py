"""
모델A: 입석여부(is_standing) 이진분류
모델B: 입석시간(standing_seconds) 회귀 — 모델A가 Y로 예측한 데이터만 사용

입력: 백엔드가 제공하는 학습용 명부 parquet
      (스펙: TCD_학습용명부_컬럼설명.md 참고)
출력: model_a.txt, model_b.txt  (LightGBM 네이티브 텍스트 포맷)
      — 파이썬 없이도(Java 등) 로딩 가능한 포맷이라 백엔드 연동에 유리
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error

# ── 0. 설정 ──────────────────────────────────────────────
DATA_PATH = "TCD_학습용명부.parquet"   # 백엔드가 준 실제 파일로 교체
RANDOM_STATE = 42

CATEGORICAL_COLS = [
    "route_id", "board_stop_id", "alight_stop_id",
    "weekday", "weather", "bus_type_code", "usertype_code",
]
NUMERIC_COLS = ["hour", "is_holiday", "headway_sec", "seat_capacity"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# ── 1. 데이터 로드 & 전처리 ──────────────────────────────
df = pd.read_parquet(DATA_PATH)

# LightGBM 범주형 처리를 위해 category 타입으로 변환
for col in CATEGORICAL_COLS:
    df[col] = df[col].astype("category")

df["y_standing"] = (df["is_standing"] == "Y").astype(int)

X = df[FEATURE_COLS]
y_cls = df["y_standing"]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y_cls, df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cls
)

# ── 2. 모델A: 입석여부 분류 ──────────────────────────────
model_a = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
)
model_a.fit(
    X_train, y_train,
    categorical_feature=CATEGORICAL_COLS,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
)

proba_test = model_a.predict_proba(X_test)[:, 1]
print("[모델A] AUC:", roc_auc_score(y_test, proba_test))

# ── 3. 모델B: 입석시간 회귀 (입석=Y 데이터만) ────────────
standing_df = df[df["y_standing"] == 1]
Xb = standing_df[FEATURE_COLS]
yb = standing_df["standing_seconds"]

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.2, random_state=RANDOM_STATE
)

model_b = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
)
model_b.fit(
    Xb_train, yb_train,
    categorical_feature=CATEGORICAL_COLS,
    eval_set=[(Xb_test, yb_test)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
)

pred_b = model_b.predict(Xb_test)
print("[모델B] MAE(초):", mean_absolute_error(yb_test, pred_b))

# ── 4. 모델 저장 (네이티브 텍스트 포맷 — 언어 무관 로딩 가능) ─
model_a.booster_.save_model("model_a.txt")
model_b.booster_.save_model("model_b.txt")
print("저장 완료: model_a.txt, model_b.txt")
