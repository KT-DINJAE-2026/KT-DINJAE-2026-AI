import time
import pandas as pd
import lightgbm as lgb
import wandb
from wandb.integration.lightgbm import wandb_callback, log_summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import sys

# ── 실행 시각 기록: 파일명/실험명 구분용 ──────────────
RUN_TS = datetime.now().strftime("%y.%m.%d.%H-%M-%S")

# ── 로그 저장 설정: print() 출력을 콘솔 + train_log 파일에 동시 기록 ──
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(f"train_log({RUN_TS}).txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)

# ── STEP 1: 데이터 불러오기 + 확인 ───────────────────
t0 = time.time()
df = pd.read_parquet(r"D:\프로젝트\KT디지털인재장학생('26.03.20~present, KT)\지역사회 문제해결 프로젝트\roster_all.parquet")
print(f"로딩 시간: {time.time()-t0:.1f}초")
print(df.shape)
print(df.isna().sum())

# ── STEP 2: 결측치 처리 ──────────────────────────────
df = df[df["alight_stop_id"].notna()]
print("결측 제외 후:", df.shape)

# ── STEP 3: 피처/타깃 정의 ──────────────────────────
categorical_cols = ["route_id","board_stop_id","alight_stop_id","weekday","weather","bus_type_code","usertype_code"]
numeric_cols = ["hour","is_holiday","headway_sec","seat_capacity"]
feature_cols = categorical_cols + numeric_cols

for col in categorical_cols:
    df[col] = df[col].astype("category")

df["y_standing"] = (df["is_standing"] == "Y").astype(int)
print(df["y_standing"].value_counts())

# ── STEP 4: 학습/검증 분리 ──────────────────────────
X = df[feature_cols]
y = df["y_standing"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── STEP 5: 모델A 학습 (입석여부 분류) ───────────────
wandb.init(
    project="bus-standing-prediction",
    name=f"model_a_1000trees_{RUN_TS}",
    config={
        "model": "A_classifier",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "train_rows": len(X_train),
        "features": feature_cols,
    }
)

model_a = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05)
model_a.fit(
    X_train, y_train,
    categorical_feature=categorical_cols,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(30),
        lgb.log_evaluation(50),
        wandb_callback(),
    ]
)
log_summary(model_a.booster_, save_model_checkpoint=True)

proba = model_a.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
auc_a = roc_auc_score(y_test, proba)
acc_a = accuracy_score(y_test, pred)
print("[모델A] AUC:", auc_a)
print("[모델A] Accuracy:", acc_a)

imp_a = pd.Series(model_a.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n[모델A 피처 중요도]")
print(imp_a)

wandb.log({
    "model_a/AUC": auc_a,
    "model_a/Accuracy": acc_a,
    "model_a/feature_importance": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_a.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model A Feature Importance"
    ),
})

model_a_path = f"model_a({RUN_TS}).txt"
model_a.booster_.save_model(model_a_path)
artifact_a = wandb.Artifact("model_a", type="model")
artifact_a.add_file(model_a_path)
wandb.log_artifact(artifact_a)
wandb.finish()

# ── STEP 6: 모델B 학습 (입석시간 회귀, 입석=Y만) ─────
standing_df = df[df["y_standing"] == 1]
Xb = standing_df[feature_cols]
yb = standing_df["standing_seconds"]

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.2, random_state=42
)

wandb.init(
    project="bus-standing-prediction",
    name=f"model_b_1000trees_{RUN_TS}",
    config={
        "model": "B_regressor",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "train_rows": len(Xb_train),
        "features": feature_cols,
    }
)

model_b = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05)
model_b.fit(
    Xb_train, yb_train,
    categorical_feature=categorical_cols,
    eval_set=[(Xb_test, yb_test)],
    callbacks=[
        lgb.early_stopping(30),
        lgb.log_evaluation(50),
        wandb_callback(),
    ]
)
log_summary(model_b.booster_, save_model_checkpoint=True)

pred_b = model_b.predict(Xb_test)
mae = mean_absolute_error(yb_test, pred_b)
rmse = mean_squared_error(yb_test, pred_b) ** 0.5
print(f"[모델B] MAE: {mae:.1f}초  RMSE: {rmse:.1f}초  (평균 입석시간: {yb.mean():.1f}초)")

imp_b = pd.Series(model_b.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n[모델B 피처 중요도]")
print(imp_b)

wandb.log({
    "model_b/MAE": mae,
    "model_b/RMSE": rmse,
    "model_b/feature_importance": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_b.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model B Feature Importance"
    ),
})

model_b_path = f"model_b({RUN_TS}).txt"
model_b.booster_.save_model(model_b_path)
artifact_b = wandb.Artifact("model_b", type="model")
artifact_b.add_file(model_b_path)
wandb.log_artifact(artifact_b)
wandb.finish()

print(f"\n저장 완료: {model_a_path}, {model_b_path}")
