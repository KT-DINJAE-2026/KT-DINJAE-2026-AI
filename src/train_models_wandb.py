import time
import os
import json
import shutil
import tempfile
import subprocess
import pandas as pd
import lightgbm as lgb
import wandb
from wandb.integration.lightgbm import wandb_callback, log_summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import sys

# ── 0. 경로 설정 ──────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(REPO_DIR, "experiments")

AUTO_PUSH = True

RUN_TS = datetime.now().strftime("%y.%m.%d.%H-%M-%S")
RUN_DIR = os.path.join(EXPERIMENTS_DIR, RUN_TS)
os.makedirs(RUN_DIR, exist_ok=True)

# ── LightGBM 저장 우회 함수: 한글 경로에서 save_model()이 실패하는 문제 회피 ──
# 영문 임시 폴더에 먼저 저장한 뒤, 파이썬 shutil로 최종(한글 포함) 경로에 복사
def save_model_safely(booster, final_path):
    tmp_dir = tempfile.mkdtemp(prefix="lgbm_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(final_path))
    booster.save_model(tmp_path)
    shutil.copy(tmp_path, final_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

# ── 로그 저장 설정: print() 출력을 콘솔 + 실험 폴더 안 train_log.txt에 동시 기록 ──
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

log_path = os.path.join(RUN_DIR, "train_log.txt")
log_file = open(log_path, "w", encoding="utf-8")
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
log_summary(model_a.booster_, save_model_checkpoint=False)

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

model_a_path = os.path.join(RUN_DIR, "model_a.txt")
save_model_safely(model_a.booster_, model_a_path)   # ← 우회 저장

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
log_summary(model_b.booster_, save_model_checkpoint=False)

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

model_b_path = os.path.join(RUN_DIR, "model_b.txt")
save_model_safely(model_b.booster_, model_b_path)   # ← 우회 저장

artifact_b = wandb.Artifact("model_b", type="model")
artifact_b.add_file(model_b_path)
wandb.log_artifact(artifact_b)
wandb.finish()

print(f"\n저장 완료: {model_a_path}, {model_b_path}")

# ── STEP 7: metrics.json 저장 ─────────────────────────
metrics = {
    "run_ts": RUN_TS,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "train_rows_a": len(X_train),
    "train_rows_b": len(Xb_train),
    "model_a": {"auc": auc_a, "accuracy": acc_a},
    "model_b": {"mae": mae, "rmse": rmse},
}
metrics_path = os.path.join(RUN_DIR, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"metrics.json 저장 완료: {metrics_path}")

# ── STEP 8: 로그 파일 닫고 stdout 원복 (git 명령 출력은 콘솔에만) ──
sys.stdout = sys.stdout.files[0]
log_file.close()

# ── STEP 9: Git add + commit + push (AUTO_PUSH=True일 때만) ──
def run_git(args):
    result = subprocess.run(
        ["git"] + args, cwd=REPO_DIR,
        capture_output=True, text=True, encoding="utf-8"
    )
    print(f"$ git {' '.join(args)}")
    print(result.stdout)
    if result.returncode != 0:
        print("⚠ git 에러:", result.stderr)
    return result.returncode == 0

rel_path = os.path.relpath(RUN_DIR, REPO_DIR)

if AUTO_PUSH:
    ok = run_git(["add", rel_path])
    if ok:
        ok = run_git(["commit", "-m", f"Add experiment result {RUN_TS} (AUC={auc_a:.4f}, MAE={mae:.1f}s)"])
    if ok:
        ok = run_git(["push"])
    if ok:
        print(f"\n✅ GitHub에 experiments/{RUN_TS}/ 업로드 완료")
    else:
        print(f"\n⚠ 자동 push 실패 — 수동으로 'git add {rel_path} && git commit && git push' 실행 필요")
else:
    print(f"\n(AUTO_PUSH 꺼짐) experiments/{RUN_TS}/ 폴더 생성만 완료 — 확인 후 AUTO_PUSH=True로 바꿔서 재실행하거나 수동으로 git add/commit/push 하세요.")