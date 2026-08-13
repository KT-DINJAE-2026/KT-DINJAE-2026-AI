import time
import os
import json
import gc
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

DATASET_LABEL = "seongbuk_2024"
DATA_DIR = r"D:\프로젝트\KT디지털인재장학생('26.03.20~present, KT)\지역사회 문제해결 프로젝트\roster-2024-final"

# merge_data.py의 SAMPLE_EVERY_N_DAYS와 반드시 같은 값으로 맞춰주세요 (캐시 파일명이 여기서 결정됨)
SAMPLE_EVERY_N_DAYS = 2 # None -> 전체 데이터 사용
suffix = f"_every{SAMPLE_EVERY_N_DAYS}d" if SAMPLE_EVERY_N_DAYS and SAMPLE_EVERY_N_DAYS > 1 else "_full"
MERGED_CACHE_PATH = os.path.join(os.path.dirname(DATA_DIR), f"roster_{DATASET_LABEL}{suffix}_merged.parquet")

# ── 하이퍼파라미터: A/B 공통 ────────────────────────
N_ESTIMATORS = 100
FEATURE_FRACTION = 1.0
BAGGING_FRACTION = 1.0
BAGGING_FREQ = 0

# 모델A(분류) 전용
LEARNING_RATE_A = 0.1
REG_LAMBDA_A = 0.0
NUM_LEAVES_A = 31
MIN_CHILD_SAMPLES_A = 20

# 모델B(회귀) 전용
LEARNING_RATE_B = 0.1
REG_LAMBDA_B = 0.0
NUM_LEAVES_B = 31
MIN_CHILD_SAMPLES_B = 20

# ── 검증 방식 ──────────────────────────────────────
USE_TEMPORAL_SPLIT = True
TEMPORAL_CUTOFF = "2024-12-01 00:00:00"

RUN_TS = datetime.now().strftime("%y.%m.%d.%H-%M-%S")
RUN_DIR = os.path.join(EXPERIMENTS_DIR, RUN_TS)
os.makedirs(RUN_DIR, exist_ok=True)

def save_model_safely(booster, final_path):
    tmp_dir = tempfile.mkdtemp(prefix="lgbm_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(final_path))
    booster.save_model(tmp_path)
    shutil.copy(tmp_path, final_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

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

# ── STEP 1: 병합된 데이터 불러오기 ───────────────────
categorical_cols = ["route_id","board_stop_id","alight_stop_id","weekday","weather","bus_type_code"]
numeric_cols = ["hour","is_holiday","headway_sec","seat_capacity"]
feature_cols = categorical_cols + numeric_cols

t0 = time.time()

if not os.path.exists(MERGED_CACHE_PATH):
    raise FileNotFoundError(
        f"병합된 데이터가 없습니다: {MERGED_CACHE_PATH}\n"
        f"먼저 merge_data.py를 실행해서 데이터를 병합해주세요."
    )

df = pd.read_parquet(MERGED_CACHE_PATH)
print(f"로딩 시간: {time.time()-t0:.1f}초")
print(df.shape)
print(df.isna().sum())

# ── STEP 2: 결측치 처리 ──────────────────────────────
df = df[df["alight_stop_id"].notna()]
print("결측 제외 후:", df.shape)

df["y_standing"] = (df["is_standing"] == "Y").astype(int)
print(df["y_standing"].value_counts())

# ── STEP 3: 모델B용 데이터를 먼저 작게 뽑아둠 (메모리 절약 핵심) ──
# df 전체(수 GB)를 계속 들고 있는 대신, 입석(Y)인 행만 미리 추려서
# 작은 사본을 만들어두고, 아래에서 df 자체는 곧 삭제합니다.
standing_slim = df.loc[
    df["y_standing"] == 1,
    feature_cols + ["board_datetime", "standing_seconds"]
].copy()
print(f"\n[모델B용 사전 추출] 입석 행: {len(standing_slim):,}")

# ── STEP 4: 학습/검증 분리 (모델A) ───────────────────
if USE_TEMPORAL_SPLIT:
    print(f"\n검증 방식: 시간분할 (기준시각={TEMPORAL_CUTOFF})")
    cutoff = pd.Timestamp(TEMPORAL_CUTOFF)
    train_mask = df["board_datetime"] < cutoff
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "y_standing"]
    X_test = df.loc[~train_mask, feature_cols]
    y_test = df.loc[~train_mask, "y_standing"]
    print(f"학습 행: {len(X_train):,} / 검증 행: {len(X_test):,}")
else:
    print("\n검증 방식: 랜덤분할 (random_state=42)")
    X = df[feature_cols]
    y = df["y_standing"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# df는 이제 필요한 부분을 다 뽑아냈으니 즉시 삭제해서 메모리 확보
del df
gc.collect()
print(f"df 삭제 완료, 메모리 확보 ({time.time()-t0:.1f}초 경과)")

# ── STEP 5: 모델A 학습 (입석여부 분류) ───────────────
wandb.init(
    project="bus-standing-prediction",
    name=f"model_a_{DATASET_LABEL}_{N_ESTIMATORS}trees_{'temporal' if USE_TEMPORAL_SPLIT else 'random'}_{RUN_TS}",
    config={
        "model": "A_classifier",
        "dataset_label": DATASET_LABEL,
        "sample_every_n_days": SAMPLE_EVERY_N_DAYS,
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE_A,
        "num_leaves": NUM_LEAVES_A,
        "min_child_samples": MIN_CHILD_SAMPLES_A,
        "feature_fraction": FEATURE_FRACTION,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "reg_lambda": REG_LAMBDA_A,
        "split_type": "temporal" if USE_TEMPORAL_SPLIT else "random",
        "train_rows": len(X_train),
        "features": feature_cols,
    }
)

model_a = lgb.LGBMClassifier(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE_A,
    num_leaves=NUM_LEAVES_A,
    min_child_samples=MIN_CHILD_SAMPLES_A,
    feature_fraction=FEATURE_FRACTION,
    bagging_fraction=BAGGING_FRACTION,
    bagging_freq=BAGGING_FREQ,
    reg_lambda=REG_LAMBDA_A,
)
model_a.fit(
    X_train, y_train,
    categorical_feature=categorical_cols,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_names=['train', 'valid'],
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

imp_a_split = pd.Series(model_a.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n[모델A 피처 중요도 - split]")
print(imp_a_split)

imp_a_gain = pd.Series(
    model_a.booster_.feature_importance(importance_type='gain'),
    index=feature_cols
).sort_values(ascending=False)
print("\n[모델A 피처 중요도 - gain]")
print(imp_a_gain)

wandb.log({
    "model_a/AUC": auc_a,
    "model_a/Accuracy": acc_a,
    "model_a/feature_importance_split": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_a_split.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model A Feature Importance (split)"
    ),
    "model_a/feature_importance_gain": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_a_gain.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model A Feature Importance (gain)"
    ),
})

model_a_path = os.path.join(RUN_DIR, "model_a.txt")
save_model_safely(model_a.booster_, model_a_path)

artifact_a = wandb.Artifact("model_a", type="model")
artifact_a.add_file(model_a_path)
wandb.log_artifact(artifact_a)
wandb.finish()

# 모델A 학습에 썼던 X_train/X_test도 이제 필요 없으니 정리
del X_train, X_test, y_train, y_test
gc.collect()

# ── STEP 6: 모델B 학습 (입석시간 회귀) — STEP3에서 미리 뽑아둔 걸 사용 ──
if USE_TEMPORAL_SPLIT:
    train_mask_b = standing_slim["board_datetime"] < cutoff
    Xb_train = standing_slim.loc[train_mask_b, feature_cols]
    yb_train = standing_slim.loc[train_mask_b, "standing_seconds"]
    Xb_test = standing_slim.loc[~train_mask_b, feature_cols]
    yb_test = standing_slim.loc[~train_mask_b, "standing_seconds"]
    print(f"\n[모델B] 학습 행: {len(Xb_train):,} / 검증 행: {len(Xb_test):,}")
else:
    Xb = standing_slim[feature_cols]
    yb = standing_slim["standing_seconds"]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=42
    )

wandb.init(
    project="bus-standing-prediction",
    name=f"model_b_{DATASET_LABEL}_{N_ESTIMATORS}trees_{'temporal' if USE_TEMPORAL_SPLIT else 'random'}_{RUN_TS}",
    config={
        "model": "B_regressor",
        "dataset_label": DATASET_LABEL,
        "sample_every_n_days": SAMPLE_EVERY_N_DAYS,
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE_B,
        "num_leaves": NUM_LEAVES_B,
        "min_child_samples": MIN_CHILD_SAMPLES_B,
        "feature_fraction": FEATURE_FRACTION,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "reg_lambda": REG_LAMBDA_B,
        "split_type": "temporal" if USE_TEMPORAL_SPLIT else "random",
        "train_rows": len(Xb_train),
        "features": feature_cols,
    }
)

model_b = lgb.LGBMRegressor(
    n_estimators=N_ESTIMATORS,
    learning_rate=LEARNING_RATE_B,
    num_leaves=NUM_LEAVES_B,
    min_child_samples=MIN_CHILD_SAMPLES_B,
    feature_fraction=FEATURE_FRACTION,
    bagging_fraction=BAGGING_FRACTION,
    bagging_freq=BAGGING_FREQ,
    reg_lambda=REG_LAMBDA_B,
)
model_b.fit(
    Xb_train, yb_train,
    categorical_feature=categorical_cols,
    eval_set=[(Xb_train, yb_train), (Xb_test, yb_test)],
    eval_names=['train', 'valid'],
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
print(f"[모델B] MAE: {mae:.1f}초  RMSE: {rmse:.1f}초  (평균 입석시간: {yb_train.mean():.1f}초)")

imp_b_split = pd.Series(model_b.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n[모델B 피처 중요도 - split]")
print(imp_b_split)

imp_b_gain = pd.Series(
    model_b.booster_.feature_importance(importance_type='gain'),
    index=feature_cols
).sort_values(ascending=False)
print("\n[모델B 피처 중요도 - gain]")
print(imp_b_gain)

wandb.log({
    "model_b/MAE": mae,
    "model_b/RMSE": rmse,
    "model_b/feature_importance_split": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_b_split.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model B Feature Importance (split)"
    ),
    "model_b/feature_importance_gain": wandb.plot.bar(
        wandb.Table(data=[[k, v] for k, v in imp_b_gain.items()], columns=["feature", "importance"]),
        "feature", "importance", title="Model B Feature Importance (gain)"
    ),
})

model_b_path = os.path.join(RUN_DIR, "model_b.txt")
save_model_safely(model_b.booster_, model_b_path)

artifact_b = wandb.Artifact("model_b", type="model")
artifact_b.add_file(model_b_path)
wandb.log_artifact(artifact_b)
wandb.finish()

print(f"\n저장 완료: {model_a_path}, {model_b_path}")

# ── STEP 7: metrics.json 저장 ─────────────────────────
metrics = {
    "run_ts": RUN_TS,
    "dataset_label": DATASET_LABEL,
    "sample_every_n_days": SAMPLE_EVERY_N_DAYS,
    "n_estimators": N_ESTIMATORS,
    "feature_fraction": FEATURE_FRACTION,
    "bagging_fraction": BAGGING_FRACTION,
    "bagging_freq": BAGGING_FREQ,
    "learning_rate_a": LEARNING_RATE_A,
    "reg_lambda_a": REG_LAMBDA_A,
    "num_leaves_a": NUM_LEAVES_A,
    "min_child_samples_a": MIN_CHILD_SAMPLES_A,
    "learning_rate_b": LEARNING_RATE_B,
    "reg_lambda_b": REG_LAMBDA_B,
    "num_leaves_b": NUM_LEAVES_B,
    "min_child_samples_b": MIN_CHILD_SAMPLES_B,
    "split_type": "temporal" if USE_TEMPORAL_SPLIT else "random",
    "temporal_cutoff": TEMPORAL_CUTOFF if USE_TEMPORAL_SPLIT else None,
    "train_rows_a": len(X_train),
    "train_rows_b": len(Xb_train),
    "model_a": {"auc": auc_a, "accuracy": acc_a},
    "model_b": {"mae": mae, "rmse": rmse},
    "feature_importance_a_gain": imp_a_gain.to_dict(),
    "feature_importance_b_gain": imp_b_gain.to_dict(),
}
metrics_path = os.path.join(RUN_DIR, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"metrics.json 저장 완료: {metrics_path}")

# ── STEP 8: 로그 파일 닫고 stdout 원복 ──────────────
sys.stdout = sys.stdout.files[0]
log_file.close()

# ── STEP 9: Git add + commit + push ──────────────────
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
        ok = run_git(["commit", "-m", f"Add experiment result {RUN_TS} ({DATASET_LABEL}, split={'temporal' if USE_TEMPORAL_SPLIT else 'random'}, A: lr={LEARNING_RATE_A}/leaves={NUM_LEAVES_A}/min_child={MIN_CHILD_SAMPLES_A}/lambda={REG_LAMBDA_A}, B: lr={LEARNING_RATE_B}/leaves={NUM_LEAVES_B}/min_child={MIN_CHILD_SAMPLES_B}/lambda={REG_LAMBDA_B}, AUC={auc_a:.4f}, MAE={mae:.1f}s)"])
    if ok:
        ok = run_git(["push"])
    if ok:
        print(f"\n✅ GitHub에 experiments/{RUN_TS}/ 업로드 완료")
    else:
        print(f"\n⚠ 자동 push 실패 — 수동으로 'git add {rel_path} && git commit && git push' 실행 필요")
else:
    print(f"\n(AUTO_PUSH 꺼짐) experiments/{RUN_TS}/ 폴더 생성만 완료 — 확인 후 AUTO_PUSH=True로 바꿔서 재실행하거나 수동으로 git add/commit/push 하세요.")