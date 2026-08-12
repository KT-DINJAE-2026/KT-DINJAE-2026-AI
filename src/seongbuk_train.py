import time
import os
import json
import glob
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

# ── 데이터 설정: 성북구 2024년 1년치 (366개 일별 파일) ──
DATASET_LABEL = "seongbuk_2024"
# 366개 roster_YYYYMMDD.parquet 파일이 들어있는 폴더 — 실제 경로로 수정하세요
DATA_DIR = r"D:\프로젝트\KT디지털인재장학생('26.03.20~present, KT)\지역사회 문제해결 프로젝트\roster-2024-final"
# 병합 결과를 캐시할 위치 (매번 366개 파일 다시 읽지 않도록)
MERGED_CACHE_PATH = os.path.join(os.path.dirname(DATA_DIR), f"roster_{DATASET_LABEL}_merged.parquet")

# 개발/테스트용: None이면 366일 전체 사용. 정수(N)를 주면 N일 간격으로만 읽어서 빠르게 파이프라인 확인
# 예: 7 → 약 52일치만 사용 (전체의 약 1/7)
SAMPLE_EVERY_N_DAYS = None

# ── 하이퍼파라미터: A/B 공통 ────────────────────────
N_ESTIMATORS = 5000
FEATURE_FRACTION = 0.8
BAGGING_FRACTION = 1.0
BAGGING_FREQ = 0

# 모델A(분류) 전용
LEARNING_RATE_A = 0.1
REG_LAMBDA_A = 1.0
NUM_LEAVES_A = 127
MIN_CHILD_SAMPLES_A = 20

# 모델B(회귀) 전용
LEARNING_RATE_B = 0.05
REG_LAMBDA_B = 0.0
NUM_LEAVES_B = 127
MIN_CHILD_SAMPLES_B = 20

# ── 검증 방식: True면 시간분할(과거→미래), False면 기존 랜덤분할 ──
USE_TEMPORAL_SPLIT = True
# 1년치 데이터 기준: 1~11월 학습, 12월(마지막 1개월) 검증
TEMPORAL_CUTOFF = "2024-12-01 00:00:00"

RUN_TS = datetime.now().strftime("%y.%m.%d.%H-%M-%S")
RUN_DIR = os.path.join(EXPERIMENTS_DIR, RUN_TS)
os.makedirs(RUN_DIR, exist_ok=True)

# ── LightGBM 저장 우회 함수: 한글 경로에서 save_model()이 실패하는 문제 회피 ──
def save_model_safely(booster, final_path):
    tmp_dir = tempfile.mkdtemp(prefix="lgbm_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(final_path))
    booster.save_model(tmp_path)
    shutil.copy(tmp_path, final_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

# ── 로그 저장 설정 ──────────────────────────────────
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

# ── STEP 1: 데이터 불러오기 (366개 일별 파일 → 병합, 캐시 활용) ──
categorical_cols = ["route_id","board_stop_id","alight_stop_id","weekday","weather","bus_type_code","usertype_code"]
numeric_cols = ["hour","is_holiday","headway_sec","seat_capacity"]
# 학습에 필요한 컬럼만 선택해서 읽음 — trip_round_id(거의 전부 고유값이라 메모리 최다 소모),
# bus_type_name, usertype_name은 학습에 안 쓰여서 제외
NEEDED_COLS = categorical_cols + numeric_cols + ["board_datetime", "is_standing", "standing_seconds", "sample_count_route_segment_hour"]

t0 = time.time()

if os.path.exists(MERGED_CACHE_PATH):
    print(f"병합 캐시 발견, 바로 로드: {MERGED_CACHE_PATH}")
    df = pd.read_parquet(MERGED_CACHE_PATH)
else:
    print(f"병합 캐시 없음 — {DATA_DIR}에서 일별 파일 병합 시작")
    print("⚠ 전체 기준 약 3.4억 행(서초구 1개월 데이터의 약 36배) — 시간이 오래 걸릴 수 있습니다")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "roster_*.parquet")))
    if SAMPLE_EVERY_N_DAYS:
        files = files[::SAMPLE_EVERY_N_DAYS]
        print(f"SAMPLE_EVERY_N_DAYS={SAMPLE_EVERY_N_DAYS} 적용 — {len(files)}개 파일만 사용")
    else:
        print(f"전체 {len(files)}개 파일 사용")

    dfs = []
    for i, fpath in enumerate(files):
        d = pd.read_parquet(fpath, columns=NEEDED_COLS)
        # dtype 다운캐스트로 메모리 절약
        d["hour"] = d["hour"].astype("int8")
        d["is_holiday"] = d["is_holiday"].astype("int8")
        d["seat_capacity"] = d["seat_capacity"].astype("int8")
        d["headway_sec"] = d["headway_sec"].astype("float32")
        d["standing_seconds"] = d["standing_seconds"].astype("float32")
        dfs.append(d)
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"  {i+1}/{len(files)} 파일 로드 완료 ({time.time()-t0:.1f}초 경과)")

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    print(f"병합 완료: {df.shape}, 캐시 저장 중...")
    df.to_parquet(MERGED_CACHE_PATH, index=False)
    print(f"캐시 저장 완료: {MERGED_CACHE_PATH}")

print(f"\n로딩 시간: {time.time()-t0:.1f}초")
print(df.shape)
print(df.isna().sum())

# ── STEP 2: 결측치 처리 ──────────────────────────────
df = df[df["alight_stop_id"].notna()]
print("결측 제외 후:", df.shape)

# ── STEP 3: 피처/타깃 정의 ──────────────────────────
for col in categorical_cols:
    df[col] = df[col].astype("category")

df["y_standing"] = (df["is_standing"] == "Y").astype(int)
print(df["y_standing"].value_counts())

feature_cols = categorical_cols + numeric_cols

# ── STEP 4: 학습/검증 분리 (랜덤분할 or 시간분할) ─────
if USE_TEMPORAL_SPLIT:
    print(f"\n검증 방식: 시간분할 (기준시각={TEMPORAL_CUTOFF})")
    cutoff = pd.Timestamp(TEMPORAL_CUTOFF)
    train_mask = df["board_datetime"] < cutoff
    X = df[feature_cols]
    y = df["y_standing"]
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    print(f"학습 행: {len(X_train):,} / 검증 행: {len(X_test):,}")
else:
    print("\n검증 방식: 랜덤분할 (random_state=42)")
    X = df[feature_cols]
    y = df["y_standing"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# ── STEP 5: 모델A 학습 (입석여부 분류) ───────────────
wandb.init(
    project="bus-standing-prediction",
    name=f"model_a_{DATASET_LABEL}_{N_ESTIMATORS}trees_{'temporal' if USE_TEMPORAL_SPLIT else 'random'}_{RUN_TS}",
    config={
        "model": "A_classifier",
        "dataset_label": DATASET_LABEL,
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

# ── STEP 6: 모델B 학습 (입석시간 회귀, 입석=Y만) ─────
standing_df = df[df["y_standing"] == 1]

if USE_TEMPORAL_SPLIT:
    train_mask_b = standing_df["board_datetime"] < cutoff
    Xb = standing_df[feature_cols]
    yb = standing_df["standing_seconds"]
    Xb_train, yb_train = Xb[train_mask_b], yb[train_mask_b]
    Xb_test, yb_test = Xb[~train_mask_b], yb[~train_mask_b]
    print(f"\n[모델B] 학습 행: {len(Xb_train):,} / 검증 행: {len(Xb_test):,}")
else:
    Xb = standing_df[feature_cols]
    yb = standing_df["standing_seconds"]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=42
    )

wandb.init(
    project="bus-standing-prediction",
    name=f"model_b_{DATASET_LABEL}_{N_ESTIMATORS}trees_{'temporal' if USE_TEMPORAL_SPLIT else 'random'}_{RUN_TS}",
    config={
        "model": "B_regressor",
        "dataset_label": DATASET_LABEL,
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
print(f"[모델B] MAE: {mae:.1f}초  RMSE: {rmse:.1f}초  (평균 입석시간: {yb.mean():.1f}초)")

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