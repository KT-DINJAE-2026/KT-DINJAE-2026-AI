import time
import os
import glob
import gc
import pandas as pd

# ── 데이터 설정 ──────────────────────────────────────
DATASET_LABEL = "seongbuk_2024"

# 366개 roster_YYYYMMDD.parquet 파일이 들어있는 폴더
DATA_DIR = r"D:\프로젝트\KT디지털인재장학생('26.03.20~present, KT)\지역사회 문제해결 프로젝트\roster-2024-final"

# 병합 결과 저장 위치
MERGED_CACHE_PATH = os.path.join(os.path.dirname(DATA_DIR), f"roster_{DATASET_LABEL}_merged.parquet")

# 개발/테스트용: None이면 366일 전체 사용, 정수(N)를 주면 N일 간격으로만 사용
SAMPLE_EVERY_N_DAYS = None

CHUNK_SIZE = 20  # 이 개수만큼 모아서 중간 병합 후 메모리 해제

categorical_cols = ["route_id","board_stop_id","alight_stop_id","weekday","weather","bus_type_code"]
numeric_cols = ["hour","is_holiday","headway_sec","seat_capacity"]
NEEDED_COLS = categorical_cols + numeric_cols + ["board_datetime", "is_standing", "standing_seconds", "sample_count_route_segment_hour"]

t0 = time.time()

if os.path.exists(MERGED_CACHE_PATH):
    print(f"이미 병합 파일이 있습니다: {MERGED_CACHE_PATH}")
    print("다시 만들고 싶으면 이 파일을 직접 지우고 재실행하세요.")
else:
    print(f"{DATA_DIR}에서 일별 파일 병합 시작")
    print("전체 기준 약 3.4억 행 — 시간이 오래 걸릴 수 있습니다")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "roster_*.parquet")))
    if SAMPLE_EVERY_N_DAYS:
        files = files[::SAMPLE_EVERY_N_DAYS]
        print(f"SAMPLE_EVERY_N_DAYS={SAMPLE_EVERY_N_DAYS} 적용 — {len(files)}개 파일만 사용")
    else:
        print(f"전체 {len(files)}개 파일 사용")

    chunk_frames = []
    merged_chunks = []

    for i, fpath in enumerate(files):
        d = pd.read_parquet(fpath, columns=NEEDED_COLS)
        d["hour"] = d["hour"].astype("int8")
        d["is_holiday"] = d["is_holiday"].astype("int8")
        d["seat_capacity"] = d["seat_capacity"].astype("int8")
        d["headway_sec"] = d["headway_sec"].astype("float32")
        d["standing_seconds"] = d["standing_seconds"].astype("float32")
        chunk_frames.append(d)

        if len(chunk_frames) >= CHUNK_SIZE or (i + 1) == len(files):
            merged_chunks.append(pd.concat(chunk_frames, ignore_index=True))
            chunk_frames = []
            gc.collect()

        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"  {i+1}/{len(files)} 파일 로드 완료 ({time.time()-t0:.1f}초 경과)")

    df = pd.concat(merged_chunks, ignore_index=True)
    del merged_chunks
    gc.collect()

    print(f"\n병합 완료: {df.shape}")
    print(df.isna().sum())

    print("저장 중...")
    df.to_parquet(MERGED_CACHE_PATH, index=False)
    print(f"저장 완료: {MERGED_CACHE_PATH}")
    print(f"총 소요시간: {time.time()-t0:.1f}초")