import time
import os
import glob
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── 데이터 설정 ──────────────────────────────────────
DATASET_LABEL = "seongbuk_2024"

DATA_DIR = r"D:\프로젝트\KT디지털인재장학생('26.03.20~present, KT)\지역사회 문제해결 프로젝트\roster-2024-final"
MERGED_CACHE_PATH = os.path.join(os.path.dirname(DATA_DIR), f"roster_{DATASET_LABEL}_merged.parquet")

SAMPLE_EVERY_N_DAYS = None

categorical_cols = ["route_id","board_stop_id","alight_stop_id","weekday","weather","bus_type_code"]
numeric_cols = ["hour","is_holiday","headway_sec","seat_capacity"]
NEEDED_COLS = categorical_cols + numeric_cols + ["board_datetime", "is_standing", "standing_seconds", "sample_count_route_segment_hour"]

t0 = time.time()

if os.path.exists(MERGED_CACHE_PATH):
    print(f"이미 병합 파일이 있습니다: {MERGED_CACHE_PATH}")
    print("다시 만들고 싶으면 이 파일을 직접 지우고 재실행하세요.")
else:
    print(f"{DATA_DIR}에서 일별 파일 병합 시작 (스트리밍 방식 — 하루치씩만 메모리 사용)")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "roster_*.parquet")))
    if SAMPLE_EVERY_N_DAYS:
        files = files[::SAMPLE_EVERY_N_DAYS]
        print(f"SAMPLE_EVERY_N_DAYS={SAMPLE_EVERY_N_DAYS} 적용 — {len(files)}개 파일만 사용")
    else:
        print(f"전체 {len(files)}개 파일 사용")

    writer = None
    total_rows = 0
    try:
        for i, fpath in enumerate(files):
            d = pd.read_parquet(fpath, columns=NEEDED_COLS)
            d["hour"] = d["hour"].astype("int8")
            d["is_holiday"] = d["is_holiday"].astype("int8")
            d["seat_capacity"] = d["seat_capacity"].astype("int8")
            d["headway_sec"] = d["headway_sec"].astype("float32")
            d["standing_seconds"] = d["standing_seconds"].astype("float32")

            table = pa.Table.from_pandas(d, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(MERGED_CACHE_PATH, table.schema)

            writer.write_table(table)
            total_rows += len(d)

            del d, table
            gc.collect()

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"  {i+1}/{len(files)} 파일 처리 완료, 누적 {total_rows:,}행 ({time.time()-t0:.1f}초 경과)")
    finally:
        if writer is not None:
            writer.close()

    print(f"\n병합 완료: 총 {total_rows:,}행")
    print(f"저장 완료: {MERGED_CACHE_PATH}")
    print(f"총 소요시간: {time.time()-t0:.1f}초")