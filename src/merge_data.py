import time
import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── 데이터 설정 ──────────────────────────────────────
DATASET_LABEL = "seongbuk_2024"

DATA_DIR = r"D:\프로젝트\KT디지털인재장학생('26.03.20~present, KT)\지역사회 문제해결 프로젝트\roster-2024-final"

# 회원님 노트북(16GB RAM, 실사용 15.7GB) 기준:
# - 366일 전체를 범주형으로 바꿔도 최종 크기는 약 10.9GB지만,
#   학습 스크립트가 X_train/X_test를 만드는 순간 원본+복사본이 동시에 떠서 최대 약 16.7GB까지 올라감 (한계 초과 위험)
# - 2일 간격(약 183일, 반년치)로 줄이면 최대 사용량이 약 8.2GB로 떨어져 안전하게 돌아감
# 여유가 되면 1로 늘려서 시도해볼 수 있지만, 다른 프로그램을 다 끄고 하시는 걸 추천
SAMPLE_EVERY_N_DAYS = 2

categorical_cols = ["route_id","board_stop_id","alight_stop_id","weekday","weather","bus_type_code"]
numeric_cols = ["hour","is_holiday","headway_sec","seat_capacity"]
NEEDED_COLS = categorical_cols + numeric_cols + ["board_datetime", "is_standing", "standing_seconds", "sample_count_route_segment_hour"]

suffix = f"_every{SAMPLE_EVERY_N_DAYS}d" if SAMPLE_EVERY_N_DAYS and SAMPLE_EVERY_N_DAYS > 1 else "_full"
MERGED_CACHE_PATH = os.path.join(os.path.dirname(DATA_DIR), f"roster_{DATASET_LABEL}{suffix}_merged.parquet")

t0 = time.time()

if os.path.exists(MERGED_CACHE_PATH):
    print(f"이미 병합 파일이 있습니다: {MERGED_CACHE_PATH}")
    print("다시 만들고 싶으면 이 파일을 직접 지우고 재실행하세요.")
else:
    print(f"{DATA_DIR}에서 일별 파일 병합 시작 (스트리밍 방식, 메모리에 전체를 안 올림)")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "roster_*.parquet")))
    if SAMPLE_EVERY_N_DAYS and SAMPLE_EVERY_N_DAYS > 1:
        files = files[::SAMPLE_EVERY_N_DAYS]
        print(f"SAMPLE_EVERY_N_DAYS={SAMPLE_EVERY_N_DAYS} 적용 — {len(files)}개 파일만 사용")
    else:
        print(f"전체 {len(files)}개 파일 사용")

    writer = None
    total_rows = 0

    for i, fpath in enumerate(files):
        d = pd.read_parquet(fpath, columns=NEEDED_COLS)

        # dtype 다운캐스트
        d["hour"] = d["hour"].astype("int8")
        d["is_holiday"] = d["is_holiday"].astype("int8")
        d["seat_capacity"] = d["seat_capacity"].astype("int8")
        d["headway_sec"] = d["headway_sec"].astype("float32")
        d["standing_seconds"] = d["standing_seconds"].astype("float32")
        d["sample_count_route_segment_hour"] = d["sample_count_route_segment_hour"].astype("int32")

        # 범주형 조기 변환 — 이걸 지금(파일 단위) 해둬야 나중에 학습 스크립트가
        # 이 병합 파일을 읽을 때 문자열 그대로가 아니라 압축된 형태로 바로 로드됨
        for c in categorical_cols:
            d[c] = d[c].astype("category")
        d["is_standing"] = d["is_standing"].astype("category")

        table = pa.Table.from_pandas(d, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(MERGED_CACHE_PATH, table.schema)
        writer.write_table(table)

        total_rows += len(d)
        del d, table

        if (i + 1) % 30 == 0 or (i + 1) == len(files):
            print(f"  {i+1}/{len(files)} 파일 처리 완료, 누적 {total_rows:,}행 ({time.time()-t0:.1f}초 경과)")

    writer.close()
    print(f"\n병합 완료: 총 {total_rows:,}행")
    print(f"저장 위치: {MERGED_CACHE_PATH}")
    print(f"총 소요시간: {time.time()-t0:.1f}초")