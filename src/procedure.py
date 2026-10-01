"""시험 절차 점검 : 원본 .mat 의 사이클 시계열에서 사이클 길이와 휴지(전류 0) 시간을 잰다.

newstructure 표기가 실제로 다른 시험 절차인지 라벨 없이 확인하려는 용도다.
원본 .mat 이 있어야 하므로 결과를 results/procedure_timing.csv 로 저장해 두고 노트북은 csv 만 읽는다.

사용법: python src/procedure.py          # data/raw 의 세 배치, cycle 10
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "results" / "procedure_timing.csv"

FILES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "b2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
REST_AMP = 0.01  # |I| 가 이보다 작으면 휴지로 본다 (전류 단위는 원본 그대로)


def _segments(rest, dt):
    """연속된 휴지 구간의 길이(분) 목록"""
    out, cur = [], 0.0
    for r, d in zip(rest, dt):
        if r:
            cur += d
        elif cur > 0:
            out.append(cur)
            cur = 0.0
    if cur > 0:
        out.append(cur)
    return [x for x in out if x > 0.01]


def cycle_timing(cycle=10):
    rows = []
    for b, name in FILES.items():
        with h5py.File(RAW_DIR / name, "r") as f:
            batch = f["batch"]
            for i in range(batch["cycles"].shape[0]):
                cyc = f[batch["cycles"][i, 0]]
                j = cycle - 1  # 행 j = cycle j+1 (features.py 와 같은 규칙)
                if cyc["t"].shape[0] <= j:
                    continue
                t = np.ravel(f[cyc["t"][j, 0]][()]).astype(float)
                cur = np.ravel(f[cyc["I"][j, 0]][()]).astype(float)
                if t.size < 10:
                    continue
                dt = np.diff(t)
                rest = np.abs(cur[:-1]) < REST_AMP
                seg = _segments(rest, dt)
                rows.append({
                    "cell": f"{b}c{i}", "batch": b,
                    "policy": f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode(),
                    "cycle_min": t[-1] - t[0],
                    "rest_min": dt[rest].sum(),
                    "n_rest": len(seg),
                    "longest_rest_min": max(seg) if seg else 0.0,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cyc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    df = cycle_timing(cyc)
    df.round(3).to_csv(OUT, index=False)
    print(f"saved {OUT.name} : {len(df)} cells, cycle {cyc}")
