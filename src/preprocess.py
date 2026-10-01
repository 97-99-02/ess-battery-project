"""MIT-Stanford 배터리 .mat(HDF5) → 셀 단위 pickle 변환.

원본 .mat 은 배치당 2~3GB 이고 사이클마다 HDF5 참조를 따라가야 해서 읽기가 느리다.
EDA·모델링에 필요한 것만 한 번 뽑아 data/processed/ 에 저장해 두고 이후에는 pickle 만 읽는다.

추출 항목 (셀마다)
- cycle_life, charge_policy
- summary : 사이클별 스칼라 전체 (cycle, QD, QC, IR, Tavg, Tmin, Tmax, chargetime)
- Qdlin, Tdlin : 앞쪽 N_CYCLES 개 사이클의 전압축 보간 곡선 (1000 포인트)

로딩 방식은 논문 공식 저장소 BuildPkl_Batch*.ipynb 를 따름
(https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)
h5py 3.x 에서는 Dataset.value 가 없어져 [()] 로 바꿈.

사용법: python src/preprocess.py            # data/raw/*.mat 전부
        python src/preprocess.py 2017-05-12 # 파일명에 포함된 문자열로 골라서
"""

import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

# 사이클 곡선(Qdlin/Tdlin)은 앞쪽 몇 개만 저장한다. 과제 질문이 cycle 10·100 을 쓰므로 여유 있게.
N_CYCLES = 120

SUMMARY_FIELDS = {
    "cycle": "cycle",
    "QD": "QDischarge",
    "QC": "QCharge",
    "IR": "IR",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "Tmax": "Tmax",
    "chargetime": "chargetime",
}

BATCH_NAMES = {
    "2017-05-12": "b1",
    "2018-02-20": "b2",
    "2018-04-12": "b3",
    "2018-04-03": "bx",  # varcharge (extra)
}


def _read_summary(f, ref):
    g = f[ref]
    out = {}
    for short, field in SUMMARY_FIELDS.items():
        if field in g:
            out[short] = np.hstack(g[field][0, :].tolist()).astype(float)
    return out


def _curve(f, ref, length=1000):
    """비어 있는 사이클(예: Batch1 cycle 1)은 NaN 으로 채워 길이를 맞춘다."""
    arr = np.ravel(f[ref][()]).astype(float)
    if arr.size != length:
        return np.full(length, np.nan)
    return arr


def _read_curves(f, cycles, n):
    """cycles 그룹에서 앞쪽 n 개 사이클의 Qdlin, Tdlin 을 (n, 1000) 배열로. 행 j = cycles 의 j 번째."""
    total = cycles["Qdlin"].shape[0]
    n = min(n, total)
    qd = [_curve(f, cycles["Qdlin"][j, 0]) for j in range(n)]
    td = [_curve(f, cycles["Tdlin"][j, 0]) for j in range(n)]
    return np.vstack(qd), np.vstack(td), total


def convert(mat_path: Path):
    prefix = next((v for k, v in BATCH_NAMES.items() if k in mat_path.name), mat_path.stem)
    t0 = time.time()
    cells = {}
    with h5py.File(mat_path, "r") as f:
        batch = f["batch"]
        num_cells = batch["summary"].shape[0]
        for i in range(num_cells):
            cl = float(np.ravel(f[batch["cycle_life"][i, 0]][()])[0])
            policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()
            summary = _read_summary(f, batch["summary"][i, 0])
            qdlin, tdlin, n_total = _read_curves(f, f[batch["cycles"][i, 0]], N_CYCLES)
            cells[f"{prefix}c{i}"] = {
                "batch": prefix,
                "cell_idx": i,
                "cycle_life": cl,
                "charge_policy": policy,
                "summary": summary,
                "Qdlin": qdlin,
                "Tdlin": tdlin,
                "n_cycles_struct": n_total,
            }
            print(f"  {prefix}c{i:02d}  life={cl:7.1f}  policy={policy:<22s} cycles={n_total}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{prefix}.pkl"
    with open(out, "wb") as fp:
        pickle.dump(cells, fp)
    print(f"[{mat_path.name}] {len(cells)} cells → {out.name}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    keys = sys.argv[1:]
    mats = sorted(RAW_DIR.glob("*.mat"))
    if keys:
        mats = [m for m in mats if any(k in m.name for k in keys)]
    if not mats:
        sys.exit(f"no .mat in {RAW_DIR}")
    for m in mats:
        convert(m)
