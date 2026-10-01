"""셀 단위 데이터(data/processed/*.pkl) → EDA·모델링용 테이블.

- load_cells   : pkl 로드 + 셀 상태 플래그(제외 사유, 수명 보정 여부)
- parse_policy : 충전 프로토콜 문자열 → C1, Q1, C2, 0~80% 평균 C-rate
- cell_table   : 셀당 1행 (수명, 정책, 초기 사이클 피처)

사이클 번호와 배열 위치 : summary['cycle'] 은 세 배치 모두 1 부터 시작하고,
Qdlin 의 행 j 는 cycle j+1 이다. Batch 1 은 cycle 1 이 비어 있다(QD=0, Qdlin=NaN).
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

# 세 배치 공통 전압축 (Vdlin) : 3.5V → 2.0V, 1000 포인트
VDLIN = np.linspace(3.5, 2.0, 1000)
EOL_AH = 0.88  # 공칭 1.1Ah 의 80%

# --- 셀 정리 기준 : 논문 공식 저장소 Load Data.ipynb -------------------------------------
# Batch1 : 80% 용량에 도달하기 전에 시험이 끝난 셀 (데이터상 최소 QD 0.91~0.97Ah)
B1_NOT_FINISHED = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
# Batch1 : 2017-06-30 배치에서 이어 측정된 셀과 추가 사이클 수 (이 파일만으로는 수명이 잘려 있음)
B1_CARRYOVER_ADD = {"b1c0": 662, "b1c1": 981, "b1c2": 1060, "b1c3": 208, "b1c4": 482}
# Batch3 : 공식 저장소에서 noisy channel 로 제외한 셀 (c23, c32 는 수명값도 NaN)
B3_NOISY = ["b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"]

BATCH_LABEL = {"b1": "Batch 1", "b2": "Batch 2", "b3": "Batch 3"}
BATCH_DATE = {"b1": "2017-05-12", "b2": "2018-02-20", "b3": "2018-04-12"}


def load_cells(names=("b1", "b2", "b3")):
    cells = {}
    for n in names:
        with open(PROC_DIR / f"{n}.pkl", "rb") as fp:
            cells.update(pickle.load(fp))
    for k, c in cells.items():
        c["life_raw"] = c["cycle_life"]
        c["life"] = c["cycle_life"] + B1_CARRYOVER_ADD.get(k, 0)
        c["carryover"] = k in B1_CARRYOVER_ADD
        if np.isnan(c["cycle_life"]):
            c["exclude"] = "no_life"  # VarCharge / SLOWCYCLE / 80% 미도달
        elif k in B1_NOT_FINISHED:
            c["exclude"] = "not_finished"
        elif k in B3_NOISY:
            c["exclude"] = "noisy"
        else:
            c["exclude"] = ""
    return cells


_POLICY_RE = re.compile(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C")


def parse_policy(policy: str) -> dict:
    """'5.4C(40%)-3.6C' → C1=5.4, Q1=40, C2=3.6, avg_crate(0~80% SOC 평균 충전 속도)."""
    m = _POLICY_RE.search(policy)
    if not m:
        return {"C1": np.nan, "Q1": np.nan, "C2": np.nan, "avg_crate": np.nan}
    c1, q1, c2 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    q1f = min(q1, 80.0) / 100.0
    # 0→80% SOC 충전 시간(h) = Q1/C1 + (0.8-Q1)/C2 → 평균 C-rate = 0.8 / 시간
    hours = q1f / c1 + max(0.8 - q1f, 0.0) / c2
    return {"C1": c1, "Q1": q1, "C2": c2, "avg_crate": 0.8 / hours}


def _pos(cycle):
    """cycle 번호 → 배열 위치"""
    return int(cycle) - 1


def _clean_qd(qd):
    """측정 스파이크(0 또는 1.3Ah 초과) 제거"""
    qd = qd.astype(float).copy()
    qd[(qd < 0.5) | (qd > 1.3)] = np.nan
    return qd


def delta_q(cell, hi=100, lo=10):
    """ΔQ_hi-lo(V) = Q_hi(V) - Q_lo(V), 길이 1000"""
    return cell["Qdlin"][_pos(hi)] - cell["Qdlin"][_pos(lo)]


def _fit_line(x, y):
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return slope, intercept


def early_features(cell, hi=100, lo=10):
    """초기 사이클(<= hi)만으로 계산하는 셀 단위 피처."""
    s = cell["summary"]
    cyc = s["cycle"]
    qd = _clean_qd(s["QD"])
    win = (cyc >= 2) & (cyc <= hi)

    dq = delta_q(cell, hi, lo)
    f = {
        # ΔQ(V) 통계량 : 크기가 작아 log10(|x|) 로 사용
        "dq_min": np.log10(abs(np.nanmin(dq))),
        "dq_mean": np.log10(abs(np.nanmean(dq))),
        "dq_var": np.log10(np.nanvar(dq)),
        "dq_skew": np.log10(abs(skew(dq, nan_policy="omit"))),
        "dq_kurt": np.log10(abs(kurtosis(dq, nan_policy="omit"))),
        "dq_2v": dq[-1],  # 2.0V 지점의 ΔQ
    }
    # 방전 용량 열화
    f["qd_c2"] = qd[_pos(2)]
    f["qd_max_minus_c2"] = np.nanmax(qd[win]) - qd[_pos(2)]
    f["qd_c100"] = qd[_pos(hi)]
    f["qd_c100_minus_c2"] = qd[_pos(hi)] - qd[_pos(2)]
    f["fade_slope_2_100"], f["fade_int_2_100"] = _fit_line(cyc[win], qd[win])
    late = (cyc >= hi - 9) & (cyc <= hi)
    f["fade_slope_91_100"], f["fade_int_91_100"] = _fit_line(cyc[late], qd[late])
    # 충전 시간 · 온도 · 내부저항
    ct = s["chargetime"].astype(float)
    f["chargetime_avg_2_6"] = np.nanmean(ct[(cyc >= 2) & (cyc <= 6)])
    f["tmax_max"] = np.nanmax(s["Tmax"][win])
    f["tmin_min"] = np.nanmin(s["Tmin"][win])
    f["tavg_mean"] = np.nanmean(s["Tavg"][win])
    ir = s["IR"].astype(float)
    ir[ir <= 0] = np.nan
    f["ir_c2"] = ir[_pos(2)]
    f["ir_min"] = np.nanmin(ir[win])
    f["ir_c100_minus_c2"] = ir[_pos(hi)] - ir[_pos(2)]
    return f


def cell_table(cells, hi=100, lo=10):
    rows = []
    for k, c in cells.items():
        row = {
            "cell": k,
            "batch": c["batch"],
            "policy": c["charge_policy"],
            "newstructure": "newstructure" in c["charge_policy"],
            "life_raw": c["life_raw"],
            "life": c["life"],
            "carryover": c["carryover"],
            "exclude": c["exclude"],
            "n_cycles_struct": c["n_cycles_struct"],
        }
        row.update(parse_policy(c["charge_policy"]))
        if c["n_cycles_struct"] >= hi and not np.isnan(c["Qdlin"][_pos(hi)]).all():
            row.update(early_features(c, hi, lo))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["log_life"] = np.log10(df["life"])
    return df


def knee_point(cell):
    """방전 용량 곡선의 knee : 시작점과 끝점을 잇는 직선에서 가장 멀리 떨어진 지점(아래쪽).
    QD 는 21사이클 이동 중앙값으로 평활한 뒤 계산한다. 반환 (knee cycle, 평활 QD 시리즈)."""
    s = cell["summary"]
    cyc = s["cycle"]
    qd = pd.Series(_clean_qd(s["QD"]), index=cyc)
    life = cell["life_raw"]
    qd = qd[(qd.index >= 2) & (qd.index <= life)].dropna()
    sm = qd.rolling(21, center=True, min_periods=5).median().dropna()
    if len(sm) < 30:
        return np.nan, sm
    x, y = sm.index.values.astype(float), sm.values
    chord = y[0] + (y[-1] - y[0]) * (x - x[0]) / (x[-1] - x[0])
    gap = chord - y  # 곡선이 직선보다 위에 있으면 음수
    return x[np.argmin(gap)], sm
