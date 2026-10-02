"""DAY 2 모델 학습·평가. DAY 1 설계서(docs/DS-MINI-Design-*.pdf) S3·S4·S5 를 그대로 구현한다.

순서
1. 셀 표 (features.load_table) : Batch 1 41셀 / Batch 2 39셀 / Batch 3 40셀. 원본 변환본이 없으면 results/cell_features.csv
2. Batch 1 정책 단위 분할 (split.split_b1) : train 32셀·17정책 / valid 9셀·5정책
3. Train : train 32셀에서 정책 단위 GroupKFold(5) MAPE. 튜닝이 있는 모델은 각 폴드 안 GroupKFold(4) (nested)
4. Valid : train 32셀로 학습한 모델로 valid 9셀을 평가. 셀 단위 부트스트랩 2000회(시드 42) 백분위 95% 구간
5. Test  : 같은 절차를 Batch 1 41셀 전체로 다시 학습한 최종 모델로 Batch 2·3 를 한 번 평가
6. 예측구간 : train CV 폴드 밖 잔차(log10)의 10·90% 분위수로 만든 80% 구간

타깃은 log10(cycle life), 예측은 10^x 로 되돌려 MAPE 를 계산한다 (역변환값은 중앙값 쪽 예측).
테스트 결과를 보고 모델·피처를 바꾸지 않는다. 아래 MODELS 는 설계서에 적힌 목록 그대로다.

사용법 : python src/train.py                # 원본 변환본(data/processed)이 있으면 그것으로, 없으면 results/cell_features.csv 로
        python src/train.py --source csv   # 원본 없이 cell_features.csv 만으로 재현
산출물 : results/model_performance.csv (주 모델 성능표, 과제 포맷)
        results/model_comparison.csv  (전체 모델 비교)
        results/predictions.csv       (셀별 예측 : CV 폴드 밖 / valid / Batch 2 / Batch 3)
        results/cv_folds.csv          (폴드별 MAPE)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBRegressor

from features import load_table
from split import SEED, split_b1

# ElasticNet 튜닝 격자의 아주 작은 alpha 에서 나는 수렴 경고 (선택 모델만 해당, 고른 값과 무관)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# ── 목표 성능 (원논문 Table 1 재구성, 설계서 부록 A1)
TARGET = 9.1            # 과제 Target : full 모델, 이례 셀 1개 제외 primary 7.5% 와 secondary 10.7% 를 셀 수(42·40)로 가중
PAPER_VARIANCE = 12.3   # 같은 방식으로 재구성한 variance 모델 (주 모델 F1 과 같은 구조)
PAPER_DISCHARGE = 9.4   # 같은 방식으로 재구성한 discharge 모델 (F3 와 같은 구성)
PAPER_SECONDARY_VAR = 11.4  # variance 모델의 secondary test (2018-04-12 배치 = Batch 3)

# ── 피처 세트 (설계서 S1, 부록 A14). F2·F3·F4 는 첨도·2V 를 논문 정의로 바꾼 열을 쓴다.
DQ_P = ["dq_var", "dq_min", "dq_mean", "dq_skew", "dq_kurt_p"]
CAP = ["qd_c2", "qd_max_minus_c2", "qd_c100", "fade_slope_2_100", "fade_int_2_100",
       "fade_slope_91_100", "fade_int_91_100"]
OTHER = ["chargetime_avg_2_6", "tmax_max", "tmin_min", "tavg_mean", "ir_c2", "ir_min", "ir_c100_minus_c2"]

# ── 튜닝 범위 (설계서 부록 A15)
EN_GRID = {"m__alpha": np.logspace(-4, 0, 13), "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
RIDGE_GRID = {"m__alpha": np.logspace(-3, 2, 16)}
RF_GRID = {"min_samples_leaf": [1, 2, 4], "max_features": [0.5, 1.0]}
XGB_GRID = {"max_depth": [2, 3], "min_child_weight": [1, 3]}


class MeanLog(BaseEstimator, RegressorMixin):
    """기준선 : 학습셋 log10 수명의 평균으로 모든 셀을 예측 (설계서 부록 A15 의 '학습 평균')."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def scaled(est):
    return Pipeline([("sc", StandardScaler()), ("m", est)])


def linear():
    return scaled(LinearRegression()), None


def enet():
    return scaled(ElasticNet(max_iter=50000, random_state=SEED)), EN_GRID


def ridge():
    return scaled(Ridge()), RIDGE_GRID


def rf():
    return RandomForestRegressor(n_estimators=300, random_state=SEED), RF_GRID


def xgb():
    return XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED, verbosity=0), XGB_GRID


def mean_log():
    return MeanLog(), None


# ── 모델 목록 (설계서 S5 'DAY 2 할 일' 의 필수 / 선택)
#   test=False : Batch 1 비교에만 쓰는 세트 (F4 는 Batch 2 내부저항 6셀 결측)
MODELS = [
    dict(name="F1", role="주 모델", cols=["dq_var"], make=linear, test=True, paper=PAPER_VARIANCE),
    dict(name="F1 + qd_c2_bc", role="확장 1순위", cols=["dq_var", "qd_c2_bc"], make=linear, test=True),
    dict(name="2피처 (F1 + qd_c2)", role="비교", cols=["dq_var", "qd_c2"], make=linear, test=True),
    dict(name="기준선 : 학습 평균", role="기준선", cols=["dq_var"], make=mean_log, test=True),
    dict(name="기준선 : 용량 곡선 7개 ElasticNet", role="기준선", cols=CAP, make=enet, test=True),
    dict(name="F1 + qd_c2_bz", role="선택 : 표준화 민감도", cols=["dq_var", "qd_c2_bz"], make=linear, test=True),
    dict(name="F2 ElasticNet", role="선택 : 확장 후보", cols=DQ_P, make=enet, test=True),
    dict(name="F2 Ridge", role="선택 : 규제 비교", cols=DQ_P, make=ridge, test=True),
    dict(name="F3 ElasticNet", role="선택 : 확장 후보", cols=DQ_P + ["dq_2v_p"] + CAP, make=enet, test=True,
         paper=PAPER_DISCHARGE),
    dict(name="F3 Ridge", role="선택 : 규제 비교", cols=DQ_P + ["dq_2v_p"] + CAP, make=ridge, test=True),
    dict(name="F4 ElasticNet", role="선택 : Batch 1 비교만", cols=DQ_P + ["dq_2v_p"] + CAP + OTHER, make=enet, test=False),
    dict(name="F1 RandomForest", role="선택 : 비선형 비교", cols=["dq_var"], make=rf, test=True),
    dict(name="F1 XGBoost", role="선택 : 비선형 비교", cols=["dq_var"], make=xgb, test=True),
]


def mape(y_log, p_log):
    t, p = 10 ** np.asarray(y_log), 10 ** np.asarray(p_log)
    return float(np.mean(np.abs(p - t) / t) * 100)


def fit(spec, df, inner_splits):
    """설계서 S4 '5 학습' : Pipeline(Scaler → 모델). 튜닝은 같은 학습셋 안의 정책 단위 GroupKFold."""
    est, grid = spec["make"]()
    X, y = df[spec["cols"]].values, df["log_life"].values
    if grid:
        gs = GridSearchCV(est, grid, cv=GroupKFold(n_splits=inner_splits), scoring="neg_mean_absolute_error")
        gs.fit(X, y, groups=df["policy"].values)
        return gs.best_estimator_, gs.best_params_
    return est.fit(X, y), {}


def run_cv(spec, train):
    """Train 행 : train 32셀 정책 단위 GroupKFold(5). 폴드 밖 예측(log10)도 돌려준다."""
    X, y, g = train[spec["cols"]].values, train["log_life"].values, train["policy"].values
    oof, scores = np.zeros(len(y)), []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        model, _ = fit(spec, train.iloc[tr], inner_splits=4)
        oof[te] = model.predict(X[te])
        scores.append(mape(y[te], oof[te]))
    return np.array(scores), oof


def bootstrap_ci(y_log, p_log, n=2000, seed=SEED):
    """셀 단위 부트스트랩 백분위 95% 구간 (설계서 S4 구현 명세)"""
    rng = np.random.default_rng(seed)
    y_log, p_log = np.asarray(y_log), np.asarray(p_log)
    idx = rng.integers(0, len(y_log), size=(n, len(y_log)))
    vals = [mape(y_log[i], p_log[i]) for i in idx]
    return np.percentile(vals, [2.5, 97.5])


def evaluate(spec, train, valid, b1, tests):
    scores, oof = run_cv(spec, train)
    resid = train["log_life"].values - oof
    q_lo, q_hi = np.quantile(resid, [0.10, 0.90])  # 80% 예측구간 (log10)

    m32, _ = fit(spec, train, inner_splits=5)
    p_valid = m32.predict(valid[spec["cols"]].values)
    ci = bootstrap_ci(valid["log_life"], p_valid)

    rows = [dict(split="cv_oof", cell=train.cell, y=train.log_life, p=oof),
            dict(split="valid", cell=valid.cell, y=valid.log_life, p=p_valid)]
    out = dict(model=spec["name"], role=spec["role"], features=" + ".join(spec["cols"]), n_feat=len(spec["cols"]),
               cv_mape=scores.mean(), cv_std=scores.std(), valid_mape=mape(valid.log_life, p_valid),
               valid_ci_lo=ci[0], valid_ci_hi=ci[1], pi_q10=q_lo, pi_q90=q_hi,
               paper_ref=spec.get("paper", np.nan))  # 같은 구성의 원논문 모델을 Target 과 같은 방식으로 재구성한 값

    if spec["test"]:
        m41, params = fit(spec, b1, inner_splits=5)
        out["final_params"] = str(params) if params else ""
        if isinstance(m41, Pipeline) and isinstance(m41.named_steps["m"], LinearRegression):
            # 표준화 → 선형회귀를 원래 단위의 식으로 : log10(수명) = intercept + Σ coef·x
            sc, lr = m41.named_steps["sc"], m41.named_steps["m"]
            coef = lr.coef_ / sc.scale_
            out["final_formula"] = f"log10(life) = {lr.intercept_ - (coef * sc.mean_).sum():.4f} " + " ".join(
                f"{c:+.4f}·{n}" for c, n in zip(coef, spec["cols"]))
        for name, df in tests.items():
            p = m41.predict(df[spec["cols"]].values)
            rows.append(dict(split=name, cell=df.cell, y=df.log_life, p=p))
            out[f"{name}_mape"] = mape(df.log_life, p)
    preds = pd.concat([pd.DataFrame({"split": r["split"], "cell": r["cell"].values, "y_log": np.asarray(r["y"]),
                                     "p_log": np.asarray(r["p"])}) for r in rows], ignore_index=True)
    preds["model"] = spec["name"]
    preds["pi_lo_log"], preds["pi_hi_log"] = preds.p_log + q_lo, preds.p_log + q_hi
    folds = pd.DataFrame({"model": spec["name"], "fold": range(1, 6), "mape": scores})
    return out, preds, folds


def format_table(r):
    """과제 Performance Reporting 포맷 (Regression, Batch 3 추가 포함) 과 같은 2단 구조.
    Gap(Train-Valid)·(Valid-Test)·(Target-Test) 는 '뒤 - 앞' (예 : Train-Valid = Valid - Train) 으로 계산해
    MAPE 에서 (+) 가 나빠짐이 되게 한다. Gap(Batch2-Batch3) 는 이름대로 Batch 2 - Batch 3 ((+) : Batch 2 가 나쁨)."""
    rows = [
        ("Train (Batch 1 CV)", "", r.cv_mape, f"train 32셀 정책 단위 GroupKFold(5), 폴드 표준편차 {r.cv_std:.1f}"),
        ("Valid (Batch 1 Hold-out)", "", r.valid_mape,
         f"9셀·5정책 (train 32셀 모델), 부트스트랩 95% 구간 {r.valid_ci_lo:.1f}~{r.valid_ci_hi:.1f}"),
        ("Test (Batch 2)", "", r.b2_mape, "Batch 1 41셀로 다시 학습한 최종 모델, 한 번 평가"),
        ("", "Gap (Train-Valid)", r.valid_mape - r.cv_mape, "Valid - Train. (+) : 과적합 의심"),
        ("", "Gap (Valid-Test)", r.b2_mape - r.valid_mape, "Test - Valid. (+) : 배치간 일반화 저하 의심"),
        ("", "Gap (Target-Test)", r.b2_mape - TARGET,
         f"Test - Target. Target : 원논문 9.1%. 같은 구조 variance 모델 재구성값 {PAPER_VARIANCE} 대비 {r.b2_mape - PAPER_VARIANCE:+.2f}"),
        ("Test (Batch 3)", "", r.b3_mape, "추가 검증. 규칙 ① 에 쓰지 않은 배치"),
        ("", "Gap (Batch2-Batch3)", r.b2_mape - r.b3_mape, "Batch 2 - Batch 3. Test 성능 간 비교, (+) : Batch 2 가 나쁨"),
        ("", "Gap (Target-Test)", r.b3_mape - TARGET,
         f"Batch 3 기준 Test - Target(9.1%). Batch 3 는 원논문 secondary test 와 같은 배치, variance 모델 secondary {PAPER_SECONDARY_VAR} 대비 {r.b3_mape - PAPER_SECONDARY_VAR:+.2f}"),
    ]
    return pd.DataFrame(rows, columns=["구분", "세부", "MAPE (%)", "비고"]).round({"MAPE (%)": 2})


def main(source="auto"):
    df = load_table(source)
    b1 = df[df.batch == "b1"].reset_index(drop=True)
    train, valid = split_b1(b1)
    train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
    tests = {"b2": df[df.batch == "b2"].reset_index(drop=True), "b3": df[df.batch == "b3"].reset_index(drop=True)}
    print(f"train {len(train)}셀·{train.policy.nunique()}정책 / valid {len(valid)}셀·{valid.policy.nunique()}정책 / "
          f"Batch 1 {len(b1)} / Batch 2 {len(tests['b2'])} / Batch 3 {len(tests['b3'])}")

    results, preds, folds = [], [], []
    for spec in MODELS:
        out, p, f = evaluate(spec, train, valid, b1, tests)
        results.append(out); preds.append(p); folds.append(f)
        msg = f"{spec['name']:<34s} CV {out['cv_mape']:5.2f} ± {out['cv_std']:4.2f}  Valid {out['valid_mape']:5.2f}"
        if spec["test"]:
            msg += f"  B2 {out['b2_mape']:6.2f}  B3 {out['b3_mape']:6.2f}"
        print(msg)

    # 반올림은 한 번만 : 비교표와 성능표가 같은 값을 쓰도록 (예측구간 분위수는 log10 이라 4자리)
    comp = pd.DataFrame(results)
    num = comp.select_dtypes("number").columns.difference(["pi_q10", "pi_q90", "n_feat"])
    comp[num] = comp[num].round(2)
    comp[["pi_q10", "pi_q90"]] = comp[["pi_q10", "pi_q90"]].round(4)
    preds = pd.concat(preds, ignore_index=True)
    meta = df[["cell", "batch", "policy", "newstructure", "carryover", "dq_var", "qd_c2", "life"]]
    preds = preds.merge(meta, on="cell", how="left")
    RES.mkdir(exist_ok=True)
    comp.to_csv(RES / "model_comparison.csv", index=False)
    preds.round(5).to_csv(RES / "predictions.csv", index=False)
    pd.concat(folds).round(4).to_csv(RES / "cv_folds.csv", index=False)
    main_row = comp[comp.role == "주 모델"].iloc[0]
    format_table(main_row).to_csv(RES / "model_performance.csv", index=False)
    print(format_table(main_row).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "pkl", "csv"], default="auto",
                    help="셀 표를 어디서 읽을지 (auto : 변환본이 있으면 pkl, 없으면 csv)")
    main(ap.parse_args().source)
