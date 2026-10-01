"""Batch 1 분할 : 충전 정책 단위 Hold-out + 정책 단위 GroupKFold.

같은 정책 셀이 train / valid 에 나뉘어 들어가지 않도록 정책을 묶음(group)으로 다룬다.
Hold-out 은 정책별 평균 수명 순위로 5개 구간을 만들고 구간마다 정책 1개를 뽑는다
(수명 범위 전체가 valid 에 들어가도록 하는 층화). 수명값은 구간을 나누는 데만 쓰고
모델 성능은 보지 않는다.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 42


def holdout_policies(df_b1: pd.DataFrame, n_bins: int = 5, seed: int = SEED):
    pol = df_b1.groupby("policy").life.mean().sort_values()
    bins = pd.qcut(np.arange(len(pol)), n_bins, labels=False)
    rng = np.random.default_rng(seed)
    picked = [rng.choice(pol.index[bins == b]) for b in range(n_bins)]
    return sorted(picked)


def split_b1(df_b1: pd.DataFrame, seed: int = SEED):
    """(train_df, valid_df) 반환"""
    hold = holdout_policies(df_b1, seed=seed)
    valid = df_b1[df_b1.policy.isin(hold)]
    train = df_b1[~df_b1.policy.isin(hold)]
    return train, valid


def group_cv(n_splits: int = 5):
    return GroupKFold(n_splits=n_splits)
