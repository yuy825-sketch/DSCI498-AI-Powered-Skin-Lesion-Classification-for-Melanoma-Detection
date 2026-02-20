from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def group_shuffle_split(
    *,
    groups: Iterable[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Split:
    groups = np.asarray(list(groups))
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    unique_groups = np.unique(groups)
    r = _rng(seed)
    r.shuffle(unique_groups)

    n = len(unique_groups)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train : n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val :])

    idx = np.arange(len(groups))
    train_idx = idx[np.array([g in train_groups for g in groups])]
    val_idx = idx[np.array([g in val_groups for g in groups])]
    test_idx = idx[np.array([g in test_groups for g in groups])]
    return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

