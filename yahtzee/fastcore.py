"""NumPy-vectorised inner loop for the turn DP.

The expensive part of evaluating a turn is, given the value ``v0[f]`` of every
final roll ``f`` (252 of them), computing

    value = sum_over_opening_rolls  P(roll) * max_over_keep_options  E_keep

where ``E_keep = sum_over_reroll_outcomes P(outcome) * v0[outcome]``.

``E_keep`` is a fixed linear map of ``v0`` (a 462x252 matrix), and the per-roll
maximisation is a segmented max.  Both are precomputed once here so each turn
evaluation is two vectorised NumPy operations instead of ~30k Python multiplies.
"""

from typing import List

import numpy as np

from .dice import (ALL_ROLLS, KEEP_OPTS, REROLL_IDX, ROLL_PROB_LIST,
                   _all_count_vectors, NUM_DICE)

# All distinct keep options (sub-multisets of size 0..5): 462 of them.
_KEPT_LIST = []
for _k in range(NUM_DICE + 1):
    _KEPT_LIST.extend(_all_count_vectors(_k))
_KEPT_INDEX = {kept: i for i, kept in enumerate(_KEPT_LIST)}
N_KEPT = len(_KEPT_LIST)
N_FULL = len(ALL_ROLLS)

# K[kept_row, full_idx] = P(reaching full_idx by rerolling from kept)
_K = np.zeros((N_KEPT, N_FULL), dtype=np.float64)
for _kept, _row in _KEPT_INDEX.items():
    for _full_idx, _prob in REROLL_IDX[_kept]:
        _K[_row, _full_idx] = _prob

# Segmented representation of "for each opening roll, its keep-option rows".
_flat_rows: List[int] = []
_seg_starts: List[int] = []
for _ri in range(N_FULL):
    _seg_starts.append(len(_flat_rows))
    for _kept in KEEP_OPTS[_ri]:
        _flat_rows.append(_KEPT_INDEX[_kept])
_FLAT_ROWS = np.array(_flat_rows, dtype=np.intp)
_SEG_STARTS = np.array(_seg_starts, dtype=np.intp)
_ROLL_PROB = np.array(ROLL_PROB_LIST, dtype=np.float64)

# For each opening roll, the kept rows of its keep options (for argmax by value).
ROLL_KEEP_ROWS = [[_KEPT_INDEX[kept] for kept in KEEP_OPTS[ri]]
                  for ri in range(N_FULL)]
# Public handles used by the vectorised opponent-distribution DP.
K = _K
ROLL_PROB = _ROLL_PROB


def e_kept_from_v0(v0) -> "np.ndarray":
    """The 462-vector of expected rr=0 values for each keep option."""
    return _K.dot(np.asarray(v0, dtype=np.float64))


def turn_value_from_v0(v0) -> float:
    """Vectorised turn value given the 252-length ``v0`` (indexed by
    FULL_INDEX)."""
    v = np.asarray(v0, dtype=np.float64)
    e_kept = _K.dot(v)                       # (462,)
    per_roll_max = np.maximum.reduceat(e_kept[_FLAT_ROWS], _SEG_STARTS)
    return float(_ROLL_PROB.dot(per_roll_max))


# ---- vectorised expected-score v0 builder ----------------------------------
# For the expected-score DP, the value of a final roll is
#   v0[f] = max_c ( raw_score(c, f) + eadd(child(c, f)) )
# and eadd(child) is CONSTANT across rolls for every category except the upper
# boxes (child upper total varies with the face count) and the six five-of-a-kind
# rolls (joker rules).  Precompute the per-(category, roll) raw scores and face
# counts so each state's v0 is a handful of vectorised array ops.

from . import scoring as _S  # noqa: E402

# RAW[c, f] = normal score of roll f in category c (no joker/bonus).
_RAW = np.array([[_S.raw_score(c, full) for full in ALL_ROLLS]
                 for c in range(_S.NUM_CATEGORIES)], dtype=np.float64)
# CNT[c, f] = number of dice showing face (c+1) in roll f  (upper cats only use this).
_CNT = np.array([[full[c] for full in ALL_ROLLS] for c in range(6)], dtype=np.intp)
# Indices of the six five-of-a-kind rolls and their face value (1..6).
_YAHTZEE_ROLLS = [(i, full.index(NUM_DICE) + 1)
                  for i, full in enumerate(ALL_ROLLS) if max(full) == NUM_DICE]

RAW = _RAW
CNT = _CNT
YAHTZEE_ROLLS = _YAHTZEE_ROLLS

