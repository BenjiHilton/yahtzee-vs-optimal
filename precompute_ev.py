"""Build and cache the exact expected-score table.

The expected-score value-to-go depends only on the scorecard state
``(open categories, upper total capped at 63, Yahtzee-bonus flag)`` -- a few
hundred thousand reachable states, one number each.  Once cached, every
expected-score decision, and the still-playing opponent model, becomes an
instant table lookup instead of a per-query solve.

We build it bottom-up by number of open categories (0 first, up to 13), so every
state's children are already solved when we reach it.  Progress is printed per
level and the (partial) table is checkpointed after each level, so an
interrupted run still leaves a usable, exact cache -- any state it contains is
correct, and anything missing is filled in lazily at query time.

    python precompute_ev.py                    # -> ev_table.pkl
    python precompute_ev.py my.pkl             # custom path
    python precompute_ev.py my.pkl 8           # only states with <= 8 open
"""

import sys
import time
from itertools import combinations

from yahtzee.engine import ExpectedScoreSolver
from yahtzee import scoring as S
from yahtzee.state import State

DEFAULT_PATH = "ev_table.pkl"

UPPER_CATS = list(S.UPPER_CATEGORIES)
UPPER_FACE_MAX = {c: 5 * (c + 1) for c in UPPER_CATS}  # most an upper box can hold


def reachable_upper_values(open_mask):
    """Superset of reachable upper totals for this open_mask: 0 .. (capped) sum
    of the maxima of the *filled* upper boxes."""
    filled_upper_max = sum(UPPER_FACE_MAX[c] for c in UPPER_CATS
                           if not (open_mask & (1 << c)))
    top = min(S.UPPER_BONUS_THRESHOLD, filled_upper_max)
    return range(top + 1)


def all_masks_with_open(k):
    """All bitmasks over 13 categories with exactly k bits set."""
    for combo in combinations(range(S.NUM_CATEGORIES), k):
        m = 0
        for c in combo:
            m |= 1 << c
        yield m


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    max_open = int(sys.argv[2]) if len(sys.argv) > 2 else S.NUM_CATEGORIES

    ev = ExpectedScoreSolver()
    ev.load_cache(path)  # resume if a partial table already exists
    print("Building exact expected-score table -> %s  (levels 0..%d)"
          % (path, max_open))
    start = time.time()
    total = len(ev._eadd)
    for k in range(0, max_open + 1):
        t0 = time.time()
        added = 0
        for mask in all_masks_with_open(k):
            yahtzee_open = bool(mask & (1 << S.YAHTZEE))
            # bonus_active can only be True once the Yahtzee box is filled.
            bonus_options = (False,) if yahtzee_open else (False, True)
            for upper in reachable_upper_values(mask):
                for bonus in bonus_options:
                    st = State(mask, upper, bonus)
                    if st not in ev._eadd:
                        ev.eadd(st)
                        added += 1
        total += added
        ev.save_cache(path)  # checkpoint this level
        print("  <=%2d open: +%-8d states  total=%-9d  %6.1fs  (elapsed %.0fs)"
              % (k, added, total, time.time() - t0, time.time() - start))
    from yahtzee.state import new_game_state
    if max_open >= S.NUM_CATEGORIES:
        print("\nOptimal expected final score from an empty card: %.3f"
              % ev.eadd(new_game_state()))
    print("Done: %d states cached to %s in %.0fs"
          % (len(ev._eadd), path, time.time() - start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
