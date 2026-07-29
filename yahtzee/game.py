"""Reusable game logic for human-vs-optimal-AI play (used by the web server).

The game state is a plain JSON-serialisable dict so it can travel to the browser
and back.  All the hard work (optimal moves) is delegated to the existing solver.
"""

import random
import threading
from typing import Dict, List, Optional

from . import scoring as S
from .dice import dice_to_counts
from .solver import GameState, OpponentInfo, Solver, category_index
from .state import State, categories_open, enumerate_placements

FULL_MASK = (1 << S.NUM_CATEGORIES) - 1

# One shared solver (warm caches, read-mostly).  A lock serialises the heavy
# win-probability computations so concurrent requests don't race the caches.
_solver: Optional[Solver] = None
_solver_lock = threading.Lock()


def get_solver() -> Solver:
    global _solver
    if _solver is None:
        # Exact win-probability in the last two turns (<=2 boxes open), using the
        # opponent's FULL score distribution -- essential for correctness: a
        # point-mass-at-the-mean opponent ignores their variance, so the AI would
        # report 100% and then lose when the opponent hit a lucky high roll (e.g.
        # a large straight). Full distribution keeps the win% honest and the play
        # variance-aware. Capping at 2 open keeps each move ~1-2s even on the free
        # host's 0.1 CPU (the smooth reward defeats pruning, so deeper is slow).
        # Everything earlier is expected-score play (near-identical that far out).
        _solver = Solver(max_exact_open=2, max_opp_dist_open=13)
    return _solver


def roll(n: int) -> List[int]:
    return sorted(random.randint(1, 6) for _ in range(n))


class Player:
    def __init__(self, name: str):
        self.name = name
        self.filled: Dict[int, int] = {}
        self.upper_total = 0
        self.bonus_active = False
        self.yz_bonus = 0
        self.upper_bonus_earned = False
        self.total = 0

    def state(self) -> State:
        mask = FULL_MASK
        for c in self.filled:
            mask &= ~(1 << c)
        return State(mask, self.upper_total, self.bonus_active)

    def open_cats(self):
        return categories_open(self.state())

    def done(self) -> bool:
        return len(self.filled) == S.NUM_CATEGORIES

    def apply(self, cat: int, counts) -> int:
        pts = S.score_placement(cat, counts, self.bonus_active)
        new_upper, upper_bonus = S.upper_delta(cat, counts, self.upper_total)
        yz = S.YAHTZEE_BONUS if (S.is_yahtzee(counts) and self.bonus_active) else 0
        self.filled[cat] = pts - yz
        self.yz_bonus += yz
        self.total += pts + upper_bonus
        self.upper_total = new_upper
        if upper_bonus:
            self.upper_bonus_earned = True
        if cat == S.YAHTZEE and S.is_yahtzee(counts):
            self.bonus_active = True
        return pts + upper_bonus

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "filled": {str(k): v for k, v in self.filled.items()},
            "upper_total": self.upper_total,
            "bonus_active": self.bonus_active,
            "yz_bonus": self.yz_bonus,
            "upper_bonus_earned": self.upper_bonus_earned,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Player":
        p = cls(d["name"])
        p.filled = {int(k): v for k, v in d["filled"].items()}
        p.upper_total = d["upper_total"]
        p.bonus_active = d["bonus_active"]
        p.yz_bonus = d["yz_bonus"]
        p.upper_bonus_earned = d["upper_bonus_earned"]
        p.total = d["total"]
        return p


# ----- optimal-move helpers -------------------------------------------------

def _opp_info(player: Player) -> OpponentInfo:
    if player.done():
        return OpponentInfo(finished_score=player.total)
    return OpponentInfo(
        open_categories=[S.CATEGORY_NAMES[c] for c in player.open_cats()],
        upper_total=player.upper_total, bonus_active=player.bonus_active,
        current_score=player.total)


def _advise(me: Player, opponent: Player, dice: List[int], rerolls_left: int):
    solver = get_solver()
    gs = GameState(open_categories=[S.CATEGORY_NAMES[c] for c in me.open_cats()],
                   upper_total=me.upper_total, bonus_active=me.bonus_active,
                   current_score=me.total, dice=dice, rerolls_left=rerolls_left)
    with _solver_lock:
        try:
            return solver.best_play(gs, _opp_info(opponent), objective="win"), "win"
        except ValueError:
            return solver.best_play(gs, _opp_info(opponent), objective="ev"), "ev"


def _previews(player: Player, dice: List[int]) -> Dict[str, int]:
    """Points the current dice would score in each legal box (to guide the human)."""
    counts = dice_to_counts(dice)
    out = {}
    for p in enumerate_placements(player.state(), counts):
        out[S.CATEGORY_NAMES[p.category]] = p.gain
    return out


def _ai_take_turn(ai: Player, human: Player) -> dict:
    dice = roll(5)
    dec, _ = _advise(ai, human, dice, 1)
    keep = list(dec.best_keep.keep)
    n = 5 - len(keep)
    final = sorted(keep + roll(n)) if n else sorted(dice)
    dec2, used2 = _advise(ai, human, final, 0)
    cat = dec2.best_score.category
    gain = ai.apply(cat, dice_to_counts(final))
    return {
        "rolled": dice, "kept": keep, "final": final,
        "category": S.CATEGORY_NAMES[cat], "points": gain,
        "win": round(100 * dec2.best_score.value, 1) if used2 == "win" else None,
    }


# ----- public API operating on the JSON game-state dict ---------------------

def _load(state: dict):
    return Player.from_dict(state["you"]), Player.from_dict(state["ai"])


def _store(state: dict, you: Player, ai: Player):
    state["you"] = you.to_dict()
    state["ai"] = ai.to_dict()


def _maybe_finish(state: dict, you: Player, ai: Player):
    if you.done() and ai.done():
        state["phase"] = "over"
        if you.total > ai.total:
            state["result"] = "you"
        elif ai.total > you.total:
            state["result"] = "ai"
        else:
            state["result"] = "tie"


def new_game(first: str = "you") -> dict:
    you, ai = Player("You"), Player("Optimal")
    state = {"you": you.to_dict(), "ai": ai.to_dict(), "first": first,
             "turn": "you", "phase": "await_roll", "dice": None,
             "rolls_left": 1, "previews": {}, "log": [], "result": None}
    if first == "ai":
        move = _ai_take_turn(ai, you)
        state["log"].append(move)
        _store(state, you, ai)
    return state


def api_roll(state: dict) -> dict:
    if state["phase"] != "await_roll":
        return state
    you, ai = _load(state)
    dice = roll(5)
    state["dice"] = dice
    state["rolls_left"] = 1
    state["phase"] = "rolled"
    state["previews"] = _previews(you, dice)
    return state


def api_reroll(state: dict, keep: List[int]) -> dict:
    if state["phase"] != "rolled" or state["rolls_left"] < 1:
        return state
    you, ai = _load(state)
    dice = state["dice"]
    # keep must be a sub-multiset of the current dice
    from collections import Counter
    kc, dc = Counter(keep), Counter(dice)
    keep = [f for f in keep if 1 <= f <= 6]
    if any(kc[f] > dc[f] for f in kc):
        return state
    n = 5 - len(keep)
    final = sorted(list(keep) + roll(n)) if n else sorted(dice)
    state["dice"] = final
    state["rolls_left"] = 0
    state["previews"] = _previews(you, final)
    return state


def api_score(state: dict, category) -> dict:
    if state["phase"] != "rolled" or state["dice"] is None:
        return state
    you, ai = _load(state)
    counts = dice_to_counts(state["dice"])
    cat = category_index(category)
    legal = set(S.legal_categories(you.state().open_mask, counts, you.bonus_active))
    if cat not in legal:
        state["error"] = "That box isn't a legal choice for these dice."
        return state
    gain = you.apply(cat, counts)
    state["log"].append({"human": True, "category": S.CATEGORY_NAMES[cat],
                         "points": gain, "final": list(state["dice"])})
    # AI replies with one turn (if it still has boxes to fill).
    if not ai.done():
        move = _ai_take_turn(ai, you)
        state["log"].append(move)
    _store(state, you, ai)
    _maybe_finish(state, you, ai)
    if state["phase"] != "over":
        state["turn"] = "you"
        state["phase"] = "await_roll"
        state["dice"] = None
        state["rolls_left"] = 1
        state["previews"] = {}
    state.pop("error", None)
    return state


def api_hint(state: dict) -> dict:
    """Recommended move for the human's current position."""
    if state["phase"] != "rolled" or state["dice"] is None:
        return {"hint": None}
    you, ai = _load(state)
    if state["rolls_left"] >= 1:
        dec, used = _advise(you, ai, state["dice"], 1)
        return {"hint": {"type": "keep", "keep": list(dec.best_keep.keep),
                         "objective": used}}
    dec, used = _advise(you, ai, state["dice"], 0)
    return {"hint": {"type": "score", "category": dec.best_score.name,
                     "points": dec.best_score.points, "objective": used}}
