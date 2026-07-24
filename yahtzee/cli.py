"""Command-line interface: describe a game state, get the optimal play.

Examples
--------
Endgame, must score, opponent already finished on 205::

    python -m yahtzee --open Sixes,Yahtzee,Chance --upper 50 --score 178 \
        --dice 6,6,6,2,1 --rerolls 0 --opp-final 205

Same but with a reroll available, and a still-playing opponent::

    python -m yahtzee --open Sixes,Yahtzee,Chance --upper 50 --score 178 \
        --dice 6,6,6,2,1 --rerolls 1 \
        --opp-open Fours,Full-House,Yahtzee --opp-upper 40 --opp-current 150
"""

import argparse
import sys
from typing import List

from .solver import GameState, OpponentInfo, Solver


def _parse_list(s: str) -> List[str]:
    return [tok.strip().replace("-", " ") for tok in s.split(",") if tok.strip()]


def _parse_dice(s: str) -> List[int]:
    return [int(tok) for tok in s.replace(",", " ").split()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yahtzee",
        description="Optimal play for 2-roll Yahtzee (1 reroll) in a 2-player game.")
    you = p.add_argument_group("your situation")
    you.add_argument("--open", required=True,
                     help="your open categories, comma-separated (names or indices)")
    you.add_argument("--upper", type=int, default=0,
                     help="your upper-section total so far (default 0)")
    you.add_argument("--bonus-active", action="store_true",
                     help="set if your Yahtzee box is already filled with 50")
    you.add_argument("--score", type=int, required=True,
                     help="your current total score")
    you.add_argument("--dice", required=True,
                     help="the five dice on the table, e.g. 6,6,6,2,1")
    you.add_argument("--rerolls", type=int, choices=(0, 1), required=True,
                     help="rerolls left: 1 = may reroll once, 0 = must score now")

    opp = p.add_argument_group("opponent (finished OR still playing)")
    opp.add_argument("--opp-final", type=int, default=None,
                     help="opponent's final score (they have finished)")
    opp.add_argument("--opp-open", default=None,
                     help="opponent's open categories (they are still playing)")
    opp.add_argument("--opp-upper", type=int, default=0,
                     help="opponent's upper-section total so far")
    opp.add_argument("--opp-bonus-active", action="store_true",
                     help="opponent's Yahtzee box already filled with 50")
    opp.add_argument("--opp-current", type=int, default=0,
                     help="opponent's current total score (still playing)")

    p.add_argument("--objective", choices=("win", "ev"), default="win",
                   help="win = maximise win probability (default); "
                        "ev = maximise your own expected score")
    p.add_argument("--tie", type=float, default=0.5,
                   help="credit for an exact tie: 0.5 draw, 1 you win, 0 you lose")
    p.add_argument("--top", type=int, default=6, help="how many options to list")
    p.add_argument("--allow-slow", action="store_true",
                   help="force exact win-prob even with many categories open")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    gs = GameState(
        open_categories=_parse_list(args.open),
        upper_total=args.upper,
        bonus_active=args.bonus_active,
        current_score=args.score,
        dice=_parse_dice(args.dice),
        rerolls_left=args.rerolls,
    )

    if args.objective == "win":
        if args.opp_final is not None:
            opp = OpponentInfo(finished_score=args.opp_final)
        elif args.opp_open is not None:
            opp = OpponentInfo(open_categories=_parse_list(args.opp_open),
                               upper_total=args.opp_upper,
                               bonus_active=args.opp_bonus_active,
                               current_score=args.opp_current)
        else:
            print("error: win objective needs --opp-final or --opp-open",
                  file=sys.stderr)
            return 2
    else:
        opp = OpponentInfo(finished_score=0)  # unused for ev

    solver = Solver(tie_value=args.tie)
    try:
        decision = solver.best_play(gs, opp, objective=args.objective,
                                    allow_slow=args.allow_slow)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    print(decision.describe(top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
