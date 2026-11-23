"""A spymaster that learns its guesser's temperature. Runs of boards with one guesser,
one clue per board. Conditions: correct (knows T), wrong (sure of 0.2 if T >= 0.5, else 1.0),
learner (LearningSpymaster, flat prior over 13 temperatures from 0.1 to 2.0, kept across
boards). w2v, max risk 0.2. Boards and guesser seeds are shared across conditions within a
run. One record per (true T, condition, run, board)."""
import random

import numpy as np

from codenames_lab import lab
from codenames_lab.game import ASSASSIN
from codenames_lab.players import LearningSpymaster, RiskSpymaster, SoftGuesser

OUT = lab.results_name(__file__)
MAX_RISK = 0.2

if __name__ == "__main__":
    lab.fresh(OUT)
    w2v = lab.models()["w2v"]
    for true_t in [0.2, 0.5, 1.0]:
        for cond in ["correct", "wrong", "learner"]:
            for run in range(lab.n(30)):
                rng = np.random.default_rng(run)
                spy: RiskSpymaster
                if cond == "learner":
                    spy = LearningSpymaster(w2v, MAX_RISK, clue_words=lab.clue_words(), rng=rng)
                else:
                    assumed = true_t if cond == "correct" else (0.2 if true_t >= 0.5 else 1.0)
                    spy = RiskSpymaster(w2v, assumed, MAX_RISK, clue_words=lab.clue_words(), rng=rng)
                for b in range(lab.n(10, 3)):
                    seed = 10_000 * run + b
                    rules = lab.Rules(lab.board(seed), lab.TEAMS)
                    if isinstance(spy, LearningSpymaster):
                        spy.seen = 0                       # new game, new history; posterior kept
                        belief = spy.mean_temperature()    # belief used for this clue
                    else:
                        belief = spy.assumed_temperature
                    clue = spy.give_clue(rules.view("red", spymaster=True))
                    owners = lab.one_turn(rules, clue, SoftGuesser(w2v, true_t, 0.0, random.Random(seed)))
                    if isinstance(spy, LearningSpymaster):
                        spy.observe(rules.view("red", spymaster=True))
                    lab.write(OUT, {"experiment": "learning", "true_T": true_t, "condition": cond, "run": run,
                                    "board": b, "seed": seed, "belief": belief, "clue": str(clue.token),
                                    "count": clue.count, "owners": owners, "found": lab.run_of_own(owners),
                                    "bad_hit": any(o not in ("red", "NEUTRAL") for o in owners),
                                    "assassin": ASSASSIN in owners})
