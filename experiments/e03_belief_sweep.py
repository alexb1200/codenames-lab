"""RiskSpymaster with an assumed guesser temperature, SoftGuesser with a true temperature.
One clue per board, w2v for both, 100 boards per cell. One record per clue."""
import random
from itertools import product

import numpy as np

from codenames_lab import lab
from codenames_lab.players import RiskSpymaster, SoftGuesser

OUT = lab.results_name(__file__)

if __name__ == "__main__":
    lab.fresh(OUT)
    w2v = lab.models()["w2v"]
    for true_t, assumed_t, max_risk in product([0.2, 0.5, 1.0], [0.2, 0.5, 1.0], [0.05, 0.2]):
        for seed in range(lab.n(100)):
            rules = lab.Rules(lab.board(seed), lab.TEAMS)
            spy = RiskSpymaster(w2v, assumed_t, max_risk, clue_words=lab.clue_words(), rng=np.random.default_rng(seed))
            clue = spy.give_clue(rules.view("red", spymaster=True))
            owners = lab.one_turn(rules, clue, SoftGuesser(w2v, true_t, 0.0, random.Random(seed)))
            lab.write(OUT, {"experiment": "belief_sweep", "true_T": true_t, "assumed_T": assumed_t,
                            "max_risk": max_risk, "seed": seed, "clue": str(clue.token), "count": clue.count,
                            "owners": owners, "found": lab.run_of_own(owners), "unsafe": spy.unsafe_clues})
