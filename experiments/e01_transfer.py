"""One clue per board from a spymaster on model X, read by a guesser on model Y.
SimSpymaster (margin 0) and SimGuesser, the 4,321 shared clue words, 500 boards.
One record per (spymaster model, guesser model, board)."""
import random

from codenames_lab import lab
from codenames_lab.players import SimGuesser, SimSpymaster

OUT = lab.results_name(__file__)
MODELS = ["w2v", "ppmi", "ppmi-cos"]

if __name__ == "__main__":
    lab.fresh(OUT)
    for spy_m in MODELS:
        spy = SimSpymaster(lab.models()[spy_m], clue_words=lab.clue_words())
        for guess_m in MODELS:
            for seed in range(lab.n(500)):
                rules = lab.Rules(lab.board(seed), lab.TEAMS)
                clue = spy.give_clue(rules.view("red", spymaster=True))
                owners = lab.one_turn(rules, clue, SimGuesser(lab.models()[guess_m], random.Random(seed)))
                lab.write(OUT, {"experiment": "transfer", "spy": spy_m, "guess": guess_m, "seed": seed,
                                "clue": str(clue.token), "count": clue.count, "owners": owners,
                                "found": lab.run_of_own(owners)})
