"""Full games between teams whose spymaster and guesser share one model (SimSpymaster
margin 0 + SimGuesser), every pair of models, 200 seat-swapped games each."""
from itertools import combinations

from codenames_lab import lab

OUT = lab.results_name(__file__)

if __name__ == "__main__":
    lab.fresh(OUT)
    for a, b in combinations(["w2v", "ppmi", "ppmi-cos"], 2):
        lab.play_match(OUT, "model_tournament", f"{a} vs {b}",
                       {"model": a, "spy": "sim", "guesser": "sim"},
                       {"model": b, "spy": "sim", "guesser": "sim"}, lab.n(200))
