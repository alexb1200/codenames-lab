"""Model gap against guesser noise. Each team uses one model for both roles: RiskSpymaster
(knows the true T, max risk 0.2) and SoftGuesser at T. 150 seat-swapped games per cell."""
from itertools import combinations

from codenames_lab import lab

OUT = lab.results_name(__file__)

if __name__ == "__main__":
    lab.fresh(OUT)
    for t in [0.05, 0.2, 0.5, 1.0]:
        for a, b in combinations(["w2v", "ppmi", "ppmi-cos"], 2):
            cfg = lambda m: {"model": m, "spy": "risk", "T": t, "max_risk": 0.2, "guesser": "soft"}
            lab.play_match(OUT, "model_gap", f"T {t}: {a} vs {b}", cfg(a), cfg(b), lab.n(150), T=t)
