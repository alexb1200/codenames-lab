"""Risk appetite: RiskSpymaster teams that differ only in max risk. w2v, guessers at T 0.5,
spymasters know it. 100 seat-swapped games per pair."""
from codenames_lab import lab

OUT = lab.results_name(__file__)


def team(max_risk):
    return {"model": "w2v", "spy": "risk", "T": 0.5, "max_risk": max_risk, "guesser": "soft"}


if __name__ == "__main__":
    lab.fresh(OUT)
    for a, b in [(0.05, 0.2), (0.05, 0.4), (0.2, 0.4)]:
        lab.play_match(OUT, "risk_appetite", f"{a} vs {b}", team(a), team(b), lab.n(100))
