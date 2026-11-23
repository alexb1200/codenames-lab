"""A crude eavesdropping guesser (EavesdropGuesser: penalty = weight x largest positive z
for any opponent clue so far, never discounted). T 0.5, max risk 0.2, 200 seat-swapped
games per row. Kept to document why the Bayesian version (e10) was needed."""
from codenames_lab import lab

OUT = lab.results_name(__file__)


def team(model, read=None, weight=0.0):
    cfg = {"model": model, "spy": "risk", "T": 0.5, "max_risk": 0.2, "guesser": "soft"}
    if read:
        cfg.update(guesser="eaves", read=read, weight=weight)
    return cfg


if __name__ == "__main__":
    lab.fresh(OUT)
    rows = [("same language, weight 0.5", team("w2v", "w2v", 0.5), team("w2v")),
            ("same language, weight 1", team("w2v", "w2v", 1.0), team("w2v")),
            ("same language, weight 2", team("w2v", "w2v", 2.0), team("w2v")),
            ("different languages, A plain", team("w2v"), team("ppmi-cos")),
            ("different languages, A reads in w2v", team("w2v", "w2v", 1.0), team("ppmi-cos")),
            ("different languages, A reads in ppmi-cos", team("w2v", "ppmi-cos", 1.0), team("ppmi-cos"))]
    for label, a, b in rows:
        lab.play_match(OUT, "eavesdrop_crude", label, a, b, lab.n(200))
