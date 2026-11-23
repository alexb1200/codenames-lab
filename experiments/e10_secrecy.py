"""Bayesian eavesdropping guesser and secrecy-aware spymaster. T 0.5, max risk 0.2,
200 seat-swapped games per row."""
from codenames_lab import lab

OUT = lab.results_name(__file__)


def team(model, listen=None, kappa=None):
    cfg = {"model": model, "spy": "risk", "T": 0.5, "max_risk": 0.2, "guesser": "soft"}
    if listen:
        cfg.update(guesser="bayes", read=listen, read_T=0.5)
    if kappa:
        cfg.update(spy="secret", kappa=kappa, secret_read=model)
    return cfg


if __name__ == "__main__":
    lab.fresh(OUT)
    rows = [("listen, same language", team("w2v", listen="w2v"), team("w2v")),
            ("plain vs listener", team("w2v"), team("w2v", listen="w2v")),
            ("baseline, different languages", team("w2v"), team("ppmi-cos")),
            ("listen in their language", team("w2v", listen="ppmi-cos"), team("ppmi-cos")),
            ("secret kappa 0.5 vs listener", team("w2v", kappa=0.5), team("w2v", listen="w2v")),
            ("secret kappa 1.0 vs listener", team("w2v", kappa=1.0), team("w2v", listen="w2v")),
            ("both listen, A secret kappa 0.5", team("w2v", listen="w2v", kappa=0.5), team("w2v", listen="w2v"))]
    for label, a, b in rows:
        lab.play_match(OUT, "secrecy", label, a, b, lab.n(200))
