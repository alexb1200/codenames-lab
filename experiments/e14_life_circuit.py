"""Game of Life computed by one Codenames game (the designed board: hand-set rankings,
not real words). life.v, a 6x6 torus for 4 generations, compiled to one shared-graph
board and played to the end on the repo's engine by honest max-score players.

Rows: the glider; 20 random starting states; and three ablations on the glider, 3 player
seeds each: a spymaster that gives any safe clue instead of the best one, no spacers in
gate words, and 300 filler pairs per team instead of 4,000. One line per game."""
import random
import time

from codenames_lab import lab
from codenames_lab.circuit.compile import DESIGNS, compile_shared, synth
from codenames_lab.circuit.life import GLIDER, W, H, generation_turns, wrong_pixels
from codenames_lab.circuit.play import play, revealed_cards, summary

OUT = lab.results_name(__file__)


def run(mod, row: str, s0: int, seed: int, mode: str = "max", **comp) -> None:
    t0 = time.time()
    circuit = compile_shared(mod, {"s0": s0}, **comp)
    rules = play(circuit, seed=seed, mode=mode)
    st = summary(circuit, rules)
    lab.write(OUT, dict(experiment="life_circuit", row=row, s0=s0, seed=seed, mode=mode,
                        compile=comp, board=circuit.stats(), n_and=circuit.n_and, depth=circuit.depth,
                        **st, wrong_pixels=wrong_pixels(circuit, revealed_cards(rules), s0),
                        pixels=4 * W * H, generation_done=generation_turns(circuit, rules.history,
                                                                           rules.board.width),
                        seconds=round(time.time() - t0, 1)))


if __name__ == "__main__":
    lab.fresh(OUT)
    mod = synth(f"{DESIGNS}/life.v", "life")
    run(mod, "glider", GLIDER, seed=1)
    rng = random.Random(3)
    for trial in range(lab.n(20)):
        run(mod, "random state", rng.getrandbits(W * H), seed=trial)
    for seed in range(lab.n(3)):
        run(mod, "ablation: any safe clue", GLIDER, seed, mode="any")
        run(mod, "ablation: no spacers", GLIDER, seed, spacers=False)
        run(mod, "ablation: 300 filler pairs", GLIDER, seed, fillers_per_team=300)
