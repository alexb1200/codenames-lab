"""Compile any combinational Verilog module into one Codenames board and play it.

  python -m codenames_lab.circuit.cli codenames_lab/circuit/designs/adder.v adder4 a=9 b=7
  python -m codenames_lab.circuit.cli codenames_lab/circuit/designs/adder.v adder4 a=9 b=7 --export board.json

Prints the board size, how the game went, and each output port as read off the
revealed cards. Unset inputs are 0.
"""
import json
import sys

from .compile import COLOUR_NAME, compile_shared, synth
from .play import play, read_port, revealed_cards, summary


def main(argv: list[str]) -> None:
    path, top = argv[0], argv[1]
    inputs: dict[str, int] = {}
    export = None
    rest = argv[2:]
    for i, a in enumerate(rest):
        if a == "--export":
            export = rest[i + 1]
        elif "=" in a:
            k, v = a.split("=", 1)
            inputs[k] = int(v, 0)
    mod = synth(path, top)
    for name, p in mod["ports"].items():
        if p["direction"] == "input":
            inputs.setdefault(name, 0)
    circuit = compile_shared(mod, inputs)
    rules = play(circuit, seed=0)
    revealed = revealed_cards(rules)
    print("board:", circuit.stats())
    print("game :", summary(circuit, rules))
    for name, p in mod["ports"].items():
        if p["direction"] == "output":
            value, bad = read_port(circuit, revealed, name)
            print(f"{name} = {value}" + (f"  ({bad} unreadable bits)" if bad else ""))
    if export:
        with open(export, "w") as f:
            json.dump({"cards": [{"id": i, "colour": COLOUR_NAME[c], "label": lb}
                                 for i, (c, lb) in enumerate(zip(circuit.colour, circuit.label))],
                       "clue_words": [{"id": i, "label": lb, "ranking": r}
                                      for i, (r, lb) in enumerate(zip(circuit.words, circuit.word_label))],
                       "outputs": {f"{n}[{i}]": {"card_for_0": c[0], "card_for_1": c[1]}
                                   for (n, i), c in circuit.outputs.items()}}, f)
        print("exported", export)


if __name__ == "__main__":
    main(sys.argv[1:])
