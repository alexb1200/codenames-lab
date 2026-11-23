"""Verilog -> one Codenames board with ONE shared association graph.

Ported from the shared-graph compiler (v2c_shared.py). The logic is unchanged; the only
additions are the `signals` and `word_signal` records the renderer uses.

Every clue word ranks the same cards for both teams. The guesser takes the top-ranked
unrevealed cards; an honest spymaster gives the safe clue that scores the most cards
(ties at random).

The problem with a shared graph: a gate word ranks the other team's input cards on top,
so that team could give the same word and take its input card early, skipping that
card's own logic (a "leak").

The fix uses only the scoring rule:
  * Leaks score 1. In any word, cards of one team never sit next to each other above the
    target: a spacer card of the target's team separates consecutive inputs. So a team
    reaching into someone else's word stops after one card.
  * Real progress scores 3. Every signal value is three cards in a row (t, t', t'').
    A gate firing reveals all three (plus any spacers).
  * Something better than a leak is always on hand: each team owns a pool of filler
    pairs, each its own word [f, f'], scoring 2.
  So a max-score spymaster fires gates first (>= 3), clears filler pairs when it must
  wait (2), and never takes a leak (1) while fillers remain.

Logic is dual rail: signal s has a 0-triplet and a 1-triplet; NOT swaps them; even gate
depth is Red, odd is Blue, buffers fix parity.
  AND y = a.b :  [a1, s, b1, y1 y1' y1'']        (s = spacer of y's team)
                 [a0, y0 y0' y0''] and [b0, y0 y0' y0'']
Inputs: the key card makes the true triplet team-coloured and the false triplet
bystanders. Word [x x' x''] for each.
End: each Blue output gets a Red ack ([o1, ack3] and [o0, ack3]); a Blue done triplet
waits on every ack (with Blue spacers between them); a Red sweep word [done, every Red
card] lets Red clear the board only after done.
Every word ends with the assassin, so it can never reach further down.
"""
import json
import os
import subprocess
import sys
from typing import Any
from collections import Counter
from dataclasses import dataclass, field

R, B, BYS, ASS = 0, 1, 2, 3
COLOUR_NAME = {R: "red", B: "blue", BYS: "bystander", ASS: "assassin"}
DESIGNS = os.path.join(os.path.dirname(__file__), "designs")


def synth(verilog_path: str, top: str, params: dict[str, int] | None = None) -> dict:
    """Run Yosys (pip install yowasp-yosys) and return the AND/NOT netlist as parsed JSON.
    yowasp-yosys runs sandboxed, so its output goes in the working folder, then is removed."""
    out = f"_synth_{top}.json"
    # -defer keeps the module unelaborated, so chparam sets its parameters without renaming it
    sets = " ".join(f"-set {k} {v}" for k, v in (params or {}).items())
    chparams = f"chparam {sets} {top}; " if params else ""
    script = (f"read_verilog -defer {verilog_path}; {chparams}hierarchy -top {top}; "
              f"synth -flatten -top {top}; aigmap; opt_clean -purge; write_json {out}")
    subprocess.run(["yowasp-yosys", "-q", "-p", script], check=True)
    with open(out) as f:
        d = json.load(f)
    os.unlink(out)
    return d["modules"][top]


@dataclass
class Signal:
    """One dual-rail signal, for drawing. heads[v] is the card read as 'this signal is v'."""
    name: str
    kind: str              # "in", "const", "and", "buf", "ack" or "done"
    level: int
    team: int
    heads: tuple[int, int]
    fanin: list[int] = field(default_factory=list)   # indices into Circuit.signals


class Circuit:
    """Cards (colour, label) and clue words (rankings over card ids, assassin last)."""

    def __init__(self) -> None:
        self.colour: list[int] = []
        self.label: list[str] = []
        self.words: list[list[int]] = []
        self.word_label: list[str] = []
        self.word_signal: list[int | None] = []    # the signal a word's target belongs to
        self.signals: list[Signal] = []
        self.outputs: dict[tuple[str, int], tuple[int, int]] = {}
        self.fillers: dict[int, list[int]] = {}
        self.pass_word = -1
        self.n_and = 0
        self.depth = 0
        self.ass = self.card(ASS, "assassin")

    def card(self, c: int, label: str) -> int:
        self.colour.append(c)
        self.label.append(label)
        return len(self.colour) - 1

    def triple(self, c: int, label: str) -> list[int]:
        return [self.card(c, f"{label}#{k}") for k in range(3)]

    def word(self, cards: list[int], label: str, signal: int | None = None) -> None:
        self.words.append(list(cards) + [self.ass])
        self.word_label.append(label)
        self.word_signal.append(signal)

    def stats(self) -> dict:
        c = Counter(self.colour)
        return dict(cards=len(self.colour), red=c[R], blue=c[B], bystanders=c[BYS],
                    clue_words=len(self.words), longest_word=max(len(w) for w in self.words))


def gate_word(bd: Circuit, inputs: list[int], target: list[int], spacer_team: int, label: str,
              spacers: bool = True, signal: int | None = None) -> None:
    """[in1, s, in2, s, ..., target triplet]. Inputs are single cards of the other team;
    spacers of the target's team keep any leak to one card."""
    cards = []
    for j, c in enumerate(inputs):
        if j > 0 and spacers:
            cards.append(bd.card(spacer_team, f"spacer {label}"))
        cards.append(c)
    bd.word(cards + target, label, signal)


def compile_shared(mod: dict, inputs: dict[str, int], fillers_per_team: int = 4000,
                   spacers: bool = True) -> Circuit:
    """mod: a Yosys JSON module of AND and NOT cells. inputs: {port name: int value}."""
    bd = Circuit()
    ports = mod["ports"]
    drv: dict = {}   # bit -> ('in', name, i) | ('and', a, b) | ('not', a)
    for name, p in ports.items():
        if p["direction"] == "input":
            for i, bit in enumerate(p["bits"]):
                drv[bit] = ("in", name, i)
    for _, c in mod["cells"].items():
        cn = c["connections"]
        if c["type"] == "$_AND_":
            drv[cn["Y"][0]] = ("and", cn["A"][0], cn["B"][0])
        elif c["type"] == "$_NOT_":
            drv[cn["Y"][0]] = ("not", cn["A"][0])
        else:
            raise ValueError(f"unexpected cell {c['type']}; synthesise with aigmap")

    def lit(bit) -> tuple[Any, int]:
        """bit -> (base, inverted). base is an input, an AND output or a constant."""
        inv = 0
        while True:
            if bit in ("0", "1"):
                return (("const", bit), inv)
            d = drv[bit]
            if d[0] == "not":
                inv ^= 1
                bit = d[1]
            else:
                return (bit, inv)

    sys.setrecursionlimit(100000)
    level: dict = {}
    order: list = []

    def lev(base):
        if base in level:
            return level[base]
        if isinstance(base, tuple) or drv[base][0] == "in":
            level[base] = 0
        else:
            _, a, b = drv[base]
            level[base] = 1 + max(lev(lit(a)[0]), lev(lit(b)[0]))
        order.append(base)
        return level[base]

    out_bits = [(n, i, bit) for n, p in ports.items() if p["direction"] == "output"
                for i, bit in enumerate(p["bits"])]
    for _, _, bit in out_bits:
        lev(lit(bit)[0])

    rail: dict = {}   # key -> [triplet for 0, triplet for 1]
    team: dict = {}
    sig: dict = {}    # key -> index into bd.signals

    def name_of(base):
        if isinstance(base, tuple):
            return f"const{base[1]}"
        d = drv[base]
        return f"{d[1]}[{d[2]}]" if d[0] == "in" else f"n{base}"

    def add_signal(key, name, kind, lvl, fanin=()):
        sig[key] = len(bd.signals)
        bd.signals.append(Signal(name, kind, lvl, team[key], (rail[key][0][0], rail[key][1][0]),
                                 [sig[k] for k in fanin]))
        return sig[key]

    def head(key, inv, v):
        """First card of the triplet meaning literal == v. Downstream words use only this card."""
        return rail[key][v ^ inv][0]

    bufs: dict = {}

    def get(base, inv, want):
        key = (base, 0)
        if team[key] == want:
            return key, inv
        if base not in bufs:
            bk = (base, 1)
            rail[bk] = [bd.triple(want, f"buf({name_of(base)})={v}") for v in (0, 1)]
            team[bk] = want
            s = add_signal(bk, f"buf({name_of(base)})", "buf", level[base], [key])
            for v in (0, 1):
                gate_word(bd, [rail[key][v][0]], rail[bk][v], want, f"buf {name_of(base)}={v}", spacers, s)
            bufs[base] = bk
        return bufs[base], inv

    for base in order:
        tm = level[base] % 2
        key = (base, 0)
        if isinstance(base, tuple) or drv[base][0] == "in":
            v = int(base[1]) if isinstance(base, tuple) else (inputs[drv[base][1]] >> drv[base][2]) & 1
            rail[key] = [bd.triple(tm if vv == v else BYS, f"{name_of(base)}={vv}") for vv in (0, 1)]
            team[key] = tm
            s = add_signal(key, name_of(base), "const" if isinstance(base, tuple) else "in", 0)
            for vv in (0, 1):
                bd.word(rail[key][vv], f"open {name_of(base)}={vv}", s)
        else:
            _, a, b = drv[base]
            ka, ai = get(*lit(a), 1 - tm)
            kb, bi = get(*lit(b), 1 - tm)
            rail[key] = [bd.triple(tm, f"{name_of(base)}={v}") for v in (0, 1)]
            team[key] = tm
            s = add_signal(key, name_of(base), "and", level[base], [ka, kb])
            gate_word(bd, [head(ka, ai, 1), head(kb, bi, 1)], rail[key][1], tm,
                      f"and {name_of(base)}=1", spacers, s)
            gate_word(bd, [head(ka, ai, 0)], rail[key][0], tm, f"and {name_of(base)}=0 via a", spacers, s)
            gate_word(bd, [head(kb, bi, 0)], rail[key][0], tm, f"and {name_of(base)}=0 via b", spacers, s)

    acks = []
    top = max(level.values())
    for name, i, bit in out_bits:
        k, inv = get(*lit(bit), B)
        o0, o1 = head(k, inv, 0), head(k, inv, 1)
        bd.outputs[(name, i)] = (o0, o1)
        ack = bd.triple(R, f"ack {name}[{i}]")
        ack_sig = len(bd.signals)
        bd.signals.append(Signal(f"ack {name}[{i}]", "ack", top + 1, R, (ack[0], ack[0]), [sig[k]]))
        gate_word(bd, [o1], ack, R, f"ack {name}[{i}] from 1", spacers, ack_sig)
        gate_word(bd, [o0], ack, R, f"ack {name}[{i}] from 0", spacers, ack_sig)
        acks.append((ack[0], ack_sig))
    done = bd.triple(B, "done")
    done_sig = len(bd.signals)
    bd.signals.append(Signal("done", "done", top + 2, B, (done[0], done[0]), [s for _, s in acks]))
    gate_word(bd, [a for a, _ in acks], done, B, "done", spacers, done_sig)
    # filler pairs: each its own word, scoring 2
    for tm in (R, B):
        bd.fillers[tm] = []
        for j in range(fillers_per_team):
            pair = [bd.card(tm, f"filler {COLOUR_NAME[tm]} {j}#{k}") for k in range(2)]
            bd.word(pair, f"filler {COLOUR_NAME[tm]} {j}")
            bd.fillers[tm].append(len(bd.words) - 1)
    # the sweep is built last so it covers every Red card, fillers included
    reds = [c for c in range(len(bd.colour)) if bd.colour[c] == R]
    bd.word([done[0]] + reds, "sweep")
    # bystander words for a pass, if a team ever has no safe clue at all
    bys = [bd.card(BYS, f"pass {j}") for j in range(64)]
    bd.word(bys, "pass")
    bd.pass_word = len(bd.words) - 1
    bd.n_and = sum(1 for b in order if not isinstance(b, tuple) and drv[b][0] == "and")
    bd.depth = top
    return bd
