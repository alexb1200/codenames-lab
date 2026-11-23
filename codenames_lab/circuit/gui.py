"""Watch one Codenames game compute Game of Life. Run with:

    python -m codenames_lab.circuit.gui              # glider
    python -m codenames_lab.circuit.gui --random 3   # random starting state from seed 3

Top: the Life frames as read off the board so far (generation 0 is the input), and a
timeline of how many pixels of each generation have been read, against game turns.
Left: every signal of the circuit by gate depth. Right: the whole board, one square per
card. Bottom: a log that says, for each turn, what the clue did in circuit terms.
Each frame is one Codenames turn: Step turn plays exactly one.
"""
import argparse
import random
import tkinter as tk

from ..game import Renderer, Rules
from .compile import ASS, BYS, DESIGNS, B, Circuit, R, compile_shared, synth
from .life import GLIDER, H, T, W, life_ref
from .play import CircuitBoard, CircuitGame, CircuitRules, teams

CELL = 14                 # Life pixel size
GATE_W, GATE_H = 640, 440
TOP = 34                  # room for the gate view's title
BOARD_ZOOM = 2
HIDDEN = {R: "#ecc4c4", B: "#c4d0ec", BYS: "#e4dcc8", ASS: "#000000"}
SHOWN = {R: "#c03030", B: "#3050c0", BYS: "#a08850", ASS: "#000000"}
JUST = "#ffcc00"          # cards revealed this turn
VALUE = {None: "#d0d0d0", 0: "#4a6a8a", 1: "#e08a00"}
FIRED = "#30a050"         # acks and done have one card: revealed = fired
PIX = {None: "#e6e6e6", 0: "#ffffff", 1: "#202020"}


def where(index: int) -> str:
    return f"({index % W},{index // W})"


class CircuitApp(Renderer):
    def __init__(self, root: tk.Tk, circuit: Circuit, s0: int, seed: int = 0) -> None:
        self.root, self.circuit, self.s0, self.seed = root, circuit, s0, seed
        self.ref = life_ref(s0, T)
        self.after_id: str | None = None
        self.playing = False
        self.index_cards()
        self.build_widgets()
        self.restart()

    # ---------- what each card and word means ----------

    def index_cards(self) -> None:
        c = self.circuit
        self.head_of: dict[int, tuple[int, int]] = {}         # card -> (signal, value)
        for si, s in enumerate(c.signals):
            for v in (0, 1):
                self.head_of.setdefault(s.heads[v], (si, v))
        self.pixel_of: dict[int, tuple[int, int, int]] = {}    # card -> (generation 0-based, index, value)
        for (name, k), (c0, c1) in c.outputs.items():
            if name == "hist":
                self.pixel_of[c0] = (k // (W * H), k % (W * H), 0)
                self.pixel_of[c1] = (k // (W * H), k % (W * H), 1)
        self.target = [set(w[-4:-1]) for w in c.words]
        # feeds[si] = (generation 0-based, pixel) of the first output whose logic uses signal si
        out_sig = {}
        for (name, k), (c0, _) in c.outputs.items():
            if name == "hist" and c0 in self.head_of:
                out_sig[self.head_of[c0][0]] = (k // (W * H), k % (W * H))
        self.feeds: dict[int, tuple[int, int]] = {}
        for si in sorted(out_sig, key=lambda k: out_sig[k]):   # earliest generation first
            stack = [si]
            while stack:
                x = stack.pop()
                if x in self.feeds:
                    continue
                self.feeds[x] = out_sig[si]
                stack.extend(c.signals[x].fanin)
        # deepest level of each generation's outputs, for the markers in the gate view
        self.gen_level: dict[int, int] = {}
        for si, (g, _) in out_sig.items():
            self.gen_level[g] = max(self.gen_level.get(g, 0), c.signals[si].level)
        self.is_signal = [not (lb.startswith(("spacer", "filler", "pass", "pad")) or lb == "assassin")
                          for lb in c.label]

    def nice(self, name: str) -> str:
        """Internal signal names in Life terms where they have one. Keeps a '=v' suffix."""
        base, eq, v = name.partition("=")
        if base != name:
            return self.nice(base) + eq + v
        if name.startswith("buf("):
            return f"copy of {self.nice(name[4:-1])}"
        if name.startswith("s0["):
            return f"input cell {where(int(name[3:-1]))}"
        if name.startswith("hist["):
            k = int(name[5:-1])
            return f"gen {k // (W * H) + 1} pixel {where(k % (W * H))}"
        return name

    def part_of(self, si: int | None) -> str:
        """Which output a signal works toward, e.g. 'gen 2 (3,1)'."""
        if si is None or si not in self.feeds:
            return "-"
        g, i = self.feeds[si]
        return f"gen {g + 1} {where(i)}"

    def card_value(self, card: int) -> str:
        """'n45=1' from a card label like 'n45=1#0'."""
        return self.nice(self.circuit.label[card].split("#")[0])

    def describe(self, wi: int) -> str:
        c = self.circuit
        label = c.word_label[wi]
        kind = label.split()[0]
        inputs = [self.card_value(x) for x in c.words[wi][:-4] if not c.label[x].startswith("spacer")]
        if kind in ("and", "buf"):
            out = self.card_value(c.words[wi][-4])
            px = self.pixel_of.get(c.words[wi][-4])
            part = self.part_of(c.word_signal[wi])
            if px:
                out += f", which is gen {px[0] + 1} pixel {where(px[1])}"
            if kind == "buf":
                return f"[{part}] COPY: {inputs[0]} copied to the other team's colour"
            return f"[{part}] AND: {out}, because {' and '.join(inputs)}"
        if kind == "open":
            return f"input {self.card_value(c.words[wi][0])}, set by the key card"
        if kind == "ack":
            px = self.pixel_of[c.words[wi][0]]
            return f"Red acknowledges output gen {px[0] + 1} pixel {where(px[1])}={px[2]}"
        if kind == "done":
            return "every output acknowledged: Blue's done card opens"
        if kind == "sweep":
            return "sweep: Red clears its remaining cards and wins"
        if kind == "filler":
            return "filler pair: nothing ready, so wait"
        return "pass: no safe clue"

    # ---------- widgets ----------

    def build_widgets(self) -> None:
        root = self.root
        top = tk.Frame(root)
        top.pack(fill="x", padx=6, pady=4)
        fw = 5 * (W * CELL + 16)
        self.frames = tk.Canvas(top, width=fw, height=H * CELL + 34, bg="white", highlightthickness=0)
        self.frames.pack(side="left")
        self.timeline = tk.Canvas(top, width=560, height=H * CELL + 34, bg="white", highlightthickness=0)
        self.timeline.pack(side="left", padx=8)

        mid = tk.Frame(root)
        mid.pack(fill="x", padx=6)
        self.gates = tk.Canvas(mid, width=GATE_W, height=GATE_H, bg="white", highlightthickness=0)
        self.gates.pack(side="left")
        bw = (int(len(self.circuit.colour) ** 0.5) + 1) * BOARD_ZOOM
        self.board_canvas = tk.Canvas(mid, width=bw, height=bw, bg="white", highlightthickness=0)
        self.board_canvas.pack(side="left", padx=8)
        self.hover = tk.Label(root, anchor="w", font=("Helvetica", 11))
        self.hover.pack(fill="x", padx=6)

        bar = tk.Frame(root)
        bar.pack(fill="x", padx=6, pady=4)
        self.play_btn = tk.Button(bar, text="Play", width=6, command=self.toggle)
        self.play_btn.pack(side="left")
        tk.Button(bar, text="Step turn", command=lambda: self.advance(1)).pack(side="left", padx=4)
        tk.Button(bar, text="Next gate", command=self.next_gate).pack(side="left", padx=4)
        tk.Button(bar, text="Restart", command=self.restart).pack(side="left", padx=4)
        tk.Label(bar, text="turns per frame while playing").pack(side="left", padx=(16, 2))
        self.speed = tk.Scale(bar, from_=1, to=500, orient="horizontal", length=200)
        self.speed.set(5)
        self.speed.pack(side="left")
        self.status = tk.Label(root, anchor="w", justify="left", font=("Helvetica", 12))
        self.status.pack(fill="x", padx=6)

        self.log = tk.Text(root, height=9, width=150, font=("Courier", 11))
        self.log.pack(fill="x", padx=6, pady=(0, 6))
        tk.Label(root, anchor="w", justify="left", font=("Helvetica", 10), text=(
            "Signals: grey = not decided yet, orange = 1, slate = 0, green = ack/done fired; "
            "the outlined one is what the last clue decided, lines go to its inputs.\n"
            "Board: every card in id order (circuit cards, then filler pairs, then pass bystanders); "
            "pale = hidden red/blue/bystander, strong = revealed, yellow = revealed this turn. "
            "Life pixels: grey = not read yet, a red box = reads wrong.")).pack(fill="x", padx=6)
        self.gates.bind("<Motion>", self.on_hover)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def layout_gates(self) -> None:
        c, g = self.circuit, self.gates
        g.delete("all")
        levels = max(s.level for s in c.signals) + 1
        colw = GATE_W / levels
        per: dict[int, list[int]] = {}
        for si, s in enumerate(c.signals):
            per.setdefault(s.level, []).append(si)
        for sis in per.values():   # within a depth, group by the generation and pixel fed
            sis.sort(key=lambda si: self.feeds.get(si, (T, 0)))
        self.gate_xy: dict[int, tuple[float, float]] = {}
        self.gate_item: dict[int, int] = {}
        self.columns: dict[int, list[int]] = per
        self.colw = colw
        for lvl, sis in per.items():
            step = (GATE_H - TOP) / len(sis)
            hgt = max(1.0, min(6.0, step - 0.5))
            for j, si in enumerate(sis):
                x, y = lvl * colw + 1, TOP + j * step
                self.gate_xy[si] = (x + colw / 2, y + hgt / 2)
                self.gate_item[si] = g.create_rectangle(x, y, x + colw - 2, y + hgt, fill=VALUE[None], width=0)
        g.create_text(4, 2, anchor="nw", font=("Helvetica", 10),
                      text=f"{len(c.signals)} signals by gate depth, left to right (0 = input cells, "
                           f"{levels - 1} = done). Hover for details.")
        for gen, lvl in sorted(self.gen_level.items()):
            x = (lvl + 1) * colw
            g.create_line(x, TOP - 4, x, GATE_H, fill="#888888", dash=(3, 3))
            g.create_text(x - 2, 19, anchor="ne", text=f"gen {gen + 1} out", font=("Helvetica", 9))

    def layout_board(self) -> None:
        bw = self.board.width * BOARD_ZOOM
        bh = self.board.height * BOARD_ZOOM
        self.image = tk.PhotoImage(width=bw, height=bh)
        rows = []
        n = len(self.circuit.colour)
        for r in range(self.board.height):
            line = []
            for col in range(self.board.width):
                i = r * self.board.width + col
                line += [HIDDEN[self.circuit.colour[i]] if i < n else "#ffffff"] * BOARD_ZOOM
            rows += ["{" + " ".join(line) + "}"] * BOARD_ZOOM
        self.image.put(" ".join(rows))
        self.board_canvas.delete("all")
        self.board_canvas.config(width=bw, height=bh)
        self.board_canvas.create_image(0, 0, anchor="nw", image=self.image)

    def layout_frames(self) -> None:
        f = self.frames
        f.delete("all")
        self.pix_item: dict[tuple[int, int], int] = {}
        for g in range(T + 1):
            x0 = g * (W * CELL + 16) + 4
            f.create_text(x0, 4, anchor="nw", text="gen 0 (input)" if g == 0 else f"gen {g}",
                          font=("Helvetica", 10))
            for i in range(W * H):
                x, y = x0 + (i % W) * CELL, 18 + (i // W) * CELL
                fill = PIX[(self.s0 >> i) & 1] if g == 0 else PIX[None]
                self.pix_item[(g, i)] = f.create_rectangle(x, y, x + CELL - 1, y + CELL - 1,
                                                           fill=fill, outline="#bbbbbb")

    # ---------- game ----------

    def restart(self) -> None:
        self.pause()
        self.board = CircuitBoard(self.circuit)
        self.game = CircuitGame(teams(self.circuit, self.board.width, self.seed), self)
        self.rules: CircuitRules = self.game.reset(self.board)   # type: ignore[assignment]
        self.seen = 0
        self.value: dict[int, int] = {}
        self.read: list[list[int | None]] = [[None] * (W * H) for _ in range(T)]
        self.done_at: dict[int, int] = {}
        self.curve: list[list[tuple[int, int]]] = [[(0, 0)] for _ in range(T)]
        self.just: list[int] = []
        self.current: int | None = None
        self.leaks = self.fillers = self.gate_turns = 0
        self.layout_gates()
        self.layout_board()
        self.layout_frames()
        self.log.delete("1.0", "end")
        self.refresh()

    def draw(self, rules: Rules) -> None:
        """Game calls this after every step; the work happens in refresh, once per frame."""

    def play_turns(self, k: int) -> None:
        """Play k whole turns (fewer if the game ends)."""
        target = self.rules.turn_number + k
        while not self.rules.done and self.rules.turn_number < target:
            self.game.step()

    def advance(self, k: int) -> None:
        self.play_turns(k)
        self.refresh()

    def next_gate(self) -> None:
        """Play turns until one decides an AND gate."""
        before = self.gate_turns
        while not self.rules.done and self.gate_turns == before:
            self.play_turns(1)
            self.consume()
        self.refresh()

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.playing = True
            self.play_btn.config(text="Pause")
            self.tick()

    def pause(self) -> None:
        self.playing = False
        if hasattr(self, "play_btn"):
            self.play_btn.config(text="Play")
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def tick(self) -> None:
        self.after_id = None
        if not self.playing:
            return
        self.advance(int(self.speed.get()))
        if self.rules.done:
            self.pause()
        else:
            self.after_id = self.root.after(30, self.tick)

    def close(self) -> None:
        self.pause()
        self.root.destroy()

    # ---------- turning history into pictures ----------

    def consume(self) -> list[str]:
        """Read new history events; update state; return log lines for finished turns."""
        c, hist, lines = self.circuit, self.rules.history, []
        events = hist[self.seen:]
        self.seen = len(hist)
        for e in events:
            if e.kind == "clue":
                for card in self.just:
                    self.put_card(card, SHOWN[c.colour[card]])
                self.just = []
                self.word = word_id_of(e)
                self.turn_line = f"turn {e.turn + 1:5d} {e.team:4s} clue for {e.count:<3d} {self.describe(self.word)}"
                self.turn_cards = 0
                label = c.word_label[self.word]
                self.fillers += label.startswith("filler")
                if label.startswith("and"):
                    self.gate_turns += 1
                sig = c.word_signal[self.word]
                if sig is not None:
                    self.current = sig
                lines.append(self.turn_line)
            elif e.kind == "guess" and e.pos is not None:
                card = e.pos[0] * self.board.width + e.pos[1]
                self.just.append(card)
                self.put_card(card, JUST)
                if self.is_signal[card] and card not in self.target[self.word] \
                        and not c.word_label[self.word].startswith("sweep"):
                    self.leaks += 1
                    lines.append(f"           LEAK: {c.label[card]} taken from a word that does not target it")
                if card in self.head_of:
                    si, v = self.head_of[card]
                    if si not in self.value:
                        self.value[si] = v
                        kind = c.signals[si].kind
                        self.gates.itemconfig(self.gate_item[si],
                                              fill=FIRED if kind in ("ack", "done") else VALUE[v])
                if card in self.pixel_of:
                    g, i, v = self.pixel_of[card]
                    self.read_pixel(g, i, v, e.turn + 1)   # turns count from 1
        return lines

    def read_pixel(self, g: int, i: int, v: int, turn: int) -> None:
        c0, c1 = self.circuit.outputs[("hist", g * W * H + i)]
        r0, r1 = self.board_revealed(c0), self.board_revealed(c1)
        val = 1 if r1 and not r0 else 0 if r0 and not r1 else None
        before = self.read[g][i]
        self.read[g][i] = val
        wrong = val is not None and val != (self.ref[g] >> i) & 1
        self.frames.itemconfig(self.pix_item[(g + 1, i)], fill=PIX[val],
                               outline="#ff0000" if wrong else "#bbbbbb", width=2 if wrong else 1)
        if before is None and val is not None:
            n = sum(x is not None for x in self.read[g])
            self.curve[g].append((turn, n))
            if n == W * H:
                self.done_at[g] = turn

    def board_revealed(self, card: int) -> bool:
        r, col = divmod(card, self.board.width)
        return self.board.grid[r][col].revealed

    def put_card(self, card: int, colour: str) -> None:
        r, col = divmod(card, self.board.width)
        x, y = col * BOARD_ZOOM, r * BOARD_ZOOM
        self.image.put(colour, to=(x, y, x + BOARD_ZOOM, y + BOARD_ZOOM))

    def refresh(self) -> None:
        lines = self.consume()
        if lines:
            self.log.insert("end", "\n".join(lines[-400:]) + "\n")
            self.log.delete("1.0", "end-400l")
            self.log.see("end")
        self.draw_current()
        self.draw_timeline()
        self.status.config(text=self.status_text())

    def draw_current(self) -> None:
        g = self.gates
        g.delete("current")
        if self.current is None:
            return
        s = self.circuit.signals[self.current]
        x, y = self.gate_xy[self.current]
        for fi in s.fanin:
            fx, fy = self.gate_xy[fi]
            g.create_line(fx, fy, x, y, fill="#cc0000", tags="current")
        g.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#cc0000", width=2, tags="current")

    def draw_timeline(self) -> None:
        t = self.timeline
        t.delete("all")
        w, h = int(t["width"]), int(t["height"])
        span = max(8000, self.rules.turn_number + 1)
        now_turn = self.rules.turn_number + 1
        x = lambda turn: 36 + (w - 44) * turn / span           # noqa: E731
        y = lambda n: h - 16 - (h - 40) * n / (W * H)          # noqa: E731
        t.create_text(4, 2, anchor="nw", font=("Helvetica", 10),
                      text="pixels read per Life generation, against Codenames turns")
        t.create_line(36, y(0), w - 8, y(0), fill="#999999")
        t.create_text(32, y(W * H), anchor="e", text="36", font=("Helvetica", 9))
        t.create_text(32, y(0), anchor="e", text="0", font=("Helvetica", 9))
        t.create_text(w - 8, h - 2, anchor="se", text=f"turn {span}", font=("Helvetica", 9))
        colours = ["#d04040", "#e08a00", "#40a050", "#4070d0"]
        for g in range(T):
            pts = self.curve[g] + [(now_turn, self.curve[g][-1][1])]
            coords = [v for turn, n in pts for v in (x(turn), y(n))]
            t.create_line(*coords, fill=colours[g], width=2)
            t.create_text(x(pts[-1][0]) + 2, y(pts[-1][1]), anchor="w", fill=colours[g],
                          text=f"gen {g + 1}", font=("Helvetica", 9))
        now = x(now_turn)
        t.create_line(now, 18, now, y(0), fill="#888888", dash=(2, 2))

    def status_text(self) -> str:
        r = self.rules
        decided = sum(1 for si in self.value if self.circuit.signals[si].kind == "and")
        head = f"{r.winner} wins ({r.reason})" if r.done else f"turn {r.turn_number + 1}, {r.current} to play"
        gens = "  ".join(f"gen {g + 1} done at turn {self.done_at[g]}" for g in sorted(self.done_at))
        return (f"{head}   red {r.board.remaining('red')} / blue {r.board.remaining('blue')} cards left   "
                f"AND gates decided {decided}/{self.circuit.n_and}   filler turns {self.fillers}   "
                f"leaks {self.leaks}\n{gens}")

    def on_hover(self, event) -> None:
        lvl = int(event.x // self.colw)
        sis = self.columns.get(lvl)
        if not sis:
            return
        step = (GATE_H - TOP) / len(sis)
        j = int((event.y - TOP) // step)
        if not 0 <= j < len(sis):
            return
        si = sis[j]
        s = self.circuit.signals[si]
        val = self.value.get(si)
        ins = ", ".join(f"{self.nice(self.circuit.signals[f].name)}={self.value.get(f, '?')}" for f in s.fanin)
        outs = [f"gen {p[0] + 1} pixel {where(p[1])}" for h in s.heads if (p := self.pixel_of.get(h))]
        note = ""
        if val is None and self.rules.done:
            note = "   (never decided: no output needed it before the game ended)"
        self.hover.config(text=f"{s.kind} {self.nice(s.name)} at depth {s.level}, a "
                               f"{'red' if s.team == R else 'blue'} signal, part of {self.part_of(si)}: "
                               f"value {'?' if val is None else val}   inputs: {ins or '-'}"
                               + (f"   it is {outs[0]}" if outs else "") + note)


def word_id_of(event) -> int:
    return int(str(event.token)[1:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", type=int, default=None, help="random 6x6 starting state from this seed")
    ap.add_argument("--seed", type=int, default=1, help="seed for the players' tie-breaks")
    args = ap.parse_args()
    s0 = GLIDER if args.random is None else random.Random(args.random).getrandbits(W * H)
    print("synthesising life.v with Yosys...")
    circuit = compile_shared(synth(f"{DESIGNS}/life.v", "life"), {"s0": s0})
    root = tk.Tk()
    root.title("Game of Life inside one game of Codenames")
    CircuitApp(root, circuit, s0, args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()
