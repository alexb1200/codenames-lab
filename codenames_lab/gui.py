"""Tkinter GUI. Run with: python -m codenames_lab.gui"""
import random
import tkinter as tk
from .game import ASSASSIN, NEUTRAL, Board, Clue, Game, Human, Pos, RandomPlayer, Renderer, Rules, Team
from .words import nltk_pool

COLORS = {"red": "#d04040", "blue": "#4070d0", "green": "#40a050", NEUTRAL: "#a08850", ASSASSIN: "#222222"}


class App(Renderer):
    def __init__(self, root: tk.Tk, delay_ms: int = 400) -> None:
        self.root = root
        self.delay_ms = delay_ms
        self.after_id: str | None = None
        self.error = ""
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.spymaster = tk.BooleanVar(value=False)
        self.board_frame = tk.Frame(root)
        self.board_frame.pack(padx=8, pady=8)
        self.cells: list[list[tk.Label]] = []
        self.status = tk.Label(root, font=("Helvetica", 14))
        self.status.pack()
        self.prompt = tk.Label(root)
        self.prompt.pack()
        bar = tk.Frame(root)
        bar.pack(pady=6)
        self.clue_word = tk.Entry(bar, width=14)
        self.clue_count = tk.Spinbox(bar, from_=0, to=9, width=3)
        self.give_btn = tk.Button(bar, text="Give clue", command=self.on_clue)
        self.pass_btn = tk.Button(bar, text="Pass", command=self.on_pass)
        for w in (self.clue_word, self.clue_count, self.give_btn, self.pass_btn):
            w.pack(side="left", padx=3)
        tk.Checkbutton(bar, text="Spymaster view", variable=self.spymaster,
                       command=self.redraw).pack(side="left", padx=8)

    def start(self, game: Game, board: Board) -> None:
        self.game = game
        game.reset(board)   # calls draw
        self.schedule()

    def schedule(self) -> None:
        """Run tick after the delay. Keep at most one tick pending."""
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(self.delay_ms, self.tick)

    def close(self) -> None:
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.root.destroy()

    # ----- turn loop: AI seats run on a timer, human seats wait for a click -----

    def tick(self) -> None:
        if not self.game.rules.done and not self.game.waiting_on_human():
            self.game.step()   # calls draw
            self.schedule()

    def human_moved(self) -> None:
        self.error = ""
        self.redraw()
        self.schedule()

    def can_act(self, phase: str) -> bool:
        rules = self.game.rules
        in_phase = (rules.clue is None) == (phase == "clue")
        return not rules.done and self.game.waiting_on_human() and in_phase

    # ----- human input -----

    def click_handler(self, pos: Pos):
        """A handler bound to one cell. A closure per cell avoids the classic late-binding bug."""
        return lambda event: self.on_click(pos)

    def on_click(self, pos: Pos) -> None:
        r, c = pos
        if self.can_act("guess") and not self.game.rules.board.grid[r][c].revealed:
            self.game.rules.guess(pos)
            self.human_moved()

    def on_pass(self) -> None:
        if self.can_act("guess"):
            try:
                self.game.rules.guess(None)
            except ValueError as e:
                self.error = str(e)
                self.redraw()
            else:
                self.human_moved()

    def on_clue(self) -> None:
        word = self.clue_word.get().strip()
        if self.can_act("clue"):
            try:
                self.game.rules.give_clue(Clue(word, int(self.clue_count.get())))
            except ValueError as e:
                self.error = str(e)
                self.redraw()
            else:
                self.clue_word.delete(0, "end")
                self.human_moved()

    # ----- drawing -----

    def build(self, rules: Rules) -> None:
        for r in range(rules.board.height):
            row = []
            for c in range(rules.board.width):
                cell = tk.Label(self.board_frame, width=12, height=3, relief="raised", bd=2)
                cell.grid(row=r, column=c, padx=2, pady=2)
                cell.bind("<Button-1>", self.click_handler((r, c)))
                row.append(cell)
            self.cells.append(row)

    def redraw(self) -> None:
        self.draw(self.game.rules)

    def draw(self, rules: Rules) -> None:
        if not self.cells:
            self.build(rules)
        for (r, c), cell in rules.board.cells():
            color = COLORS[cell.owner]
            if cell.revealed:
                self.cells[r][c].config(text=str(cell.token), bg=color, fg="white", relief="sunken")
            elif self.spymaster.get():
                self.cells[r][c].config(text=str(cell.token), bg="#eeeeee", fg=color, relief="raised")
            else:
                self.cells[r][c].config(text=str(cell.token), bg="#eeeeee", fg="black", relief="raised")
        self.status.config(text=self.status_text(rules))
        self.prompt.config(text=self.prompt_text(rules))
        for w in (self.clue_word, self.clue_count, self.give_btn):
            w.config(state="normal" if self.can_act("clue") else "disabled")
        self.pass_btn.config(state="normal" if self.can_act("guess") else "disabled")

    def status_text(self, rules: Rules) -> str:
        if rules.done:
            text = f"{rules.winner} wins ({rules.reason})"
        else:
            left = "  ".join(f"{t}: {rules.board.remaining(t)} left" for t in rules.alive)
            text = f"{rules.current} to play    clue: {rules.clue.token if rules.clue else '-'}"
            text += f" ({rules.guesses_left} guesses left)" if rules.clue else ""
            text += f"\n{left}"
        return text

    def prompt_text(self, rules: Rules) -> str:
        if rules.done:
            text = ""
        elif not self.game.waiting_on_human():
            text = f"{rules.current} is thinking..."
        elif self.error:
            text = f"Not allowed: {self.error}. Try again."
        elif rules.clue is None:
            text = f"{rules.current} spymaster: tick Spymaster view, type a clue and a count, press Give clue"
        else:
            text = f"{rules.current} guessers: untick Spymaster view, click a card or press Pass"
        return text


if __name__ == "__main__":
    rng = random.Random()
    root = tk.Tk()
    root.title("Codenames")
    names = ["red", "blue"]
    app = App(root)
    teams = [Team("red", Human(), [Human()]), Team("blue", RandomPlayer(rng), [RandomPlayer(rng)])]
    app.start(Game(teams, app), Board(5, 5, names, nltk_pool(), rng))
    root.mainloop()
