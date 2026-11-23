"""Build every results table from the raw JSONL files in results/.

    python analyze.py            # writes results/tables.md and prints it

Numbers are means over the recorded units (games, clues or boards). "± x" is one standard
error: sqrt(p(1-p)/n) for a rate, s/sqrt(n) for a mean. Matches are seat-swapped (team A
moves first on even seeds), so each match also shows A's win rate by seat.
"""
import json
import os
from collections import Counter, defaultdict
from math import comb, sqrt

import numpy as np

RESULTS = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))


def load(name: str) -> list[dict]:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def rate(xs) -> str:
    xs = [bool(x) for x in xs]
    n = len(xs)
    p = sum(xs) / n
    return f"{p:.3f} ± {sqrt(p * (1 - p) / n):.3f}"


def mean(xs, digits: int = 2) -> str:
    a = np.asarray(xs, dtype=float)
    se = a.std(ddof=1) / sqrt(len(a)) if len(a) > 1 else 0.0
    return f"{a.mean():.{digits}f} ± {se:.{digits}f}"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def by(records, *keys):
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in keys)].append(r)
    return groups


# ---------- matches ----------

def match_rows(records: list[dict]) -> list[list]:
    rows = []
    for (row,), games in by(records, "row").items():
        a_won = [g["winner"] == g["a_color"] for g in games]
        first = [g["winner"] == g["a_color"] for g in games if g["a_color"] == "red"]
        second = [g["winner"] == g["a_color"] for g in games if g["a_color"] == "blue"]
        a_clues = [c for g in games for c in g["clues"][g["a_color"]]]
        b_clues = [c for g in games for c in g["clues"]["blue" if g["a_color"] == "red" else "red"]]
        assassin = [g["reason"] == "assassin" for g in games]
        rows.append([row, len(games), rate(a_won), f"{np.mean(first):.3f}", f"{np.mean(second):.3f}",
                     f"{np.mean(a_clues):.2f}", f"{np.mean(b_clues):.2f}", f"{np.mean(assassin):.3f}",
                     f"{np.mean([g['turns'] for g in games]):.1f}"])
    return rows


MATCH_HEADER = ["Match (A vs B)", "Games", "A wins", "A wins moving first", "A wins moving second",
                "A cards per clue", "B cards per clue", "Games ending on assassin", "Mean turns"]


def guess_mix_rows(records: list[dict]) -> list[list]:
    rows = []
    for (row,), games in by(records, "row").items():
        for side in ("A", "B"):
            c: Counter[str] = Counter()
            for g in games:
                color = g["a_color"] if side == "A" else ("blue" if g["a_color"] == "red" else "red")
                c.update(g["guesses"][color])
            n = sum(c.values())
            rows.append([row, side, n] + [f"{c[k] / n:.3f}" for k in ("own", "opp", "neutral", "assassin")])
    return rows


GUESS_HEADER = ["Match", "Side", "Guesses", "Own", "Other team's", "Bystander", "Assassin"]


# ---------- sections ----------

def transfer() -> str:
    recs = load("e01_transfer.jsonl")
    if not recs:
        return ""
    rows = []
    for (spy, guess), rs in by(recs, "spy", "guess").items():
        rows.append([spy, guess, len(rs), f"{np.mean([r['count'] for r in rs]):.2f}",
                     rate([bool(r["owners"]) and r["owners"][0] == "red" for r in rs]),
                     mean([r["found"] for r in rs]), rate([r["found"] >= r["count"] for r in rs]),
                     rate(["ASSASSIN" in r["owners"] for r in rs])])
    return "## e01 Transfer: one clue, spymaster model X, guesser model Y\n\nA random first guess is right 9/25 = 0.36.\n\n" + \
        table(["Spymaster", "Guesser", "Boards", "Cards asked", "First guess right", "Cards found",
               "Clue fully read", "Hit assassin"], rows)


def matches(name: str, title: str, note: str = "", mix: bool = False) -> str:
    recs = load(name)
    if not recs:
        return ""
    out = f"## {title}\n\n{note}\n\n" + table(MATCH_HEADER, match_rows(recs))
    if mix:
        out += "\n\nWhere each side's guesses landed:\n\n" + table(GUESS_HEADER, guess_mix_rows(recs))
    return out


def belief_sweep() -> str:
    recs = load("e03_belief_sweep.jsonl")
    if not recs:
        return ""
    out = ["## e03 Belief sweep: spymaster's assumed temperature vs the guesser's true one\n\n"
           "w2v, one clue per board. Each cell: cards found / rate of hitting an opponent card or the assassin."]
    for (risk,), rs in sorted(by(recs, "max_risk").items()):
        cells = by(rs, "true_T", "assumed_T")
        temps = sorted({r["assumed_T"] for r in rs})
        rows = []
        for true_t in sorted({r["true_T"] for r in rs}):
            row = [true_t]
            for a in temps:
                c = cells[(true_t, a)]
                row.append(f"{np.mean([r['found'] for r in c]):.2f} / "
                           f"{np.mean([any(o not in ('red', 'NEUTRAL') for o in r['owners']) for r in c]):.3f}"
                           f" (asked {np.mean([r['count'] for r in c]):.2f}, unsafe {sum(r['unsafe'] for r in c)})")
            rows.append(row)
        out.append(f"### Max risk {risk}, {len(cells[(rs[0]['true_T'], temps[0])])} boards per cell\n\n" +
                   table(["True T \\ assumed T"] + [str(a) for a in temps], rows))
    return "\n\n".join(out)


def model_gap() -> str:
    recs = load("e05_model_gap.jsonl")
    if not recs:
        return ""
    out = matches("e05_model_gap.jsonl", "e05 Model gap against guesser noise",
                  "Each team: one model for both roles, RiskSpymaster that knows T, SoftGuesser at T, max risk 0.2.")
    rows = [[t, len(gs), rate([g["winner"] == "red" for g in gs])] for (t,), gs in sorted(by(recs, "T").items())]
    return out + "\n\nFirst mover (red, 9 cards against 8) wins, pooled over all three pairs:\n\n" + \
        table(["Guesser T", "Games", "First mover wins"], rows)


def assassin() -> str:
    recs = load("e06_assassin.jsonl")
    if not recs:
        return ""
    k1 = [r for r in recs if r["count"] == 1]
    k9 = [r for r in recs if r["count"] == 9]
    n_samples = 4000
    worst = lambda pairs: max(abs(a - b) for a, b in pairs)
    lines = ["## e06 Assassin risk under a soft guesser", "",
             f"{len(recs)} rows ({len({(r['seed'], r['clue']) for r in recs})} board-clue pairs). "
             f"Each rate is from {n_samples} sampled orders, so its sampling SE is at most {0.5 / sqrt(n_samples):.4f}. "
             f"Each row below is the largest gap over {len(k1)} comparisons, where about 3 to 4 SE is expected by chance; "
             "tests/test_players.py checks both forms with 200,000 samples.", "",
             table(["Exact form", "Largest |measured − formula|"], [
                 ["P(assassin is first non-own pick) = s", f"{worst((r['assassin_first_miss'], r['s']) for r in k1):.4f}"],
                 ["P(hit) at count 1 = w_A / Σ w", f"{worst((r['hit'], r['first']) for r in k1):.4f}"],
                 ["P(hit) at count 9 = s", f"{worst((r['hit'], r['s']) for r in k9):.4f}"]])]
    temps = sorted({r["T"] for r in recs})
    counts = sorted({r["count"] for r in recs})
    rows = []
    for t in temps:
        rows.append([t] + [f"{np.mean([r['hit'] for r in recs if r['T'] == t and r['count'] == k]) / np.mean([r['s'] for r in recs if r['T'] == t and r['count'] == k]):.3f}" for k in counts])
    lines += ["", "Count cap: mean hit rate / mean s.", "", table(["T"] + [f"count {k}" for k in counts], rows)]
    rows = []
    for t in temps:
        row = [t]
        for above in [0, 1, 2, 3, 5, 8]:
            c = [r["hit"] for r in recs if r["T"] == t and r["count"] == 3 and r["above"] == above]
            row.append(f"{np.mean(c):.3f} (n={len(c)})" if len(c) >= 30 else "-")
        rows.append(row)
    lines += ["", "Shielding: hit rate at count 3 by how many non-own cards rank above the assassin.", "",
              table(["T", "0", "1", "2", "3", "5", "8"], rows)]
    rows = []
    for t in temps:
        c = [r for r in recs if r["T"] == t and r["count"] == 3 and r["hit"] > 0]
        lh = np.log([r["hit"] for r in c])
        rows.append([t, len(c), f"{np.corrcoef(lh, np.log([r['s'] for r in c]))[0, 1]:.3f}",
                     f"{np.corrcoef(lh, [r['gap_bad'] for r in c])[0, 1]:.3f}",
                     f"{np.corrcoef(lh, [-r['gap_own'] for r in c])[0, 1]:.3f}"])
    lines += ["", "Correlation of log hit rate (count 3, hit > 0) with:", "",
              table(["T", "Rows", "log s", "assassin minus closest other non-own (/T)",
                     "assassin minus k-th own card (/T)"], rows)]
    return "\n".join(lines)


def learning() -> str:
    recs = load("e07_learning.jsonl")
    if not recs:
        return ""
    boards = max(r["board"] for r in recs) + 1
    early = lambda r: r["board"] < 2
    late = lambda r: r["board"] >= boards - 3
    rows = []
    for (t, cond), rs in by(recs, "true_T", "condition").items():
        beliefs = [np.mean([r["belief"] for r in rs if r["board"] == b]) for b in range(boards)]
        rows.append([t, cond, len({r["run"] for r in rs}),
                     mean([r["found"] for r in rs if early(r)]), mean([r["found"] for r in rs if late(r)]),
                     rate([r["bad_hit"] for r in rs if early(r)]), rate([r["bad_hit"] for r in rs if late(r)]),
                     rate([r["assassin"] for r in rs]),
                     " ".join(f"{b:.2f}" for b in beliefs) if cond == "learner" else f"{beliefs[0]:.2f}"])
    return "## e07 Learning the guesser's temperature\n\nOne clue per board, boards 1–2 vs the last 3. " \
           "Belief = the temperature the spymaster planned with on each board (learner: posterior mean).\n\n" + \
        table(["True T", "Spymaster", "Runs", "Cards found, boards 1–2", "Cards found, last 3",
               "Bad hit, boards 1–2", "Bad hit, last 3", "Assassin (all)", "Belief by board"], rows)


def vocab() -> str:
    recs = load("e09_vocab.jsonl")
    if not recs:
        return ""
    shared = [r for r in recs if r["experiment"] == "vocab_shared"]
    rows = []
    for m in sorted({r["M"] for r in shared}):
        tail = [1.0] + [1 - (1 - comb(9, k) / comb(25, k)) ** m for k in range(1, 10)]
        rows.append(["null", m, f"{sum(tail[1:]):.2f}", "exact", ""])
    for (model, m), rs in sorted(by(shared, "model", "M").items()):
        per_board = [np.mean(r["k_star"]) for r in rs]
        values = Counter(v for r in rs for v in r["k_star"])
        n = sum(values.values())
        top2 = values.most_common(2)
        rows.append([model + (" (all words)" if rs[0].get("full") else ""), m, mean(per_board), len(rs),
                     f"{sorted(v for v, _ in top2)}: {sum(c for _, c in top2) / n:.3f}"])
    noisy = by([r for r in recs if r["experiment"] == "vocab_noisy"], "M")
    rows2 = [[m, len(rs), f"{np.mean([r['count'] for r in rs]):.2f}", mean([r["expected_found"] for r in rs]),
              f"{np.mean([r['risk'] for r in rs]):.3f}"] for (m,), rs in sorted(noisy.items())]
    return "## e09 Vocabulary scaling\n\nShared model: best clue size K* from M random legal clue words " \
           "(± is over boards). Null: uniformly random rankings, exact.\n\n" + \
        table(["Model", "M", "E[K*]", "Boards", "Two most likely K* and their mass"], rows) + \
        "\n\nNoisy guesser (T 0.5), RiskSpymaster's first clue, w2v:\n\n" + \
        table(["M", "Boards", "Cards asked", "Expected cards found", "Estimated risk"], rows2)


def solvability() -> str:
    recs = load("e11_solvability.jsonl")
    if not recs:
        return ""
    out = ["## e11 Solvability of a real word graph (w2v, global cosine threshold)",
           "Solvable = every card of mine has a clean clue. Columns are P(solvable) over boards; predictions are means of per-board predictions."]
    for (n,), rs in sorted(by(recs, "N").items()):
        rows = []
        for (p,), c in sorted(by(rs, "p").items()):
            du = [d for r in c for d in r["deg_unreachable"]]
            dr = [d for r in c for d in r["deg_reachable"]]
            rows.append([p, f"{np.mean([r['real'] for r in c]):.2f}", f"{np.mean([r['clue_shuffle'] for r in c]):.2f}",
                         f"{np.mean([r['two_sided'] for r in c]):.2f}", f"{np.mean([r['per_clue_threshold'] for r in c]):.2f}",
                         f"{np.mean([r['pred_fixed_degree'] for r in c]):.2f}", f"{np.mean([r['pred_per_clue'] for r in c]):.2f}",
                         f"{np.mean([r['pred_equal'] for r in c]):.2f}", f"{np.mean([r['unreachable'] for r in c]):.2f}",
                         f"{np.mean(du):.1f} vs {np.mean(dr):.1f}" if du else f"- vs {np.mean(dr):.1f}"])
        out.append(f"### N = {n}, t = {rs[0]['t']}, {len(by(rs, 'p')[(rs[0]['p'],)])} boards per row\n\n" + table(
            ["p", "Real", "Clue-degree shuffle", "Two-sided (both degrees)", "Per-clue threshold",
             "Fixed-degree prediction", "Per-clue binomial prediction", "Equal-density prediction",
             "Unreachable cards (real)", "Card degree: unreachable vs reachable"], rows))
    return "\n\n".join(out)


def ranked_scale() -> str:
    recs = load("e12_ranked_scale.jsonl")
    if not recs:
        return ""
    rows = []
    for (n,), rs in sorted(by(recs, "N").items()):
        t, m = rs[0]["t"], rs[0]["M"]
        null_k = sum(1 - (1 - comb(t, k) / comb(n, k)) ** m for k in range(1, t + 1))
        rows.append([n, t, len(rs), mean([r["k_star"] for r in rs]), f"{null_k:.2f}",
                     mean([r["covered_share"] for r in rs], 3), f"{1 - (1 - 1 / n) ** m:.3f}",
                     rate([r["all_covered"] for r in rs])])
    return "## e12 Ranked clues against board size (w2v, M = 4,000)\n\n" + table(
        ["N", "t", "Boards", "E[K*] real", "E[K*] null (exact)", "My cards that are some clue's top card",
         "Same, null", "Boards where all are"], rows)


def hub() -> str:
    recs = load("e13_hub.jsonl")
    if not recs:
        return ""
    per_board = [r for r in recs if r["experiment"] == "hub"]
    rows = []
    for (model,), rs in by(per_board, "model").items():
        conc = [r["top_card_share"] for r in rs if "top_card_share" in r]
        rows.append([model, len(rs), mean(conc, 3), mean([r["k_star"] for r in rs]),
                     rate([bool(r["transfer_owners"]) and r["transfer_owners"][0] == "red" for r in rs]),
                     mean([r["transfer_found"] for r in rs])])
    games = [r for r in recs if r["experiment"] == "hub_games"]
    return "## e13 Removing card hubness\n\n" + table(
        ["Model", "Boards", "Share of clues pointing first at the most popular card", "K* (same-model)",
         "Transfer: ppmi-cos first guess right", "Transfer: cards found"], rows) + \
        "\n\n" + table(MATCH_HEADER, match_rows(games))


def life_circuit() -> str:
    recs = load("e14_life_circuit.jsonl")
    if not recs:
        return ""
    b = recs[0]["board"]
    rows = []
    for (row,), rs in by(recs, "row").items():
        gens = [r["generation_done"] for r in rs if r["wrong_pixels"] == 0]
        per_gen = " / ".join(mean([g[k] for g in gens], 0) if gens else "-" for k in range(4))
        rows.append([row, len(rs), rate([r["wrong_pixels"] == 0 for r in rs]),
                     mean([r["wrong_pixels"] for r in rs], 1), mean([r["leaks"] for r in rs], 1),
                     mean([r["turns"] for r in rs], 0), mean([r["filler_clues"] for r in rs], 0),
                     sum(r["passes"] for r in rs), per_gen])
    return ("## e14 Game of Life in one game (designed board)\n\n"
            f"life.v, 6x6 torus, 4 generations (144 output pixels): {recs[0]['n_and']} AND gates, depth "
            f"{recs[0]['depth']}. Board with 4,000 filler pairs per team: {b['cards']:,} cards, "
            f"{b['clue_words']:,} clue words. The rankings are set by the compiler, not real words.\n"
            "Last column: mean turn by which each generation's 36 pixels were all readable, over games "
            "with no wrong pixel.\n\n" + table(
                ["Row", "Games", "All 144 pixels right", "Wrong or unreadable pixels", "Leaks", "Turns",
                 "Filler clues", "Passes", "Generation 1 / 2 / 3 / 4 readable by turn"], rows))


SECTIONS = [
    transfer,
    lambda: matches("e02_model_tournament.jsonl", "e02 Model tournament (shared codec)",
                    "SimSpymaster (margin 0) + SimGuesser on one model per team."),
    belief_sweep,
    lambda: matches("e04_risk_appetite.jsonl", "e04 Risk appetite",
                    "w2v, RiskSpymaster knows T 0.5, SoftGuesser at T 0.5; teams differ only in max risk."),
    model_gap,
    assassin,
    learning,
    lambda: matches("e08_eavesdrop_crude.jsonl", "e08 Crude eavesdropping guesser",
                    "T 0.5, max risk 0.2.", mix=True),
    vocab,
    lambda: matches("e10_secrecy.jsonl", "e10 Bayesian listener and secrecy-aware spymaster",
                    "T 0.5, max risk 0.2. In the guess table, the other side's 'Other team's' column is how often it picked A's cards.",
                    mix=True),
    solvability,
    ranked_scale,
    hub,
    life_circuit,
]

if __name__ == "__main__":
    parts = [s() for s in SECTIONS]
    text = "# Results\n\nGenerated by analyze.py from the JSONL files in this folder.\n\n" + \
        "\n\n".join(p for p in parts if p) + "\n"
    with open(os.path.join(RESULTS, "tables.md"), "w") as f:
        f.write(text)
    print(text)
