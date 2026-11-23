"""Game of Life helpers for the life.v design: reference simulation and readout as frames."""
from .compile import Circuit

W = H = 6
T = 4


def life_ref(state: int, gens: int, w: int = W, h: int = H) -> list[int]:
    """Plain Python Life on a w x h torus. Bit y*w + x is cell (x, y). Returns generations 1..gens."""
    hist = []
    s = state
    for _ in range(gens):
        t = 0
        for y in range(h):
            for x in range(w):
                n = sum((s >> (((y + dy) % h) * w + (x + dx) % w)) & 1
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy)
                alive = (s >> (y * w + x)) & 1
                if n == 3 or (alive and n == 2):
                    t |= 1 << (y * w + x)
        hist.append(t)
        s = t
    return hist


def grid_bits(cells: list[tuple[int, int]], w: int = W) -> int:
    v = 0
    for x, y in cells:
        v |= 1 << (y * w + x)
    return v


GLIDER = grid_bits([(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)])


def pixel(circuit: Circuit, revealed: list[bool], g: int, i: int, w: int = W, h: int = H) -> int | None:
    """Pixel i of generation g+1 as read off the board: 1, 0, or None if unreadable."""
    c0, c1 = circuit.outputs[("hist", g * w * h + i)]
    if revealed[c1] and not revealed[c0]:
        return 1
    if revealed[c0] and not revealed[c1]:
        return 0
    return None


def wrong_pixels(circuit: Circuit, revealed: list[bool], s0: int, gens: int = T, w: int = W, h: int = H) -> int:
    """Pixels that read the wrong value or can't be read, over all generations."""
    ref = life_ref(s0, gens, w, h)
    return sum(pixel(circuit, revealed, g, i, w, h) != (ref[g] >> i) & 1
               for g in range(gens) for i in range(w * h))





def generation_turns(circuit: Circuit, history, width: int, gens: int = T, w: int = W, h: int = H) -> list[int | None]:
    """For each generation, the turn (counted from 1) on which its last pixel first became
    readable, or None if some pixel never did."""
    first: dict[int, int] = {}   # output index -> turn
    card_out: dict[int, int] = {}
    for (name, k), (c0, c1) in circuit.outputs.items():
        if name == "hist":
            card_out[c0] = card_out[c1] = k
    for e in history:
        if e.kind == "guess" and e.pos is not None:
            out_k = card_out.get(e.pos[0] * width + e.pos[1])
            if out_k is not None and out_k not in first:
                first[out_k] = e.turn + 1
    out: list[int | None] = []
    for g in range(gens):
        ks = range(g * w * h, (g + 1) * w * h)
        out.append(max(first[k] for k in ks) if all(k in first for k in ks) else None)
    return out
