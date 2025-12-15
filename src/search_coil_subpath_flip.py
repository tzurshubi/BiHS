#!/usr/bin/env python3
"""
Coil-lifting search: try to expand a (longest) coil in Q_d into a (longest) coil in Q_{d+1}
using local actions described.

Action A (3 edges -> 5 edges):
(v1,v2),(v2,v3),(v3,v4)  => replace (v1,v2,v3,v4) with (v1, v1', v2', v3', v4', v4)

Action B (2 edges -> 4 edges):
(v1,v2),(v2,v3)          => replace (v1,v2,v3) with (v1, v1', v2', v3', v3)

Action C (3 edges -> 5 edges, reversed middle order):
For 3 consecutive edges traversing dimensions d1,d2,d3,
replace (v1,v2,v3,v4) with a subpath from v1 that traverses: i, then d3, then d2, then d1, then i again.

where v' = v xor (1<<i), for some dimension i.

We search over sequences of such actions while maintaining the coil-in-the-box constraint
(induced cycle).

Notes:
- Internally the cycle is represented as a list of distinct vertices WITHOUT repeating the start at the end.
- Validity checks treat it as a cycle by connecting last -> first.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


# ---------------------------
# Logging utilities
# ---------------------------

def fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    days = s // 86400
    s %= 86400
    hours = s // 3600
    s %= 3600
    minutes = s // 60
    s %= 60
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{s:02d}"


def make_logger(t0: float, logfile):
    def log(msg: str) -> None:
        line = f"[{fmt_elapsed(time.time() - t0)}] {msg}"
        print(line)
        logfile.write(line + "\n")
        logfile.flush()
    return log


# ---------------------------
# Bit / hypercube utilities
# ---------------------------

def hamming_dist_is_1(a: int, b: int) -> bool:
    x = a ^ b
    return x != 0 and (x & (x - 1)) == 0


def in_range_vertex(v: int, dim: int) -> bool:
    return 0 <= v < (1 << dim)


def rotate_to_min_repr(cycle: List[int]) -> Tuple[int, ...]:
    n = len(cycle)
    if n == 0:
        return tuple()

    best = None
    for start in range(n):
        rot = tuple(cycle[start:] + cycle[:start])
        if best is None or rot < best:
            best = rot

    rev = list(reversed(cycle))
    for start in range(n):
        rot = tuple(rev[start:] + rev[:start])
        if rot < best:
            best = rot

    return best


def set_coil_and_target(dim: int) -> Tuple[List[int], Optional[int]]:
    if dim == 4:
        return [0, 1, 3, 7, 6, 4, 0], 8
    elif dim == 5:
        return [0, 1, 3, 7, 15, 13, 12, 4, 0], 14
    elif dim == 6:
        return [0, 1, 3, 7, 6, 14, 12, 13, 29, 31, 27, 26, 18, 16, 0], 26
    elif dim == 7:
        return [0, 1, 3, 7, 15, 31, 29, 25, 24, 26, 10, 42, 43, 59, 51, 49, 53, 37, 45, 44, 60, 62, 54, 22, 20, 4, 0], 48
    elif dim == 8:
        return [0, 1, 3, 7, 15, 13, 12, 28, 30, 26, 27, 25, 57, 56, 40, 104, 72, 73, 75, 107, 111, 110, 46, 38, 36, 52, 116, 124, 125, 93, 95, 87, 119, 55, 51, 50, 114, 98, 66, 70, 68, 69, 101, 97, 113, 81, 80, 16, 0], 96
    elif dim == 9:
        return [0, 1, 3, 7, 6, 14, 12, 13, 29, 31, 27, 26, 18, 50, 54, 62, 60, 56, 57, 49, 53, 37, 101, 69, 68, 196, 132, 133, 149, 151, 150, 158, 156, 220, 92, 94, 86, 87, 119, 115, 123, 122, 250, 254, 255, 191, 187, 179, 163, 167, 231, 230, 226, 98, 66, 74, 202, 200, 136, 137, 139, 143, 207, 205, 237, 173, 172, 174, 170, 42, 43, 47, 111, 110, 108, 104, 105, 73, 89, 217, 219, 211, 195, 193, 225, 241, 245, 244, 116, 112, 80, 208, 144, 176, 160, 32, 0], None
    else:
        raise ValueError("Please define coil and target for this dimension.")


# ---------------------------
# Coil validity
# ---------------------------

def is_valid_coil(cycle: List[int], dim: int) -> bool:
    n = len(cycle)
    if n < 4:
        return False

    if len(set(cycle)) != n:
        return False

    for v in cycle:
        if not in_range_vertex(v, dim):
            return False

    for i in range(n):
        if not hamming_dist_is_1(cycle[i], cycle[(i + 1) % n]):
            return False

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_dist_is_1(cycle[i], cycle[j]):
                if j != i + 1 and not (i == 0 and j == n - 1):
                    return False

    return True


# ---------------------------
# Actions
# ---------------------------

Action = Tuple[str, int, int]


def applicable_actions(cycle: List[int], dim: int) -> List[Action]:
    n = len(cycle)
    actions = []
    cycle_set = set(cycle)

    for a in range(n):
        v1 = cycle[a]
        v2 = cycle[(a + 1) % n]
        v3 = cycle[(a + 2) % n]
        v4 = cycle[(a + 3) % n]

        d1 = v1 ^ v2
        d2 = v2 ^ v3
        d3 = v3 ^ v4

        for bit in range(dim):
            m = 1 << bit

            # Action A
            v1p, v2p, v3p, v4p = v1 ^ m, v2 ^ m, v3 ^ m, v4 ^ m
            if (
                v1p not in cycle_set and v2p not in cycle_set and
                v3p not in cycle_set and v4p not in cycle_set and
                len({v1p, v2p, v3p, v4p}) == 4 and
                hamming_dist_is_1(v1, v1p) and
                (v1 ^ v2) == (v1p ^ v2p) and
                (v2 ^ v3) == (v2p ^ v3p) and
                (v3 ^ v4) == (v3p ^ v4p) and
                hamming_dist_is_1(v4p, v4)
            ):
                actions.append(("A", a, bit))

            # Action B
            v1p, v2p, v3p = v1 ^ m, v2 ^ m, v3 ^ m
            if (
                v1p not in cycle_set and v2p not in cycle_set and v3p not in cycle_set and
                len({v1p, v2p, v3p}) == 3 and
                hamming_dist_is_1(v1, v1p) and
                (v1 ^ v2) == (v1p ^ v2p) and
                (v2 ^ v3) == (v2p ^ v3p) and
                hamming_dist_is_1(v3p, v3)
            ):
                actions.append(("B", a, bit))

            # Action C (reverse d1,d2,d3 order in the middle)
            w1 = v1 ^ m
            w2 = w1 ^ d3
            w3 = w2 ^ d2
            w4 = w3 ^ d1
            if (
                w1 not in cycle_set and w2 not in cycle_set and w3 not in cycle_set and w4 not in cycle_set and
                len({w1, w2, w3, w4}) == 4 and
                hamming_dist_is_1(v1, w1) and
                hamming_dist_is_1(w1, w2) and
                hamming_dist_is_1(w2, w3) and
                hamming_dist_is_1(w3, w4) and
                hamming_dist_is_1(w4, v4)
            ):
                actions.append(("C", a, bit))

    return actions


def apply_action(cycle: List[int], action: Action) -> List[int]:
    kind, a, bit = action
    n = len(cycle)
    m = 1 << bit

    if kind == "A":
        remove = {(a + 1) % n, (a + 2) % n}
        new_cycle = [cycle[i] for i in range(n) if i not in remove]
        v1 = cycle[a]
        idx = new_cycle.index(v1)
        return new_cycle[:idx+1] + [v1 ^ m, cycle[(a+1)%n]^m, cycle[(a+2)%n]^m, cycle[(a+3)%n]^m] + new_cycle[idx+1:]

    if kind == "B":
        remove = {(a + 1) % n}
        new_cycle = [cycle[i] for i in range(n) if i not in remove]
        v1 = cycle[a]
        idx = new_cycle.index(v1)
        return new_cycle[:idx+1] + [v1 ^ m, cycle[(a+1)%n]^m, cycle[(a+2)%n]^m] + new_cycle[idx+1:]

    if kind == "C":
        v1 = cycle[a]
        v2 = cycle[(a + 1) % n]
        v3 = cycle[(a + 2) % n]
        v4 = cycle[(a + 3) % n]
        d1 = v1 ^ v2
        d2 = v2 ^ v3
        d3 = v3 ^ v4

        w1 = v1 ^ m
        w2 = w1 ^ d3
        w3 = w2 ^ d2
        w4 = w3 ^ d1

        remove = {(a + 1) % n, (a + 2) % n}
        new_cycle = [cycle[i] for i in range(n) if i not in remove]
        idx = new_cycle.index(v1)
        return new_cycle[:idx+1] + [w1, w2, w3, w4] + new_cycle[idx+1:]

    raise ValueError


# ---------------------------
# DFS
# ---------------------------

@dataclass
class SearchConfig:
    dim: int
    target_len: Optional[int]
    time_limit_sec: Optional[float]
    node_limit: Optional[int]
    depth_limit: Optional[int]
    verbose: bool
    log_every: int


@dataclass
class SearchResult:
    best_cycle: List[int]
    best_len: int
    expanded_nodes: int
    found_target: bool
    elapsed_sec: float


def dfs_search(start_cycle: List[int], cfg: SearchConfig, log) -> SearchResult:
    t0 = time.time()
    best_cycle = start_cycle[:]
    best_len = len(start_cycle)
    expanded = 0
    found_target = False
    visited_best = {}

    def rec(cycle, depth):
        nonlocal best_cycle, best_len, expanded, found_target

        if cfg.time_limit_sec and time.time() - t0 >= cfg.time_limit_sec:
            return

        canon = rotate_to_min_repr(cycle)
        if canon in visited_best and visited_best[canon] >= len(cycle):
            return
        visited_best[canon] = len(cycle)

        if len(cycle) > best_len:
            best_len = len(cycle)
            best_cycle = cycle[:]
            log(f"[best] |V|={best_len} depth={depth} expanded={expanded}")

        if cfg.target_len and len(cycle) >= cfg.target_len:
            found_target = True
            log(f"[goal] reached |V|={len(cycle)} expanded={expanded}")
            return

        for act in applicable_actions(cycle, cfg.dim):
            new_cycle = apply_action(cycle, act)
            expanded += 1

            if cfg.log_every and expanded % cfg.log_every == 0:
                log(f"[progress] expanded={expanded} best|V|={best_len} depth={depth}")

            if is_valid_coil(new_cycle, cfg.dim):
                rec(new_cycle, depth + 1)

    rec(start_cycle, 0)
    return SearchResult(best_cycle, best_len, expanded, found_target, time.time() - t0)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    DEFAULT_D = 6

    DEFAULT_TIME = None
    DEFAULT_NODES = None
    DEFAULT_DEPTH = None
    DEFAULT_LOG_EVERY = 1_000_000

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--d",
        type=int,
        default=DEFAULT_D,
        help="Base dimension d (input coil is in Q_d; search runs in Q_(d+1)).",
    )
    # ap.add_argument(
    #     "--target",
    #     type=int,
    #     default=DEFAULT_TARGET,
    #     help="Target cycle size |V| to reach in Q_(d+1). If reached or exceeded, search stops.",
    # )
    ap.add_argument(
        "--time",
        type=float,
        default=DEFAULT_TIME,
        help="Time limit in seconds for the search. Omit for no time limit.",
    )
    ap.add_argument(
        "--nodes",
        type=int,
        default=DEFAULT_NODES,
        help="Maximum number of attempted actions (node expansions). Omit for no node limit.",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help="Maximum number of successful actions applied along a path (DFS depth limit). Omit for no depth limit.",
    )
    ap.add_argument(
        "--log_every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Print a progress line every N attempted actions (expansions). Set to 0 to disable.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (best improvements, progress lines, etc.).",
    )
    ap.add_argument(
        "--coil_json",
        type=str,
        default=None,
        help="Optional path to a JSON list of vertex IDs for a coil in Q_d. "
             "If the list repeats the first vertex at the end, it will be trimmed automatically.",
    )
    args = ap.parse_args()
    d = args.d
    coil, target = set_coil_and_target(d)

    # logger for main prints (elapsed since program start)
    filename = f"coil_subpath_flip_{d}d"
    logfile = open(filename, "w")
    t0_main = time.time()
    log = make_logger(t0_main, logfile)

   # internal format
    start = coil[:-1]

    cfg = SearchConfig(
        dim=d,
        target_len=target,
        time_limit_sec=args.time,
        node_limit=args.nodes,
        depth_limit=args.depth,
        verbose=not args.quiet,
        log_every=args.log_every,
    )

    log(f"Start: |V|={len(start)} (edges={len(start)})  dim=Q_{d}")
    res = dfs_search(start, cfg, log)

    log("Done.")
    log(f"Expanded nodes: {res.expanded_nodes}")
    log(f"Best |V| found: {res.best_len} (edges={res.best_len})")

    if target is not None:
        status = "!" if res.found_target else "Did NOT"
        log(f"{status} Reached target {target}")

    best_cycle = res.best_cycle[:] + [res.best_cycle[0]]
    log(f"Best coil found (cycle list): {best_cycle}")

    logfile.close()

if __name__ == "__main__":
    main()
