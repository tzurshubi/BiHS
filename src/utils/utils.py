import time
import networkx as nx
import json
import random
import numpy as np
import matplotlib.patches as patches
import os, math, psutil
import matplotlib.pyplot as plt
import re
import csv
from models.state import State
from typing import List, Tuple, Optional

longest_coil_lengths = {
    2: 4,
    3: 6,
    4: 8,
    5: 14,
    6: 26,
    7: 48,
    8: 96,
    9: 192, # upper bound
    10: 384, # upper bound
    11: 768, # upper bound
}

longest_sym_coil_lengths = {
    2: 4,
    3: 6,
    4: 8,
    5: 14,
    6: 26,
    7: 48,   # 46 for finding, 48 for proving
    8: 96,   # 94 for finding, 96 for proving
    9: 186,  # 186 for finding, 192 for proving
    10: 384, # upper bound
    11: 768, # upper bound
}

def ms2str(time_ms):
    """
    Convert elapsed time in milliseconds to a formatted string: 
    - DD:HH:MM:SS (omit DD if 0, omit HH if 0)
    - Always include MM:SS
    
    Args:
        time_ms (float): Elapsed time in milliseconds.
    
    Returns:
        str: A string representing the formatted elapsed time.
    """
    # Convert from milliseconds to seconds
    elapsed_time = time_ms / 1000.0

    # Convert to DD:HH:MM:SS
    days = int(elapsed_time // 86400)
    hours = int((elapsed_time % 86400) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Build the formatted string
    if days > 0:
        return f"{days}:{hours:02}:{minutes:02}:{seconds:02}"
    elif hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"

def time2str(start_time, end_time):
    """
    Calculate elapsed time formatted as DD:HH:MM:SS, but remove days if 00 
    and hours if 00. Always include MM:SS.

    Args:
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        str: A string representing the formatted elapsed time.
    """
    elapsed_time = end_time - start_time

    # Convert to DD:HH:MM:SS
    days = int(elapsed_time // 86400)
    hours = int((elapsed_time % 86400) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Build the formatted string
    if days > 0:
        return f"{days}:{hours:02}:{minutes:02}:{seconds:02}"
    elif hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"

def time_ms(start_time, end_time):
    """
    Calculate elapsed time formatted as DD:HH:MM:SS, but remove days if 00 
    and hours if 00. Always include MM:SS.

    Args:
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        str: A string representing the formatted elapsed time.
    """
    elapsed_time = end_time - start_time

    return int(1000*elapsed_time)
    
def calculate_averages(avgs, log_file_name=None):
    """
    Calculate and print the averages for each metric across categories.

    Parameters:
    avgs (dict): A dictionary where keys are categories (e.g., "uni_st", "uni_ts", "bi"),
                 and values are dictionaries with metrics (e.g., "expansions", "time") as keys and lists of numbers as values.
    """
    metrics = list(next(iter(avgs.values())).keys())  # Get metric names from the first category
    averages_per_metric = {}

    for metric in metrics:
        avgs_list = []
        for category in avgs:
            values = avgs[category].get(metric, [])
            avg = round(sum(values) / len(values)) if values else 0
            avgs_list.append(avg)
        averages_per_metric[metric] = avgs_list

        formatted = ", ".join(str(x) for x in avgs_list)
        line = f"average {metric}: ({formatted})"
        print(line)
        if log_file_name:
            with open(log_file_name, 'a') as f:
                f.write(line + "\n")
                

    # Calculate and print average expansions per second
    expansions_per_second = []
    for category in avgs:
        expansions = avgs[category]["expansions"]
        times = avgs[category]["time"]
        avg_expansions_per_sec = round(sum(expansions) / (sum(times) / 1000)) if times else 0
        expansions_per_second.append(avg_expansions_per_sec)
    formatted_expansions_per_second = ", ".join(f"{eps}" for eps in expansions_per_second)
    # print(f"average expansions per second: ({formatted_expansions_per_second})")
    # if log_file_name:
    #     with open(log_file_name, 'a') as file:
    #         file.write(f"average expansions per second: ({formatted_expansions_per_second})")

    # Summary two lines
    exp_avgs = averages_per_metric.get("expansions", [])
    time_avgs = averages_per_metric.get("time", [])
    if exp_avgs[1]==0: exp_avgs[1]=exp_avgs[0]
    if time_avgs[1]==0: time_avgs[1]=time_avgs[0]
    if len(exp_avgs) >= 2 and len(time_avgs) >= 2:
        first_min_exp = min(exp_avgs[0], exp_avgs[1])
        first_min_time = min(time_avgs[0], time_avgs[1])
        mid_exp = exp_avgs[2] if len(exp_avgs) >=3 else 0
        mid_time = time_avgs[2] if len(time_avgs) >=3 else 0
        last_exp = exp_avgs[-1]
        last_time = time_avgs[-1]

        line1 = f"A*: {first_min_exp} , {first_min_time} (expansions , time[ms])"
        line2 = f"XMM: {mid_exp} , {mid_time} (expansions , time[ms])"
        line3 = f"MDS1: {last_exp} , {last_time} (expansions , time[ms])"
        print()
        print(line1)
        print(line2)
        print(line3)
        if log_file_name:
            with open(log_file_name, 'a') as f:
                f.write("\n" + line1 + "\n")
                f.write(line2 + "\n")
                f.write(line3 + "\n")


# ---------------------------
# Logging utilities
# ---------------------------

def memory_used_mb() -> float:
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss   # Resident Set Size
    mem_mb = mem_bytes / (1024 ** 2)
    return mem_mb

def fmt_elapsed(seconds: float) -> str:
    """
    Format elapsed time as DD:HH:MM:SS.mmm
    """
    total_ms = int(seconds * 1000)

    ms = total_ms % 1000
    s = total_ms // 1000

    days = s // 86400
    s %= 86400
    hours = s // 3600
    s %= 3600
    minutes = s // 60
    s %= 60

    return f"{days:02d}:{hours:02d}:{minutes:02d}:{s:02d}:{ms:03d}"

_INT_RE = re.compile(r"\b\d+\b")

def format_numbers(s: str) -> str:
    def repl(m):
        return f"{int(m.group()):,}"
    return _INT_RE.sub(repl, s)

def make_logger(logfile, t0: float | None = None):
    _t0 = t0
    _closed = False

    def log(msg: str) -> None:
        if _closed:
            return

        msg = format_numbers(msg)   # <<< NEW LINE

        if _t0 is not None:
            line = f"[{fmt_elapsed(time.time() - _t0)}] {msg}"
        else:
            line = msg

        print(line)
        logfile.write(line + "\n")
        logfile.flush()


    def set_t0(t: float | None = None) -> None:
        nonlocal _t0
        _t0 = time.time() if t is None else t

    def close() -> None:
        nonlocal _closed
        if not _closed:
            logfile.flush()
            logfile.close()
            _closed = True

    log.set_t0 = set_t0
    log.close = close

    return log


# ---------------------------
# Coil utilities
# ---------------------------


def check_2_st_paths_form_coil(s1, s2, d: int) -> bool:
    """
    Return True iff two States, each representing an s-t path in Q_d (direction can be s->t or t->s),
    together form a valid coil-in-the-box (an induced cycle) when combined.

    Combination rule:
      - Orient both paths to the same direction s->t
      - Cycle order is: p1 (s->t) + reverse(p2[1:-1]) (t->s without endpoints)
        This yields a simple cycle that uses both s and t exactly once.

    Checks performed:
      1) both paths are simple
      2) both paths are valid hypercube paths (each consecutive edge flips exactly one bit)
      3) they share the same endpoints (unordered)
      4) their internal vertices are disjoint
      5) the combined cycle is a valid hypercube cycle
      6) the cycle is induced (no chords): no nonconsecutive pair in the cycle differs by 1 bit
    """
    if s1 is None or s2 is None:
        return False, None, None

    # Get concrete paths (works if you store path or only parents)
    p1 = s1.path if getattr(s1, "path", None) is not None else s1.materialize_path()
    p2 = s2.path if getattr(s2, "path", None) is not None else s2.materialize_path()

    if not p1 or not p2:
        return False, None, None
    if len(p1) < 2 or len(p2) < 2:
        return False, None, None

    # Endpoints must match as an unordered set (allow reversed paths)
    e1 = (p1[0], p1[-1])
    e2 = (p2[0], p2[-1])
    if {e1[0], e1[1]} != {e2[0], e2[1]}:
        return False, None, None

    # Choose canonical orientation based on p1: s = p1[0], t = p1[-1]
    s = p1[0]
    t = p1[-1]

    # If p1 is actually t->s, flip it (so the function works even if p1 is reversed)
    if p1[0] == t and p1[-1] == s:
        p1 = list(reversed(p1))
        s, t = p1[0], p1[-1]

    # Orient p2 to also be s->t
    if p2[0] == t and p2[-1] == s:
        p2 = list(reversed(p2))
    elif not (p2[0] == s and p2[-1] == t):
        return False, None, None  # endpoints match but not consistent (shouldn't happen, but keep safe)

    # Each path must be simple
    if len(set(p1)) != len(p1) or len(set(p2)) != len(p2):
        return False, None, None
    # Validate path edges are hypercube edges (Hamming distance 1)
    def is_hcube_edge(u: int, v: int) -> bool:
        diff = u ^ v
        return diff != 0 and (diff & (diff - 1)) == 0

    for i in range(1, len(p1)):
        if not is_hcube_edge(p1[i - 1], p1[i]):
            return False, None, None
    for i in range(1, len(p2)):
        if not is_hcube_edge(p2[i - 1], p2[i]):
            return False, None, None

    # Internal vertices must be disjoint for a simple cycle union
    int1 = set(p1[1:-1])
    int2 = set(p2[1:-1])
    if int1 & int2:
        return False, None, None

    # Build combined cycle: p1 (s->t) then return via p2 (t->s) excluding endpoints
    cycle = p1 + list(reversed(p2[1:-1]))

    # Must be a simple cycle
    if len(cycle) < 4:
        return False, None, None
    if len(set(cycle)) != len(cycle):
        return False, None, None

    # Check the cycle's consecutive edges, including wrap-around
    m = len(cycle)
    for i in range(m):
        if not is_hcube_edge(cycle[i], cycle[(i + 1) % m]):
            return False, None, None

    # Induced cycle check (no chords):
    # In Q_d, a chord exists iff two nonconsecutive cycle vertices differ in exactly 1 bit.
    pos = {v: i for i, v in enumerate(cycle)}
    for u in cycle:
        iu = pos[u]
        for bit in range(d):
            v = u ^ (1 << bit)
            iv = pos.get(v)
            if iv is None:
                continue
            # allowed only if consecutive on the cycle
            if (iu - iv) % m in (1, m - 1):
                continue
            return False, None, None

    return True, p1, p2

def node_num_to_bits_on(dim, numbers):
    """
    Convert a list of node numbers to their corresponding bit representations.

    Parameters
    ----------
    dim : int
        The dimension of the cube (number of bits).
    numbers : list of int
        List of node numbers to convert.

    Returns
    -------
    list of list of int
        A list where each element is a list of bit positions that are '1' in the binary representation of the corresponding node number.
    """
    bits_on_list = []
    for n in numbers:
        bits_on = [i for i, b in enumerate(f"{n:0{dim}b}"[::-1], start=1) if b == '1']
        bits_on_list.append(bits_on)
    return bits_on_list

def print_bit_statistics(l):
    # Find the maximum index that appears
    max_index = max((max(sublist) for sublist in l if sublist), default=0)

    for i in range(1, max_index + 1):
        count_on = sum(1 for sublist in l if i in sublist)
        count_off = len(l) - count_on
        print(f"bit {i} on: {count_on} times, bit {i} off: {count_off} times")

def print_bits_pattern(bit_lists):
    for bits in bit_lists:
        if len(bits) == 0:
            print("X")
            continue
        line = " "
        for i in range(1, max(max(sub) if sub else 0 for sub in bit_lists) + 1):
            line += str(i) if i in bits else " "
        print(line.rstrip())  # remove trailing spaces for neatness
    print("X")  # bottom boundary

def bits_to_moves(bit_lists):
    # Convert list of bits turned on to list of moves (bit changes)
    moves = []
    for prev, curr in zip(bit_lists, bit_lists[1:]):
        # XOR logic: the changed bit is the one that’s in one list but not the other
        diff = set(curr) ^ set(prev)
        if len(diff) == 1:
            moves.append(next(iter(diff)))  # extract the single changed bit
        else:
            moves.append(None)  # if something’s wrong (shouldn’t happen in valid coils)
    if len(bit_lists[-1]) > 1:
        print("Warning: Last element has multiple bits on; cannot determine move.")
    return moves + bit_lists[-1]

def coil_dim_crossed_to_vertices(coil_str):
    """
    Convert a coil string into a list of vertex integers.
    coil_str: a string where each char is a digit '0'..'6' (or higher for bigger cubes)
    Returns: list of integer vertex values, starting with vertex 0.
    """

    vertices = [0]               # start at vertex 0
    current = 0

    for ch in coil_str:
        d = int(ch)             # dimension to flip
        bit = 1 << d            # value of that bit

        # flip the bit
        current ^= bit          # XOR toggles the bit
        vertices.append(current)

    return vertices

def swap_dims_vertex(v: int, i: int, j: int) -> int:
    """Swap bits i and j in vertex id v (hypercube coordinate permutation)."""
    if i == j:
        return v
    bi = (v >> i) & 1
    bj = (v >> j) & 1
    if bi == bj:
        return v
    # toggle both bits
    return v ^ ((1 << i) | (1 << j))

def flip_dims_vertex(v: int, dims: list[int]) -> int:
    """Flip (toggle) bits in `dims` of vertex id v."""
    mask = 0
    for d in dims:
        mask |= 1 << d
    return v ^ mask

def symmetric_state_transform(
    s: State,
    flip_dims: list[int],
    dim_swaps: list[tuple[int, int]],
) -> State:
    """
    Apply BOTH transformations to every vertex in s.path:
      1) flip all dimensions in flip_dims (toggle those bits)
      2) swap dimensions according to dim_swaps (sequentially, in order)

    Also applies the same transform to meet_points.
    """
    # if s is None or not s.path:
    #     raise ValueError("symmetric_state_transform expects a State with a non-empty path.")

    s_path = s.materialize_path()
    # Build flip mask once
    flip_mask = 0
    for d in flip_dims:
        flip_mask |= 1 << d

    def transform_vertex(v: int) -> int:
        # First flip bits
        v ^= flip_mask
        # Then apply swaps in order
        for a, b in dim_swaps:
            v = swap_dims_vertex(v, a, b)
        return v

    new_path = [transform_vertex(v) for v in s_path]
    new_meet_points = [transform_vertex(v) for v in (s.meet_points or [])]

    s2 = State(
        graph=s.graph,
        path=new_path,
        meet_points=new_meet_points,
        snake=s.snake,
    )

    # Preserve fields you rely on
    s2.traversed_buffer_dimension = getattr(s, "traversed_buffer_dimension", False)

    # Recompute max_dim_crossed for consistency
    if getattr(s2, "snake", False):
        s2.max_dim_crossed = State._compute_max_dim_crossed_from_path(new_path)

    return s2

def has_bridge_edge_across_dim(a: State, b: State, dim: int) -> bool:
    """
    Fast check: is there exactly ONE hypercube edge between state a's head and state b's head
    that traverses dimension `dim`?

    Interprets “one edge between them” as: their heads are adjacent in Q_n and
    differ exactly in bit `dim` (i.e., b.head == a.head xor (1<<dim)).

    Works in snake and non-snake, O(1).
    """
    if a.head is None or b.head is None:
        return False

    diff = a.head ^ b.head
    # adjacent in hypercube <=> diff is a power of two
    # and traverses dimension dim <=> diff == (1<<dim)
    return diff == (1 << dim)

def print_with_timestamp(message: str):
    """
    Print a message prefixed with the current timestamp.

    Args:
        message (str): The message to print.
    """
    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")

def _is_edge(u: int, v: int) -> bool:
    x = u ^ v
    return x != 0 and (x & (x - 1)) == 0

def is_half_of_symmetric_double_coil(path: List[int], d: int) -> Tuple[bool, Optional[str]]:
    """
    Fast check: is `path` exactly half of a symmetric (double) coil in Q_d,
    in the sense that the transition (dimension) sequence repeats twice?

    Meaning checked:
      Let path be v0, v1, ..., vk  (k = len(path)-1).
      We require that there exists a coil cycle C of length 2k such that:
        - The first k edges of C follow exactly the given path edges.
        - The next k edges repeat the *same dimension sequence* as the first k edges.
        - C closes back to v0, and is an induced cycle (coil-in-the-box).

    Assumptions:
      - `path` itself is already a valid snake-in-the-box (simple, induced path).
        So we do NOT re-check chords/repeats *within the path itself*.
      - We still must verify:
          (A) the doubled construction does not reuse vertices across halves,
          (B) the "middle" and "closing" edges exist,
          (C) induced-cycle property on the *full* 2k-cycle.

    Returns (ok, reason_if_not_ok).
    """
    if d <= 0:
        return False, "d must be >= 1."
    if not path:
        return False, "path is empty."
    if len(path) < 2:
        return False, "path must contain at least 2 vertices (one edge)."

    n = 1 << d
    mask = n - 1

    # Range check (fast)
    # for v in path:
    #     if not isinstance(v, int):
    #         return False, "path contains a non-integer vertex."
    #     if v < 0 or v >= n:
    #         return False, f"vertex {v} is out of range for Q_{d}."

    k = len(path) - 1  # edges in half
    L = 2 * k          # edges in full cycle
    V = 2 * k          # vertices in full cycle (cycle has L vertices)

    # Compute the half transition "masks" e_i = v_i xor v_{i+1}, each must be a power of 2.
    # Also compute total XOR over half, which gives the translation mask M between halves.
    half_edges = [0] * k
    M = 0
    for i in range(k):
        e = path[i] ^ path[i + 1]
        if e == 0 or (e & (e - 1)) != 0:
            return False, f"not a hypercube edge at i={i}: {path[i]} -> {path[i+1]}"
        half_edges[i] = e
        M ^= e

    # Build the full symmetric cycle vertices WITHOUT allocating big structures:
    # First half vertices: A[i] = path[i] for i=0..k-1  (note: do not repeat the last vertex)
    # Second half vertices: B[i] = A[i] xor M for i=0..k-1
    # Cycle order: A[0],...,A[k-1], B[0],...,B[k-1] (length 2k vertices)
    #
    # Edges are:
    #   A[i]--A[i+1] for i=0..k-2  (given)
    #   A[k-1]--B[0] (must be a hypercube edge)
    #   B[i]--B[i+1] for i=0..k-2  (same masks as half_edges)
    #   B[k-1]--A[0] (must be a hypercube edge)
    #
    # Note: B[0] = A[0]^M = path[0]^M. In your example, that equals 21.

    A0 = path[0]
    Akm1 = path[k - 1]  # path has k+1 vertices, but A uses only first k vertices
    B0 = A0 ^ M
    Bkm1 = Akm1 ^ M

    # Middle and closing edges must exist:
    if not _is_edge(Akm1, B0):
        return False, "middle edge A[k-1] -> B[0] is not a hypercube edge; cannot form double coil."
    if not _is_edge(Bkm1, A0):
        return False, "closing edge B[k-1] -> A[0] is not a hypercube edge; cannot close the cycle."

    # Ensure no vertex collision between halves:
    # Need: A[i] != A[j] (assumed by snake), and A[i] != B[j] for all i,j.
    # We'll check A-set membership against all B's.
    # Use array when possible for speed; fall back to set otherwise.
    use_array = n <= 2_000_000
    if use_array:
        seen = bytearray(n)

        # mark A vertices (only first k, not including path[k])
        for i in range(k):
            v = path[i]
            if seen[v]:
                return False, "repeated vertex inside first half (unexpected for a snake)."
            seen[v] = 1

        for i in range(k):
            v = path[i] ^ M
            if seen[v]:
                return False, "vertex overlap between halves; doubled cycle not simple."
            seen[v] = 1
    else:
        Aset = set(path[:k])
        for i in range(k):
            if (path[i] ^ M) in Aset:
                return False, "vertex overlap between halves; doubled cycle not simple."

    # Induced-cycle (no chords) check on the full 2k-cycle:
    # For each vertex v in cycle, any neighbor u that is also in the cycle
    # must be prev/next on the cycle.
    #
    # We'll build a pos lookup for cycle vertices: pos[v] = index in [0..2k-1].
    if use_array:
        pos = [-1] * n
        cycle = [0] * (2 * k)

        # Fill cycle vertices
        for i in range(k):
            v = path[i]
            pos[v] = i
            cycle[i] = v
        base = k
        for i in range(k):
            v = path[i] ^ M
            j = base + i
            pos[v] = j
            cycle[j] = v

        # Check chordlessness
        two_k = 2 * k
        for i in range(two_k):
            v = cycle[i]
            nxt = i + 1 if i + 1 < two_k else 0
            prv = i - 1 if i > 0 else two_k - 1

            for bit in range(d):
                u = v ^ (1 << bit)
                j = pos[u]
                if j == -1:
                    continue
                if j != nxt and j != prv:
                    return False, f"chord found: {v} -- {u} (cycle indices {i} and {j})."
    else:
        cycle = [0] * (2 * k)
        posd = {}

        for i in range(k):
            v = path[i]
            posd[v] = i
            cycle[i] = v
        base = k
        for i in range(k):
            v = path[i] ^ M
            j = base + i
            posd[v] = j
            cycle[j] = v

        two_k = 2 * k
        for i in range(two_k):
            v = cycle[i]
            nxt = i + 1 if i + 1 < two_k else 0
            prv = i - 1 if i > 0 else two_k - 1
            for bit in range(d):
                u = v ^ (1 << bit)
                j = posd.get(u)
                if j is None:
                    continue
                if j != nxt and j != prv:
                    return False, f"chord found: {v} -- {u} (cycle indices {i} and {j})."

    # build the full symmetric coil explicitly
    coil = [path[i] for i in range(k)]
    coil += [path[i] ^ M for i in range(k)]
    coil.append(coil[0])  # close the cycle
    return True, coil


# ---------------------------
# Graph utilities
# ---------------------------

# Function to display the graph
def display_graph(graph, title="Graph", filename=None):
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=500,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# Function to parse results file and write to CSV
def parse_results_file(input_file, output_file):
    """
    Parse a results .txt file and write a structured .csv file.

    Parameters
    ----------
    input_file : str
        Path to the input .txt file with experiment results.
    output_file : str
        Path where the .csv file will be written.
    """

    # Regex patterns
    uni_pattern = re.compile(r"! unidirectional (s-t|t-s)\. expansions: ([\d,]+), time: ([\d,]+) \[ms\]")
    bi_pattern = re.compile(r"! bidirectional\. expansions: ([\d,]+), time: ([\d,]+) \[ms\]")

    rows = []
    grid_index = 0  # increments when we hit a new "SM_Grids/..." header

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # # Detect grid header
            # if line.startswith("SM_Grids/"):
            #     grid_index += 1
            #     continue

            # Match unidirectional
            m_uni = uni_pattern.search(line)
            if m_uni:
                direction, expansions, time = m_uni.groups()
                direction = "s -> t" if direction == "s-t" else "t -> s"
                rows.append([grid_index, direction, expansions, time])
                continue

            # Match bidirectional (XMM)
            m_bi = bi_pattern.search(line)
            if m_bi:
                expansions, time = m_bi.groups()
                rows.append([grid_index, "XMM", expansions, time])
                continue

    # Write to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["grid number", "Direction", "Expansions", "Time [ms]"])
        writer.writerows(rows)