#!/usr/bin/env python3
"""
Exhaustive backtracking in Q_6 to search for a counterexample coil.

We search for an induced 26-cycle (coil) in the 6D hypercube Q6 such that
NO cyclic window of length 6 in its transition sequence contains all 6
distinct dimensions.

Normalization:
  - start vertex fixed to 0
  - first 3 traversed dimensions fixed to [0,1,2] (w.l.o.g.)

Vertices are integers 0..63 representing 6-bit vectors.
An edge in dimension k toggles bit k: v -> v ^ (1<<k).

Induced-cycle constraint (coil):
  - All visited vertices are distinct (simple cycle)
  - No "chord" edges between nonconsecutive vertices on the cycle:
    when adding a new vertex, it must not be adjacent (Hamming distance 1)
    to any earlier vertex except the immediate predecessor.
  - At closure, the last vertex must be adjacent to start, and must not
    be adjacent to any other vertex except its predecessor and start.

Property we are trying to violate (counterexample):
  - For every cyclic block of 6 consecutive transition dimensions,
    they are NOT all distinct.

So, during search we prune any step that creates a 6-distinct window in the
linear prefix (and at the end we also check the wrap-around windows).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

D = 6
N_EDGES = 26  # optimal coil length in Q6
START = 0

def popcount(x: int) -> int:
    return x.bit_count()

def adjacent(u: int, v: int) -> bool:
    # In Q_d, adjacent iff XOR is a power of two (Hamming distance 1)
    x = u ^ v
    return x != 0 and (x & (x - 1)) == 0

def dim_of_edge(u: int, v: int) -> int:
    x = u ^ v
    # x is power of 2, return index
    return (x.bit_length() - 1)

def window_all_distinct(dims: List[int]) -> bool:
    return len(set(dims)) == len(dims)

def has_6_distinct_window_cyclic(seq: List[int]) -> bool:
    """Check if seq (length 26) has any cyclic window of length 6 with 6 distinct dims."""
    n = len(seq)
    assert n == N_EDGES
    for i in range(n):
        w = [seq[(i + t) % n] for t in range(D)]
        if len(set(w)) == D:
            return True
    return False

@dataclass
class State:
    path: List[int]          # vertices, length = steps+1, starts with START
    seq: List[int]           # transition dims so far, length = steps
    visited: Set[int]        # vertices in path
    # optional speed-up: store all visited vertices as list for iteration
    # (path already has them in order)

def violates_induced_constraint(new_v: int, st: State) -> bool:
    """
    When adding new_v as the next vertex in the path:
      - it cannot be visited already
      - it cannot be adjacent to any earlier vertex except the immediate predecessor
    """
    if new_v in st.visited:
        return True

    prev = st.path[-1]

    # new_v is adjacent to prev by construction; now forbid adjacency to any other
    # previously visited vertex (induced/chordless condition)
    for u in st.path[:-1]:  # all earlier vertices excluding prev
        if adjacent(new_v, u):
            return True
    return False

def closure_is_valid(st: State) -> bool:
    """
    We have a full path of length 26 edges, so st.path has 26 vertices + start? Actually:
      - path length should be N_EDGES (26) vertices visited after 25 edges? Let's define carefully:
    In this script:
      - st.seq length = number of edges taken so far
      - st.path length = len(st.seq) + 1
    At completion we want len(st.seq) == 26 and st.path length == 27,
    and st.path[-1] must be adjacent to START and then we close with that final edge
    already included in seq by construction.
    However we construct edges one by one; so at completion:
      - last vertex st.path[-1] should equal START? No, we keep vertices distinct,
        so last vertex must be adjacent to START and we "close" conceptually.
    We'll represent the cycle as:
      vertices v0=START, v1,...,v25 (26 distinct vertices)
      edges between vi and v_{i+1} for i=0..24, plus edge between v25 and v0
      transition seq has 26 dims: dims[0..24] for vi->v_{i+1} and dims[25] for v25->v0
    So we must add the closing edge dim at the end and NOT add START again to path.
    """
    if len(st.seq) != N_EDGES:
        return False
    if len(st.path) != N_EDGES + 1:
        return False

    v_last = st.path[-1]

    # The last edge in st.seq is intended to be v_last -> START,
    # so v_last must be adjacent to START by that dim.
    if not adjacent(v_last, START):
        return False

    # Induced condition at closure: START is adjacent to v1 (first move),
    # and adjacent to v_last (closing), and must NOT be adjacent to any other vertex.
    # Also v_last must not be adjacent to any other vertex besides its predecessor and START
    v1 = st.path[1]
    v_prev = st.path[-2]

    # Check START adjacency only to v1 and v_last
    for u in st.path[2:-1]:  # exclude START, v1, v_last
        if adjacent(START, u):
            return False

    # Check v_last adjacency only to v_prev and START (and itself not relevant)
    for u in st.path[:-2]:  # exclude v_prev and v_last itself
        if adjacent(v_last, u):
            return False

    return True

def backtrack_find_counterexample() -> Optional[Tuple[List[int], List[int]]]:
    """
    Returns (vertex_path, transition_seq) of a counterexample coil if found, else None.
    transition_seq is length 26; vertex_path is length 27 (START then 26 distinct vertices).
    """
    # Build the forced prefix: dims [0,1,2]
    path = [START]
    visited = {START}
    seq: List[int] = []

    cur = START
    for d in [0, 1, 2]:
        nxt = cur ^ (1 << d)
        # apply induced constraint w.r.t. existing path
        tmp = State(path=path, seq=seq, visited=visited)
        if violates_induced_constraint(nxt, tmp):
            raise RuntimeError("Normalization prefix invalid (should not happen).")
        path.append(nxt)
        visited.add(nxt)
        seq.append(d)
        cur = nxt

    st0 = State(path=path, seq=seq, visited=visited)

    # We now need to choose edges dims[3..24] for internal steps (22 edges),
    # and dim[25] for closure (last vertex adjacent to START).
    # We'll recurse choosing next dimension (0..5), creating next vertex.

    def rec(st: State) -> Optional[Tuple[List[int], List[int]]]:
        steps = len(st.seq)

        # If we have chosen 25 edges (i.e., about to choose the closing edge),
        # we will choose the 26th dim to close to START and then validate.
        if steps == N_EDGES - 1:
            v_last = st.path[-1]
            # closing dim must be the bit that transforms v_last to START
            x = v_last ^ START
            if x == 0 or (x & (x - 1)) != 0:
                return None
            closing_dim = x.bit_length() - 1

            # Reject if consecutive equal dims (would backtrack into START?):
            if closing_dim == st.seq[-1]:
                return None

            # Prune if the new edge creates a 6-distinct window in the linear sequence end.
            candidate_seq = st.seq + [closing_dim]
            if len(candidate_seq) >= D:
                w = candidate_seq[-D:]
                if len(set(w)) == D:
                    return None

            # Now check wrap-around windows as well (full cyclic check) AND induced closure validity.
            # Note: closure validity checks induced/chords.
            st_closed = State(path=st.path, seq=candidate_seq, visited=st.visited)
            if not closure_is_valid(st_closed):
                return None
            if has_6_distinct_window_cyclic(candidate_seq):
                return None

            # Found a counterexample
            return (st.path.copy(), candidate_seq)

        # Otherwise choose the next dim for a new internal vertex
        cur = st.path[-1]
        prev_dim = st.seq[-1]

        for dim in range(D):
            if dim == prev_dim:
                continue  # would immediately go back

            nxt = cur ^ (1 << dim)

            # Induced constraint while growing
            if nxt in st.visited:
                continue

            # chordless: nxt must not be adjacent to any earlier vertex except cur
            chord = False
            for u in st.path[:-1]:
                if adjacent(nxt, u):
                    chord = True
                    break
            if chord:
                continue

            # Prune: if adding this dim creates a length-6 window with 6 distinct dims, reject
            if len(st.seq) + 1 >= D:
                w = st.seq[-(D - 1):] + [dim]
                if len(set(w)) == D:
                    continue

            # Apply step
            st.path.append(nxt)
            st.visited.add(nxt)
            st.seq.append(dim)

            ans = rec(st)
            if ans is not None:
                return ans

            # Undo
            st.seq.pop()
            st.visited.remove(nxt)
            st.path.pop()

        return None

    return rec(st0)

def main() -> None:
    res = backtrack_find_counterexample()
    if res is None:
        print("No counterexample found under normalization start=0, prefix dims [0,1,2].")
        print("Meaning: every induced 26-cycle respecting that normalization")
        print("has at least one cyclic window of length 6 with all 6 dims distinct.")
        return

    vpath, tseq = res
    print("FOUND COUNTEREXAMPLE!")
    print("Transition sequence (dims):", tseq)
    print("Vertex path (start + 26 vertices):", vpath)
    print("Has 6-distinct cyclic window?", has_6_distinct_window_cyclic(tseq))

if __name__ == "__main__":
    main()
