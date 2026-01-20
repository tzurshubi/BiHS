from typing import Dict, List, Tuple, Optional

from heuristics.heuristic import heuristic
from models.state import State
from models.heapq_state import HeapqState
from models.openvopen import Openvopen



def multidirectional_search1(graph, s, t, v, heuristic_name, snake, args):
    """
    MDS1 Algorithm for GLSP (single mandatory vertex v on the longest s-t path).
    Returns (U, S) where:
      - U is the length in edges (|S|-1),
      - S is the best path as a list of vertices.
    """
    log = args.log
    mov_path = args.graph_image_path.replace(".png", "_mov.mp4") if args.graph_image_path else None
    if isinstance(v, List):
        if len(v) != 1:
            raise ValueError("multidirectional_search1 only supports a single solution vertex v.")
        v = v[0]

    # OPEN lists initialization
    OPEN_0F = HeapqState()  # from s to v
    OPEN_0B = HeapqState()  # from v to s
    OPEN_1F = HeapqState()  # from v to t
    OPEN_1B = HeapqState()  # from t to v

    # P0, P1 store complete segment paths (each is a vertex list)
    P0 = set()
    P1 = set()
    OPENvOPEN0 = Openvopen(graph, s, v)
    OPENvOPEN1 = Openvopen(graph, v, t)

    # Best global answer
    U = -1
    S = []

    # Initial state
    s0F = State(graph, [s], [], snake)
    s0F.h = heuristic(s0F, v, heuristic_name, snake)
    OPEN_0F.push(s0F, s0F.g + s0F.h)
    OPENvOPEN0.insert_state(s0F, True)

    s0B = State(graph, [v], [], snake)
    s0B.h = heuristic(s0B, s, heuristic_name, snake)
    OPEN_0B.push(s0B, s0B.g + s0B.h)
    OPENvOPEN0.insert_state(s0B, False)

    s1F = State(graph, [v], [], snake)
    s1F.h = heuristic(s1F, t, heuristic_name, snake)
    OPEN_1F.push(s1F, s1F.g + s1F.h)
    OPENvOPEN1.insert_state(s1F, True)

    s1B = State(graph, [t], [], snake)
    s1B.h = heuristic(s1B, v, heuristic_name, snake)
    OPEN_1B.push(s1B, s1B.g + s1B.h)
    OPENvOPEN1.insert_state(s1B, False)

    expansions = 0
    generated = 0

    # Main loop
    while (len(OPEN_0F) > 0) or (len(OPEN_0B) > 0) or (len(OPEN_1F) > 0) or (len(OPEN_1B) > 0):

        # ----------------------------
        # f_max as in pseudocode:
        # f_max = min(max f in OPEN_0F, max f in OPEN_0B) + min(max f in OPEN_1F, max f in OPEN_1B)
        # Here: HeapqState.top()[0] is the best-key f of that OPEN (your ordering).
        # If empty -> inf for max-f, That makes f_max become inf when a side is empty, which prevents termination.
        # ----------------------------
        maxf_0F = OPEN_0F.top()[0] if len(OPEN_0F) > 0 else float("inf")
        maxf_0B = OPEN_0B.top()[0] if len(OPEN_0B) > 0 else float("inf")
        maxf_1F = OPEN_1F.top()[0] if len(OPEN_1F) > 0 else float("inf")
        maxf_1B = OPEN_1B.top()[0] if len(OPEN_1B) > 0 else float("inf")

        f_max = min(maxf_0F, maxf_0B) + min(maxf_1F, maxf_1B)

        # Pick (i, D, N) = argmax over all OPEN_{i,D} of f_{i,D}(N)
        i = None
        D = None
        OPEN = None
        fN = float("-inf")
        N = None

        if len(OPEN_0F) > 0 and OPEN_0F.top()[0] > fN:
            fN, _, N = OPEN_0F.top()
            i, D, OPEN = 0, "F", OPEN_0F
        if len(OPEN_0B) > 0 and OPEN_0B.top()[0] > fN:
            fN, _, N = OPEN_0B.top()
            i, D, OPEN = 0, "B", OPEN_0B
        if len(OPEN_1F) > 0 and OPEN_1F.top()[0] > fN:
            fN, _, N = OPEN_1F.top()
            i, D, OPEN = 1, "F", OPEN_1F
        if len(OPEN_1B) > 0 and OPEN_1B.top()[0] > fN:
            fN, _, N = OPEN_1B.top()
            i, D, OPEN = 1, "B", OPEN_1B

        # Should not happen, but safe
        if N is None:
            break
        # if log: print(f"Expanding {N.path} from OPEN_{i}{D} with f={fN}, g={N.g}, path length={len(N.path)-1}.")

        # -----------------------------------------
        # Meeting checks
        # -----------------------------------------
        if (i == 0 and D == "B") or (i == 1 and D == "F"):
            sv_paths, _, _ = OPENvOPEN0.find_all_non_overlapping_paths(N, False, U, fN, snake)
            vt_paths, _, _ = OPENvOPEN1.find_all_non_overlapping_paths(N, True,  U, fN, snake)
        elif (i == 0 and D == "F"):
            sv_paths, _, _ = OPENvOPEN0.find_all_non_overlapping_paths(N, True,  U, fN, snake)
            vt_paths = []
        else: # i == 1 and D == "B"
            sv_paths = []
            vt_paths, _, _ = OPENvOPEN1.find_all_non_overlapping_paths(N, False, U, fN, snake)

        # Add newly found segment paths
        for p in sv_paths:
            P0.add(tuple(p.path))

        for p in vt_paths:
            P1.add(tuple(p.path))

        # Try stitching s->v with v->t
        for p0 in P0:
            for p1 in P1:
                # they must intersect only at v
                if set(p0) & set(p1) != {v}:
                    continue

                if p0[-1] != p1[0]:
                    continue

                full = list(p0) + list(p1[1:])
                U_candidate = len(full) - 1
                if U_candidate > U:
                    U = U_candidate
                    S = full
                    if log: print(f"---({expansions}) Found new best path of length {U}: {S} ---")
                    c=1
        

        # Termination: if U >= f_max return
        if U >= f_max:
            return S, generated, expansions, None

        # We do not expand states whose head is in {s,v,t} (unless g==0).
        # We still allow them to participate in meeting checks above.
        if N.g > 0 and N.head in {s, v, t}:
            # Remove it from the chosen OPEN and move on (no successor generation).
            OPEN.pop()
            continue

        # XMM_full. if g > f_max/2 don't expant it, but keep it in OPENvOPEN for checking collision of search from the other side
        if args.algo in ("cutoff", "full"):
            if (D == "F" and N.g > (fN / 2.0) - 1.0) or (D == "B" and N.g > ((fN - 1.0) / 2.0)):
                # Remove it from OPEN but keep it available for meeting checks in future iterations.
                OPEN.pop()
                # if log: print(f"Skipping expansion of {N.path} from OPEN_{i}{D} due to XMM_full g > f_max/2 condition.")
                # expansions += 1
                continue

        # Remove N from OPEN_{i,D}
        OPEN.pop()
        expansions += 1

        # Expand successors Γ_{i,D}(N)
        directionF = (D == "F")
        successors = N.generate_successors(args, snake, directionF)

        # Determine the target for heuristic based on which OPEN we are expanding
        if i == 0 and D == "F":
            h_target = v
        elif i == 0 and D == "B":
            h_target = s
        elif i == 1 and D == "F":
            h_target = t
        else:
            h_target = v

        # if log: print(f"Generated {len(successors)} successors for {'0F' if i==0 and D=='F' else '0B' if i==0 and D=='B' else '1F' if i==1 and D=='F' else '1B'}: {[st.path for st in successors]}")
        for Np in successors:
            generated += 1
            # If expanding from OPEN_0B or OPEN_1F, insert successors
            # into BOTH OPEN_0B and OPEN_1F with different heuristics.
            if (i == 0 and D == "B") or (i == 1 and D == "F"):
                # Heuristic for OPEN_0B (target = s)
                h_to_s = heuristic(Np, s, heuristic_name, snake)
                if h_to_s >= 0:
                    f_to_s = Np.g + h_to_s
                    # Do NOT rely on Np.h staying as h_to_s; just push with key.
                    OPEN_0B.push(Np, f_to_s)
                    OPENvOPEN0.insert_state(Np, False)

                # Heuristic for OPEN_1F (target = t)
                h_to_t = heuristic(Np, t, heuristic_name, snake)
                if h_to_t >= 0:
                    f_to_t = Np.g + h_to_t
                    OPEN_1F.push(Np, f_to_t)
                    OPENvOPEN1.insert_state(Np, True)

                continue

            # Otherwise: normal single-OPEN insertion (your original)
            h_single = heuristic(Np, h_target, heuristic_name, snake)
            if h_single < 0:
                continue
            f_single = Np.g + h_single
            OPEN.push(Np, f_single)
            OPENvOPEN0.insert_state(Np, True) if i == 0 else OPENvOPEN1.insert_state(Np, False)

    return S, generated, expansions, None