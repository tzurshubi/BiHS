from models.state import State

class Openvopen:
    def __init__(self, graph, start, goal):
        """
        OPENvOPEN with n cells.
        For each cell we maintain two bucketed structures: F and B.
        Each of F and B is a list of n buckets (lists), indexed by g in [0..n-1].
        """
        self.n = max(graph.nodes) + 1
        self.graph = graph

        # cells[i]['F'][g] is a list of states with head=i, in forward direction, and g=g (unsorted bucket)
        # cells[i]['B'][g] is a list of states with head=i, in backward direction, and g=g (unsorted bucket)
        self.cells = [
            {
                'F': [[] for _ in range(self.n)],
                'B': [[] for _ in range(self.n)],
            }
            for _ in range(self.n)
        ]

        self.counter = 0  # number of states inserted
        self.start = start
        self.goal = goal
        # Map: state -> (cell_index, 'F'/'B', g, index_in_bucket)
        # Enables O(1) removal via swap-pop.
        self._loc = {}

    def _validate_state(self, state):
        if state.head is None:
            raise ValueError("State has no valid head.")
        if not (0 <= state.head < self.n):
            raise ValueError(f"State head {state.head} out of range [0,{self.n-1}].")
        if not (0 <= state.g < self.n):
            raise ValueError(f"State g {state.g} out of range [0,{self.n-1}].")

    def insert_state(self, state, is_f):
        """
        O(1): Append to the bucket indexed by (cell=state.head, dir=F/B, g=state.g).
        """
        self._validate_state(state)
        cell_index = state.head # if state.head not in [self.start, self.goal] else state.path[0]
        target = 'F' if is_f else 'B'
        g_value = state.g

        bucket = self.cells[cell_index][target][g_value]
        bucket.append(state)
        idx = len(bucket) - 1
        self._loc[state] = (cell_index, target, g_value, idx)
        self.counter += 1

    def remove_state(self, state, is_f):
        """
        O(1): Remove exact state object from its bucket using swap-pop.
        """
        self._validate_state(state)

        info = self._loc.get(state)
        if not info:
            raise ValueError("State not found (no recorded location).")

        cell_index, target_recorded, g_value, idx = info
        target_expected = 'F' if is_f else 'B'
        if target_recorded != target_expected:
            # If you'd like to allow removal without trusting is_f, you could skip this check.
            raise ValueError("State found in the other direction than requested.")

        bucket = self.cells[cell_index][target_recorded][g_value]
        last_idx = len(bucket) - 1
        if idx < 0 or idx > last_idx:
            raise ValueError("Corrupted index for state location.")

        # Swap-pop to O(1) delete
        if idx != last_idx:
            bucket[idx] = bucket[last_idx]
            # Update moved element's location index
            moved = bucket[idx]
            self._loc[moved] = (cell_index, target_recorded, g_value, idx)

        bucket.pop()
        self.counter -= 1
        del self._loc[state]

    def is_path_st(self, s):
        """
        Check if a given path (list of vertices) exists in OPENvOPEN as a full s-t path.
        """
        return (self.start == s.tailtip and self.goal == s.head) or \
                (self.goal == s.tailtip and self.start == s.head)

    def find_longest_non_overlapping_state(self, state, is_f, best_path_length, f_max, snake=False):
        """
        Scan buckets from highest g down in the opposite direction within the same head cell.
        Features:
          - Count num_checks per opposite state inspected
          - Count num_checks_sum_g_under_f_max when state.g + opp.g < f_max
          - Early-exit with None if state.g + current_g <= best_path_length
          - Return the first (thus highest-g) opposite state that doesn't share vertices
        Returns: (opposite_state_or_None, num_checks, num_checks_sum_g_under_f_max)
        """
        self._validate_state(state)

        num_checks = 0
        num_checks_sum_g_under_f_max = 0

        if self.is_path_st(state):
            return None, -1, state, state.g, num_checks, num_checks_sum_g_under_f_max

        cell_index = state.head # if state.head not in [self.start, self.goal] else state.path[0]
        opposite = 'B' if is_f else 'F'
        opp_struct = self.cells[cell_index][opposite]
        solution = None
        solution_g = -1

        # Iterate g from high to low
        for g_candidate in range(self.n - 1, -1, -1):
            # Early bound check (bucket-level): if even the best remaining g fails best_path_length,
            # then all later (smaller g) fail as well.
            if state.g + g_candidate <= best_path_length:
                return None, -1, solution, solution_g, num_checks, num_checks_sum_g_under_f_max

            bucket = opp_struct[g_candidate]
            if not bucket:
                continue

            # Iterate all opposite states with this g
            for opposite_state in bucket:
                num_checks += 1
                if state.g + g_candidate < f_max:
                    num_checks_sum_g_under_f_max += 1

                # Original per-state early check (kept for parity)
                if state.g + g_candidate <= best_path_length:
                    return None, -1, solution, solution_g, num_checks, num_checks_sum_g_under_f_max

                if not state.shares_vertex_with(opposite_state, snake):
                    solution = state + opposite_state
                    solution_g = state.g + opposite_state.g
                    return opposite_state, opposite_state.g, solution, solution_g, num_checks, num_checks_sum_g_under_f_max 

        return None, -1, solution, solution_g, num_checks, num_checks_sum_g_under_f_max

    def find_all_non_overlapping_paths(self, state, is_f, best_path_length, f_max, segment_key, snake=False):
        """
        Find all non-overlapping simple paths formed by concatenating `state`
        with states in the opposite direction within the same head cell.

        For each opposite_state in the same head cell:
          - ensure no vertex-overlap (except the shared head, handled by concatenation)
          - (optionally) require state.g + opp.g > best_path_length
          - build a full simple path by concatenating the two partial paths.

        Returns:
            (full_paths, num_checks, num_checks_sum_g_under_f_max)

            full_paths: list of lists of vertices, each a simple path from s to t
                        constructed from (state, opposite_state).
        """
        self._validate_state(state)

        num_checks = 0
        num_checks_sum_g_under_f_max = 0
        full_paths = []

        cell_index = state.head # if state.head not in [self.start, self.goal] else state.path[0]
        opposite = 'B' if is_f else 'F'
        opp_struct = self.cells[cell_index][opposite]

        # Scan all g buckets from high to low (order not essential here, but keeps style)
        for g_candidate in range(self.n - 1, -1, -1):
            bucket = opp_struct[g_candidate]
            if not bucket:
                continue

            for opposite_state in bucket:
                num_checks += 1
                total_g = state.g + g_candidate

                if total_g < f_max:
                    num_checks_sum_g_under_f_max += 1

                # If you only care about combinations that can beat the current best:
                if total_g <= best_path_length:
                    continue

                # Check vertex overlap (other than the head, which we handle in concatenation)
                if state.shares_vertex_with(opposite_state, snake):
                    continue

                # Build full simple path depending on direction
                if is_f:
                    # state: s -> ... -> head
                    # opposite_state: t -> ... -> head
                    full_path = state.path[:-1] + opposite_state.path[::-1]
                    full_path_state = State(self.graph, full_path, [state.path[-1]], snake)
                else:
                    # state: t -> ... -> head
                    # opposite_state: s -> ... -> head
                    full_path = opposite_state.path[:-1] + state.path[::-1]
                    full_path_state = State(self.graph, full_path, [state.path[-1]], snake)

                full_paths.append(full_path_state)

        return full_paths, num_checks, num_checks_sum_g_under_f_max

    def get_states_ending_in(self, vertex, is_f):
        """
        Return all states whose head == vertex in the given direction (F/B),
        in descending order of g (longest paths first).

        Args:
            vertex (int): the head vertex to look for.
            is_f (bool): True for forward (F), False for backward (B).

        Returns:
            list[State]: all matching states, sorted by descending g.
        """
        if not (0 <= vertex < self.n):
            raise ValueError(f"Vertex {vertex} out of range [0,{self.n-1}].")

        direction = 'F' if is_f else 'B'
        dir_struct = self.cells[vertex][direction]

        result = []
        # Scan g from high to low so result is in descending g
        for g in range(self.n - 1, -1, -1):
            bucket = dir_struct[g]
            if bucket:
                # You can extend directly; buckets already hold the State objects
                result.extend(bucket)

        return result

    def __len__(self):
        return self.counter