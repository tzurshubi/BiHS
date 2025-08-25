class Openvopen:
    def __init__(self, n):
        """
        OPENvOPEN with n cells.
        For each cell we maintain two bucketed structures: F and B.
        Each of F and B is a list of n buckets (lists), indexed by g in [0..n-1].
        """
        self.n = n
        # cells[i]['F'][g] is a list of states with head=i, in forward direction, and g=g (unsorted bucket)
        # cells[i]['B'][g] is a list of states with head=i, in backward direction, and g=g (unsorted bucket)
        self.cells = [
            {
                'F': [[] for _ in range(n)],
                'B': [[] for _ in range(n)],
            }
            for _ in range(n)
        ]

        self.counter = 0  # number of states inserted
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
        cell_index = state.head
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

    def find_highest_non_overlapping_state(self, state, is_f, best_path_length, f_max, snake=False):
        """
        Scan buckets from highest g down in the opposite direction within the same head cell.
        Keeps the same semantics as your original:
          - Count num_checks per opposite state inspected
          - Count num_checks_sum_g_under_f_max when state.g + opp.g < f_max
          - Early-exit with None if state.g + current_g <= best_path_length
          - Return the first (thus highest-g) opposite state that doesn't share vertices
        Returns: (opposite_state_or_None, num_checks, num_checks_sum_g_under_f_max)
        """
        self._validate_state(state)

        num_checks = 0
        num_checks_sum_g_under_f_max = 0

        cell_index = state.head
        opposite = 'B' if is_f else 'F'
        opp_struct = self.cells[cell_index][opposite]

        # Iterate g from high to low
        for g_candidate in range(self.n - 1, -1, -1):
            # Early bound check (bucket-level): if even the best remaining g fails best_path_length,
            # then all later (smaller g) fail as well.
            if state.g + g_candidate <= best_path_length:
                return None, num_checks, num_checks_sum_g_under_f_max

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
                    return None, num_checks, num_checks_sum_g_under_f_max

                if not state.shares_vertex_with(opposite_state, snake):
                    return opposite_state, num_checks, num_checks_sum_g_under_f_max

        return None, num_checks, num_checks_sum_g_under_f_max
