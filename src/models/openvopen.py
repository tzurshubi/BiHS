import random

class _TreapNode:
    __slots__ = ("key", "val", "prio", "left", "right")
    def __init__(self, key, val):
        self.key  = key
        self.val  = val
        self.prio = random.randrange(1 << 30)
        self.left = None
        self.right = None

class _Treap:
    """
    Randomized balanced BST (treap) keyed by tuples (g, uid).
    Natural in-order yields ascending by g then uid.
    We’ll traverse *descending* by doing reverse in-order.
    """
    def __init__(self):
        self.root = None
        self.size = 0

    # --- rotations ---------------------------------------------------------
    def _rotate_right(self, y):
        x = y.left
        y.left = x.right
        x.right = y
        return x

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        y.left = x
        return y

    # --- insert ------------------------------------------------------------
    def insert(self, key, val):
        def _insert(node, key, val):
            if not node:
                return _TreapNode(key, val)
            if key < node.key:
                node.left = _insert(node.left, key, val)
                if node.left.prio < node.prio:
                    node = self._rotate_right(node)
            else:
                node.right = _insert(node.right, key, val)
                if node.right.prio < node.prio:
                    node = self._rotate_left(node)
            return node
        self.root = _insert(self.root, key, val)
        self.size += 1

    # --- delete by key -----------------------------------------------------
    def remove(self, key):
        found = [False]
        def _remove(node, key):
            if not node:
                return None
            if key < node.key:
                node.left = _remove(node.left, key)
            elif key > node.key:
                node.right = _remove(node.right, key)
            else:
                found[0] = True
                # merge children by rotating the higher-priority one up
                if not node.left and not node.right:
                    return None
                elif not node.left:
                    node = self._rotate_left(node)
                    node.left = _remove(node.left, key)
                elif not node.right:
                    node = self._rotate_right(node)
                    node.right = _remove(node.right, key)
                else:
                    if node.left.prio < node.right.prio:
                        node = self._rotate_right(node)
                        node.right = _remove(node.right, key)
                    else:
                        node = self._rotate_left(node)
                        node.left = _remove(node.left, key)
            return node

        self.root = _remove(self.root, key)
        if found[0]:
            self.size -= 1
        return found[0]

    # --- descending iterator -----------------------------------------------
    def iter_desc(self):
        """Yield values in descending order of key (i.e., descending g then uid)."""
        stack = []
        cur = self.root
        # reverse in-order: right -> node -> left
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.right
            cur = stack.pop()
            yield cur.val
            cur = cur.left

    def __len__(self):
        return self.size


class Openvopen:
    def __init__(self, n):
        """
        OPENvOPEN with n cells.
        Each cell contains two BSTs (treaps): F and B, initially empty.
        """
        self.cells = [{'F': _Treap(), 'B': _Treap()} for _ in range(n)]
        self.counter = 0  # number of states inserted
        # map state object -> (cell_index, target_char, key_tuple)
        self._loc = {}

    @staticmethod
    def _make_key(state):
        """
        Key used in the BST: (g, uid) so ordering is by g, then tie-broken by uid.
        We’ll iterate treap in descending order, so highest g comes first.
        """
        # Using id(state) as stable per-object uid; if you prefer, use your own counter.
        return (state.g, id(state))

    def insert_state(self, state, is_f):
        """
        Insert into F (is_f=True) or B (is_f=False) of the cell indexed by state.head,
        keeping descending order by g (via BST order + reverse traversal).
        """
        if state.head is None:
            raise ValueError("State has no valid head.")
        cell_index = state.head
        target = 'F' if is_f else 'B'
        treap = self.cells[cell_index][target]
        key = self._make_key(state)

        treap.insert(key, state)
        self._loc[state] = (cell_index, target, key)
        self.counter += 1

    def remove_state(self, state, is_f):
        """
        Remove the exact state object from the chosen direction tree (F/B).
        """
        if state.head is None:
            raise ValueError("State has no valid head.")

        info = self._loc.get(state)
        if not info:
            raise ValueError("State not found (no recorded location).")
        cell_index, target_recorded, key = info

        target_expected = 'F' if is_f else 'B'
        if target_recorded != target_expected:
            # If you want to allow removing regardless of is_f flag, you could try both.
            raise ValueError("State found in the other direction tree than requested.")

        treap = self.cells[cell_index][target_recorded]
        ok = treap.remove(key)
        if not ok:
            raise ValueError("State not found in the target BST.")
        self.counter -= 1
        del self._loc[state]

    def find_highest_non_overlapping_state(self, state, is_f, best_path_length, f_max, snake=False):
        """
        Find the first (highest-g) opposite-direction state in the same head cell
        that does not share vertices with `state` (per `shares_vertex_with`),
        with early stopping like the original logic.

        Returns: (opposite_state_or_None, num_checks, num_checks_sum_g_under_f_max)
        """
        if state.head is None:
            raise ValueError("State has no valid head.")

        num_checks = 0
        num_checks_sum_g_under_f_max = 0

        cell_index = state.head
        opposite = 'B' if is_f else 'F'
        treap = self.cells[cell_index][opposite]

        # Iterate in descending g order
        for opposite_state in treap.iter_desc():
            num_checks += 1
            if state.g + opposite_state.g < f_max:
                num_checks_sum_g_under_f_max += 1

            # If even the current (best remaining) fails the best_path_length bound,
            # all later (smaller g) will fail as well → early exit with None.
            if state.g + opposite_state.g <= best_path_length:
                return None, num_checks, num_checks_sum_g_under_f_max

            # Check vertex overlap
            if not state.shares_vertex_with(opposite_state, snake):
                return opposite_state, num_checks, num_checks_sum_g_under_f_max

        return None, num_checks, num_checks_sum_g_under_f_max


# from bisect import bisect_left

# class Openvopen:
#     def __init__(self, n):
#         """
#         Constructor that initializes an OPENvOPEN object with n cells.
#         Each cell contains two lists: F and B, both of which are initially empty.
#         """
#         self.cells = [{'F': [], 'B': []} for _ in range(n)]
#         self.counter = 0 # Counter to track the number of states inserted

#     def insert_state(self, state, is_f):
#         """
#         Inserts a state into the appropriate list (F or B) of the corresponding cell
#         based on the state's head() value, while maintaining the list's order by
#         descending g() value.

#         :param state: The state to insert.
#         :param is_f: A boolean indicating whether to insert into F (True) or B (False).
#         """
#         if state.head is None:
#             raise ValueError("State has no valid head.")

#         cell_index = state.head
#         target_list = 'F' if is_f else 'B'
        
#         # Insert the state while maintaining descending order by g()
#         cell = self.cells[cell_index]
#         list_to_update = cell[target_list]

#         # Use binary search to find the correct position
#         g_value = state.g
#         index = bisect_left([-s.g for s in list_to_update], -g_value)  # Use negative values for descending order
#         list_to_update.insert(index, state)

#         self.counter += 1  # Increment the counter for each inserted state

#     def remove_state(self, state, is_f):
#         """
#         Removes the specified state from the appropriate list (F or B) in the corresponding cell.

#         :param state: The state to remove.
#         :param is_f: A boolean indicating the direction of the state (True for F, False for B).
#         """
#         if state.head is None:
#             raise ValueError("State has no valid head.")

#         cell_index = state.head
#         target_list = 'F' if is_f else 'B'
        
#         cell = self.cells[cell_index]
#         list_to_update = cell[target_list]

#         # Remove the state by identity (not by value equality)
#         try:
#             list_to_update.remove(state)
#             self.counter -= 1  # Decrement the counter for each removed state
#         except ValueError:
#             raise ValueError("State not found in the target list.")

#     def find_highest_non_overlapping_state(self, state, is_f, best_path_length, f_max, snake = False):
#         """
#         Finds the state with the highest g() value from the opposite direction
#         that shares the same head() but has no common vertices with the given state.

#         :param state: The state to compare against.
#         :param is_f: A boolean indicating the direction of the given state (True for F, False for B).
#         :return: The highest g() state from the opposite direction, or None if no such state exists.
#         """
#         if state.head is None:
#             raise ValueError("State has no valid head.")

#         num_checks = 0
#         num_checks_sum_g_under_f_max = 0

#         # Determine the opposite list based on is_f
#         cell_index = state.head
#         opposite_list = 'B' if is_f else 'F'
        
#         cell = self.cells[cell_index]
#         opposite_states = cell[opposite_list]
        
#         # Iterate over the opposite list, which is sorted by descending g()
#         for opposite_state in opposite_states:
#             # Count the number of checks
#             num_checks += 1
#             if state.g + opposite_state.g < f_max:
#                 num_checks_sum_g_under_f_max += 1

#             # Check if the sum of g() values is less than or equal to the best path length
#             if state.g + opposite_state.g <= best_path_length:
#                 return None, num_checks, num_checks_sum_g_under_f_max
            
#             #Check if this is a valid meeting point
#             if not state.shares_vertex_with(opposite_state, snake):
#                 return opposite_state, num_checks, num_checks_sum_g_under_f_max  # Return the first non-overlapping state (highest g() due to sorting)

#         return None, num_checks, num_checks_sum_g_under_f_max  # No valid state found

