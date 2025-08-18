from bisect import bisect_left

class Openvopen:
    def __init__(self, n):
        """
        Constructor that initializes an OPENvOPEN object with n cells.
        Each cell contains two lists: F and B, both of which are initially empty.
        """
        self.cells = [{'F': [], 'B': []} for _ in range(n)]
        self.counter = 0 # Counter to track the number of states inserted

    def insert_state(self, state, is_f):
        """
        Inserts a state into the appropriate list (F or B) of the corresponding cell
        based on the state's head() value, while maintaining the list's order by
        descending g() value.

        :param state: The state to insert.
        :param is_f: A boolean indicating whether to insert into F (True) or B (False).
        """
        if state.head is None:
            raise ValueError("State has no valid head.")

        cell_index = state.head
        target_list = 'F' if is_f else 'B'
        
        # Insert the state while maintaining descending order by g()
        cell = self.cells[cell_index]
        list_to_update = cell[target_list]

        # Use binary search to find the correct position
        g_value = state.g
        index = bisect_left([-s.g for s in list_to_update], -g_value)  # Use negative values for descending order
        list_to_update.insert(index, state)

        self.counter += 1  # Increment the counter for each inserted state

    def remove_state(self, state, is_f):
        """
        Removes the specified state from the appropriate list (F or B) in the corresponding cell.

        :param state: The state to remove.
        :param is_f: A boolean indicating the direction of the state (True for F, False for B).
        """
        if state.head is None:
            raise ValueError("State has no valid head.")

        cell_index = state.head
        target_list = 'F' if is_f else 'B'
        
        cell = self.cells[cell_index]
        list_to_update = cell[target_list]

        # Remove the state by identity (not by value equality)
        try:
            list_to_update.remove(state)
            self.counter -= 1  # Decrement the counter for each removed state
        except ValueError:
            raise ValueError("State not found in the target list.")

    def find_highest_non_overlapping_state(self, state, is_f, best_path_length, f_max, snake = False):
        """
        Finds the state with the highest g() value from the opposite direction
        that shares the same head() but has no common vertices with the given state.

        :param state: The state to compare against.
        :param is_f: A boolean indicating the direction of the given state (True for F, False for B).
        :return: The highest g() state from the opposite direction, or None if no such state exists.
        """
        if state.head is None:
            raise ValueError("State has no valid head.")

        num_checks = 0
        num_checks_sum_g_under_f_max = 0

        # Determine the opposite list based on is_f
        cell_index = state.head
        opposite_list = 'B' if is_f else 'F'
        
        cell = self.cells[cell_index]
        opposite_states = cell[opposite_list]
        
        # Iterate over the opposite list, which is sorted by descending g()
        for opposite_state in opposite_states:
            # Count the number of checks
            num_checks += 1
            if state.g + opposite_state.g < f_max:
                num_checks_sum_g_under_f_max += 1

            # Check if the sum of g() values is less than or equal to the best path length
            if state.g + opposite_state.g <= best_path_length:
                return None, num_checks, num_checks_sum_g_under_f_max
            
            #Check if this is a valid meeting point
            if not state.shares_vertex_with(opposite_state, snake):
                return opposite_state, num_checks, num_checks_sum_g_under_f_max  # Return the first non-overlapping state (highest g() due to sorting)

        return None, num_checks, num_checks_sum_g_under_f_max  # No valid state found

