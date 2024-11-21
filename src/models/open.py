from bisect import bisect_left

class Open:
    def __init__(self, n):
        """
        Constructor that initializes an OPEN object with n cells.
        Each cell contains a single list, which is initially empty.
        """
        self.cells = [[] for _ in range(n)]

    def insert_state(self, state):
        """
        Inserts a state into the appropriate cell based on the state's head() value,
        while maintaining the list's order by descending g() value.

        :param state: The state to insert.
        """
        if state.head is None:
            raise ValueError("State has no valid head.")

        cell_index = state.head
        
        # Insert the state while maintaining descending order by g()
        cell = self.cells[cell_index]

        # Use binary search to find the correct position
        g_value = state.g
        index = bisect_left([-s.g for s in cell], -g_value)  # Use negative values for descending order
        cell.insert(index, state)

    def remove_state(self, state):
        """
        Removes the specified state from the corresponding cell.

        :param state: The state to remove.
        """
        if state.head is None:
            raise ValueError("State has no valid head.")

        cell_index = state.head
        cell = self.cells[cell_index]

        # Remove the state by identity (not by value equality)
        try:
            cell.remove(state)
        except ValueError:
            raise ValueError("State not found in the cell.")

    def find_highest_non_overlapping_state(self, state, snake=False):
        """
        Finds the state with the highest g() value in the same cell
        that has no common vertices with the given state.

        :param state: The state to compare against.
        :param snake: A boolean indicating whether to consider snake mode.
        :return: The highest g() state that doesn't overlap, or None if no such state exists.
        """
        if state.head is None:
            raise ValueError("State has no valid head.")

        cell_index = state.head
        cell = self.cells[cell_index]
        
        # Iterate over the list, which is sorted by descending g()
        for other_state in cell:
            if not state.shares_vertex_with(other_state, snake):
                return other_state  # Return the first non-overlapping state (highest g() due to sorting)

        return None  # No valid state found
