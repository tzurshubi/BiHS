from bisect import bisect_left

class Open:
    def __init__(self, n):
        """
        Constructor that initializes an OPEN object with n cells.
        Each cell contains a single list, which is initially empty.
        """
        self.cells = [[] for _ in range(n)]
        self.counter = 0  # number of states inserted

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
        self.counter += 1


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
        self.counter -= 1
