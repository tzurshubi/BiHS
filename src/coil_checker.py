import pickle

# Load the Open data structures from files
def load_open(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# Function to find the longest coil
def find_longest_coil(openF, openB, snake=True):
    longest_coil = []
    longest_length = -1

    # Iterate over indices
    for index in range(len(openF.cells)):
        print(f"Processing index {index}...")

        listF = openF.cells[index]
        listB = openB.cells[index]

        # Iterate over the lists to find the longest coil
        for stateF in listF:
            for stateB in listB:
                # Check if the two states do not share vertices
                if not stateF.shares_vertex_with(stateB, snake):
                    # Combine the paths
                    combined_path = stateF.path[:-1] + stateB.path[::-1]
                    combined_length = len(combined_path) - 1  # Exclude the head overlap
                    combined_length += 1 # Add the edge (0,1)

                    # Update the longest coil if this one is longer
                    if combined_length > longest_length:
                        longest_coil = combined_path
                        longest_length = combined_length

                    # Break inner loop if the sum of g-values decreases
                    if stateF.g() + stateB.g() < longest_length:
                        break

    return longest_coil, longest_length

# Main script
if __name__ == "__main__":
    # Load OpenF and OpenB
    openF_file = "openF"
    openB_file = "openB"
    print("Loading OpenF and OpenB...")
    openF = load_open(openF_file)
    openB = load_open(openB_file)

    # Find the longest coil
    print("Finding the longest coil...")
    longest_coil, longest_length = find_longest_coil(openF, openB, snake=True)

    # Output the result
    print("--------------------------")
    print(f"Longest Coil Length: {longest_length}")
    print(f"Longest Coil Path: {longest_coil}")
