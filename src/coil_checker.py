import pickle
import argparse
import time

DEFAULT_ABOVE = 0
DEFAULT_RUN_FOREVER = False 

# Function to parse command-line arguments
def parse_args():
    """
    Parse command-line arguments to handle only one input:
    --above: A threshold value.
    """
    parser = argparse.ArgumentParser(description="Run graph search experiments.")
    parser.add_argument("--above", type=int, default=DEFAULT_ABOVE, help="Threshold value for the experiment.")
    parser.add_argument("--run_forever", action="store_true", default=DEFAULT_RUN_FOREVER, help="Make the program run forever.")

    return parser.parse_args()

# Load the Open data structures from files
def load_open(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# Function to find the longest coil
def find_longest_coil(above, run_forever, snake=True):
    longest_coil = []
    longest_length = -1
    ran_once = False
    runs = 0

    while not ran_once or run_forever:
        runs +=1 

        # Load OpenF and OpenB
        if not ran_once:
            time.sleep(5)
        openF_file = "openF"
        openB_file = "openB"
        print("Loading OpenF and OpenB...")
        try:
            openF = load_open(openF_file)
            openB = load_open(openB_file)
        except:
            time.sleep(2)
            continue

        print(f"{runs} longest length: {longest_length}")
        print(f"{runs} longest coil: {longest_coil}")
        with open('longest.txt', 'w') as file:
            file.write(f"longest_coil = {longest_coil}\n")
            file.write(f"longest_length = {longest_length}\n")
            file.write(f"runs = {runs}\n")

        ran_once = True
        # Iterate over indices
        for index in range(len(openF.cells)):
            # print(f"Processing index {index}...")

            listF = openF.cells[index]
            listB = openB.cells[index]

            # Iterate over the lists to find the longest coil
            i, j = 0, 0  # Pointers for listF and listB
            while i < len(listF) and j < len(listB):
                stateF = listF[i]
                stateB = listB[j]

                # Compute the combined g-value
                combined_g = stateF.g + stateB.g + 1

                # Break the loop if the combined g-value is less than the longest found
                if combined_g <= longest_length:
                    break

                if combined_g <= above:
                    break

                # Check if the two states do not share vertices
                if not stateF.shares_vertex_with(stateB, snake):
                    # Combine the paths
                    combined_path = stateF.path[:-1] + stateB.path[::-1]

                    # Update the longest coil if this one is longer
                    if combined_g > longest_length:
                        longest_coil = combined_path
                        longest_length = combined_g

                    # Move to the next state in both lists (reduce complexity)
                    i += 1
                    j += 1
                else:
                    # If they share vertices, move the pointer with the smaller g-value
                    if stateF.g >= stateB.g:
                        i += 1
                    else:
                        j += 1

    return longest_coil, longest_length


# Main script
if __name__ == "__main__":
    args = parse_args()
    above = args.above
    run_forever = args.run_forever

    # Find the longest coil
    print(f"Finding the longest coil, longer than {above}")
    longest_coil, longest_length = find_longest_coil(above,run_forever, snake=True)

    # Output the result
    print("--------------------------")
    print(f"Longest Coil Length: {longest_length}")
    print(f"Longest Coil Path: {longest_coil}")
