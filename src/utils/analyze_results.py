import os
import re
import numpy as np

def parse_line(line):
    """
    Parse a single line to extract metrics.
    """
    match = re.search(r"expansions: ([\d,]+), time: ([\d,]+) \[ms\], memory: (\d+) \[kB\], path length: (\d+)(?:, g_F: (\d+), g_B: (\d+))?", line)
    if match:
        expansions = int(match.group(1).replace(",", ""))
        time = int(match.group(2).replace(",", ""))
        memory = int(match.group(3))
        path_length = int(match.group(4))
        g_F = int(match.group(5)) if match.group(5) else None
        g_B = int(match.group(6)) if match.group(6) else None
        return expansions, time, memory, path_length, g_F, g_B
    return None


def average_metrics(folder_path):
    """
    Calculate average metrics for unidirectional s-t, unidirectional t-s, and bidirectional search.
    """
    metrics = {
        "unidirectional_s_t": [],
        "unidirectional_t_s": [],
        "bidirectional": []
    }

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if not '8x8_grid_with_random_blocks_16per' in filename: continue
        file_path = os.path.join(folder_path, filename)

        # Read and process each file
        with open(file_path, 'r') as file:
            for line in file:
                if "unidirectional s-t" in line:
                    result = parse_line(line)
                    if result:
                        metrics["unidirectional_s_t"].append(result)
                elif "unidirectional t-s" in line:
                    result = parse_line(line)
                    if result:
                        metrics["unidirectional_t_s"].append(result)
                elif "bidirectional" in line:
                    result = parse_line(line)
                    if result:
                        metrics["bidirectional"].append(result)

    # Compute averages
    averages = {}
    for key, values in metrics.items():
        if values:
            expansions, times, memories, path_lengths, g_Fs, g_Bs = zip(*values)
            averages[key] = {
                "average_expansions": np.mean(expansions),
                "average_time_ms": np.mean(times),
                "average_memory_kb": np.mean(memories),
                "average_path_length": np.mean(path_lengths),
                "average_g_F": np.mean([g for g in g_Fs if g is not None]) if any(g_Fs) else None,
                "average_g_B": np.mean([g for g in g_Bs if g is not None]) if any(g_Bs) else None,
            }

    return averages

if __name__ == "__main__":
    folder_path = "/home/tzur-shubi/Documents/Programming/BiHS/results/SM_Grids"
    averages = average_metrics(folder_path)
    expansions_summary = []
    time_summary = []

    for search_type, metrics in averages.items():
        print(f"{search_type}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

        # Collect expansions and time summaries
        expansions_summary.append(round(metrics["average_expansions"]))
        time_summary.append(round(metrics["average_time_ms"]))

    print(f"expansions: ({', '.join(map(str, expansions_summary))})")
    print(f"time[ms]: ({', '.join(map(str, time_summary))})")
