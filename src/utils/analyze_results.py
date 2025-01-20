import os
import re
import numpy as np

def parse_line(line):
    """
    Parse a single line to extract metrics.
    """
    match = re.search(
        r"expansions: ([\d,]+), time: ([\d,]+) \[ms\], memory: (\d+) \[kB\], path length: (\d+)(?:, g_F: (\d+), g_B: (\d+))?",
        line
    )
    if match:
        expansions = int(match.group(1).replace(",", ""))
        time = int(match.group(2).replace(",", ""))
        memory = int(match.group(3))
        path_length = int(match.group(4))
        g_F = int(match.group(5)) if match.group(5) else None
        g_B = int(match.group(6)) if match.group(6) else None
        return expansions, time, memory, path_length, g_F, g_B
    return None


def average_metrics(folder_path,file_name_substring=""):
    """
    Calculate average (and standard deviation) metrics for:
    - unidirectional s-t
    - unidirectional t-s
    - bidirectional search
    """
    print(folder_path)
    print(file_name_substring)
    metrics = {
        "unidirectional_s_t": [],
        "unidirectional_t_s": [],
        "bidirectional": []
    }

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if file_name_substring not in filename:
            continue
        print(filename)
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

    # Compute averages and standard deviations
    summaries = {}
    for key, values in metrics.items():
        if values:
            expansions, times, memories, path_lengths, g_Fs, g_Bs = zip(*values)

            # Filter out None values for g_F and g_B
            valid_g_Fs = [g for g in g_Fs if g is not None]
            valid_g_Bs = [g for g in g_Bs if g is not None]

            # print(f"{key} - expansion: {expansions}")
            summaries[key] = {
                "expansions":expansions,
                "time":times,

                "average_expansions": np.mean(expansions),
                "std_expansions": np.std(expansions, ddof=1),  # Sample standard deviation

                "average_time_ms": np.mean(times),
                "std_time_ms": np.std(times, ddof=1),

                "average_memory_kb": np.mean(memories),
                "std_memory_kb": np.std(memories, ddof=1),

                "average_path_length": np.mean(path_lengths),
                "std_path_length": np.std(path_lengths, ddof=1),

                "average_g_F": np.mean(valid_g_Fs) if valid_g_Fs else None,
                "std_g_F": np.std(valid_g_Fs, ddof=1) if len(valid_g_Fs) > 1 else None,

                "average_g_B": np.mean(valid_g_Bs) if valid_g_Bs else None,
                "std_g_B": np.std(valid_g_Bs, ddof=1) if len(valid_g_Bs) > 1 else None,
            }
        else:
            # Handle case where there are no lines for that search type
            summaries[key] = {}
    min_uni = [min(a, b) for a, b in zip(summaries['unidirectional_s_t']['expansions'], summaries['unidirectional_t_s']['expansions'])]
    bi = list(summaries['bidirectional']['expansions'])
    print(min_uni)
    print(bi)
    print(f"uni - {len(min_uni)} instances. bi - {len(bi)} instances. ")
    return summaries


if __name__ == "__main__":
    folder_path = "/home/tzur-shubi/Documents/Programming/BiHS/results/SM_Grids"
    file_name_substring = '8x8'
    results = average_metrics(folder_path,file_name_substring)

    # Lists to store average expansions/time and corresponding standard deviations
    expansions_avg = []
    expansions_std = []
    time_avg = []
    time_std = []

    for search_type, metrics_dict in results.items():
        # print(f"{search_type}:")
        # Print out all metrics (including std)
        # for metric_name, value in metrics_dict.items():
        #     print(f"  {metric_name}: {value}")

        # Collect expansions/time averages and std if they exist
        if "average_expansions" in metrics_dict:
            expansions_avg.append(round(metrics_dict["average_expansions"]))
        if "std_expansions" in metrics_dict:
            expansions_std.append(round(metrics_dict["std_expansions"]))

        if "average_time_ms" in metrics_dict:
            time_avg.append(round(metrics_dict["average_time_ms"]))
        if "std_time_ms" in metrics_dict:
            time_std.append(round(metrics_dict["std_time_ms"]))

    # Print expansions and time summaries
    print(f"expansions AVG: ({', '.join(map(str, expansions_avg))}). expansions STD: ({', '.join(map(str, expansions_std))})")
    print(f"time[ms] AVG: ({', '.join(map(str, time_avg))}). time[ms] STD: ({', '.join(map(str, time_std))})")
