import os
import re
import csv
from pathlib import Path
from collections import defaultdict

from plotly.express import line

# ==========================================
# CONFIGURATION
# ==========================================
# Set this to the parent folder produced by split_result_files_to_folders.py
base_dir = "/home/tzur-shubi/Documents/Programming/BiHS/results/2026_05_05"

def get_leaf_dirs(root):
    """Return all subdirectories that contain files but no subdirectories."""
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if not dirnames and filenames:
            leaf_dirs.append(dirpath)
    return leaf_dirs

def get_dir_type(directory):
    parts = set(Path(directory).parts)
    if "CIB" in parts:
        return "cib"
    d = directory.lower()
    if "maze" in d:
        return "maze"
    if "snake" in d:
        return "snake"
    return "grid"

def parse_and_check_results(directory):
    bug_reports = []

    # data[row_key][algorithm][col_key] = (expansions, time_ms)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ("", ""))))

    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return data, bug_reports

    dir_type = get_dir_type(directory)

    for filename in os.listdir(directory):
        if not filename.startswith("results_"):
            continue

        if dir_type == "maze":
            match = re.search(r"results_(\d+x\d+)_maze_(\d+)blocked_", filename)
            if not match:
                continue
            row_key = match.group(1)        # "13x13"
            col_key = match.group(2) + "blocked"  # "0blocked", "1blocked", "2blocked"
        elif dir_type == "cib":
            match = re.search(r"results_(\d+)d_cube_", filename)
            if not match:
                continue
            row_key = match.group(1)  # "4", "5", "6", "7"
            col_key = ""
        else:  # grid or snake
            match = re.search(r"results_(\d+x\d+)_grid_(\d+)per_", filename)
            if not match:
                continue
            row_key = match.group(1)
            col_key = match.group(2) + "%"

        filepath = os.path.join(directory, filename)

        with open(filepath, 'r') as f:
            lines = f.readlines()

        current_graph_id = None
        lengths_for_graph = defaultdict(list)

        for line in lines:
            # 1. Identify the current graph block
            header_match = re.search(r"^----------\s*(.*?)\s*----------", line)
            if header_match:
                current_graph_id = header_match.group(1)
                continue

            # 2. Extract path lengths for the BUG CHECKER
            length_match = re.search(r"path length:\s*([\d,]+)\s*\[edges\]", line)
            if length_match and current_graph_id:
                val = int(length_match.group(1).replace(',', ''))
                lengths_for_graph[current_graph_id].append(val)

            # 3. Extract the final summary stats for the CSV
            summary_match = re.search(r"(A\*|XMM|XDFBnB|BiXDFBnB|XIDA|BiXIDA):\s*([\d,]+)\s*,\s*([\d,]+)\s*\(expansions", line)
            if summary_match:
                alg = summary_match.group(1)
                expansions = summary_match.group(2).replace(',', '')
                time_ms = summary_match.group(3).replace(',', '')
                data[row_key][alg][col_key] = (expansions, time_ms)

        # Evaluate the bug checker for the current file
        for graph_id, lengths in lengths_for_graph.items():
            if len(set(lengths)) > 1:
                bug_reports.append(f"BUG in {filename} -> Graph '{graph_id}': Conflicting lengths found {lengths}")

    return data, bug_reports

def write_csv(data, directory):
    dir_type = get_dir_type(directory)
    algs = ['A*', 'XMM', 'XDFBnB', 'BiXDFBnB', 'XIDA', 'BiXIDA']
    output_file = os.path.join(directory, "output.csv")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        if dir_type == "maze":
            variants = [("0blocked", "0 blocks"), ("1blocked", "1 block"), ("2blocked", "2 blocks")]
            header = ["Grid", "Algorithm"]
            for _, label in variants:
                header.extend([f"{label} Expansions", f"{label} Time [ms]"])
            writer.writerow(header)

            for grid in ['13x13']:
                for alg in algs:
                    row = [grid, alg]
                    for key, _ in variants:
                        exp, time_ms = data[grid][alg][key]
                        row.extend([exp, time_ms])
                    writer.writerow(row)

        elif dir_type == "cib":
            writer.writerow(["Dimension", "Algorithm", "Expansions", "Time [ms]"])

            for dim in ['4', '5', '6', '7']:
                for alg in algs:
                    exp, time_ms = data[dim][alg][""]
                    writer.writerow([dim, alg, exp, time_ms])

        else:  # grid or snake
            grids = ['7x7', '7x8', '7x9', '8x8', '8x9', '9x9'] if dir_type == "snake" else ['6x6', '6x7', '6x8', '7x7', '7x8', '8x8']
            percents = ['20%', '16%', '12%', '8%', '4%']

            header = ["Grid", "Algorithm"]
            for p in percents:
                header.extend([f"{p} Expansions", f"{p} Time [ms]"])
            writer.writerow(header)

            for grid in grids:
                for alg in algs:
                    row = [grid, alg]
                    for p in percents:
                        exp, time_ms = data[grid][alg][p]
                        row.extend([exp, time_ms])
                    writer.writerow(row)

if __name__ == "__main__":
    leaf_dirs = get_leaf_dirs(base_dir)

    if not leaf_dirs:
        print(f"No leaf subdirectories found in '{base_dir}'.")

    for directory in leaf_dirs:
        print(f"\n{'='*60}")
        print(f"Scanning: {directory}\n")

        parsed_data, bugs = parse_and_check_results(directory)

        if bugs:
            print("====== ALGORITHM BUGS DETECTED ======")
            for bug in bugs:
                print(f"[!] {bug}")
            print("=====================================\n")
        else:
            print("====== BUG CHECK PASSED ======")
            print("All algorithms reported consistent path lengths across all graphs.\n")

        write_csv(parsed_data, directory)
        print(f"Success! Results table saved to {directory}/output.csv.")
