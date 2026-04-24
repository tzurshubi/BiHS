import os
import re
import csv
from collections import defaultdict

from plotly.express import line

# ==========================================
# CONFIGURATION
# ==========================================
# Set this to the path where your results files are located
results_dir = "/home/tzur-shubi/Documents/Programming/BiHS/results/2026_04_24/LSP_Grids/1la" 

def parse_and_check_results(directory):
    bug_reports = []
    
    # Nested dictionary structure: data[grid][algorithm][percent] = (expansions, time)
    # Default values are "-" for missing data
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: ("-", "-"))))

    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return data, bug_reports

    for filename in os.listdir(directory):
        if not filename.startswith("results_"):
            continue

        # Extract grid size and percentage from the filename
        # Expected format: results_{6x6}_grid_{12}per_...
        match = re.search(r"results_(\d+x\d+)_grid_(\d+)per_", filename)
        if not match:
            continue
            
        grid = match.group(1)
        percent = match.group(2) + "%"

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
            summary_match = re.search(r"(A\*|XMM|X-DFBnB|BiX-DFBnB):\s*([\d,]+)\s*,\s*([\d,]+)\s*\(expansions", line)
            if summary_match:
                alg = summary_match.group(1)
                expansions = summary_match.group(2).replace(',', '')
                time_ms = summary_match.group(3).replace(',', '')
                data[grid][alg][percent] = (expansions, time_ms)

        # Evaluate the bug checker for the current file
        for graph_id, lengths in lengths_for_graph.items():
            unique_lengths = set(lengths)
            if len(unique_lengths) > 1:
                bug_reports.append(f"BUG in {filename} -> Graph '{graph_id}': Conflicting lengths found {lengths}")

    return data, bug_reports

def write_csv(data, output_file):
    grids = ['6x6', '6x7', '6x8', '7x7', '7x8', '8x8'] if "snake" not in results_dir.lower() else ['7x7', '7x8', '7x9', '8x8', '8x9', '9x9']
    percents = ['20%', '16%', '12%', '8%', '4%']
    algs = ['A*', 'XMM', 'X-DFBnB', 'BiX-DFBnB']

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Build the dynamic header
        header = ["Grid", "Algorithm"]
        for p in percents:
            header.extend([f"{p} Expansions", f"{p} Time [ms]"])
        writer.writerow(header)

        # Populate the rows
        for grid in grids:
            for alg in algs:
                row = [grid, alg]
                for p in percents:
                    exp, time_ms = data[grid][alg][p]
                    row.extend([exp, time_ms])
                writer.writerow(row)

if __name__ == "__main__":
    print(f"Scanning directory: {results_dir}...\n")
    
    parsed_data, bugs = parse_and_check_results(results_dir)

    # --- Bug Report Output ---
    if bugs:
        print("====== ALGORITHM BUGS DETECTED ======")
        for bug in bugs:
            print(f"[!] {bug}")
        print("=====================================\n")
    else:
        print("====== BUG CHECK PASSED ======")
        print("All algorithms reported consistent path lengths across all graphs.\n")

    # --- CSV Generation Output ---
    if parsed_data:
        write_csv(parsed_data, results_dir + "/output.csv")
        print(f"Success! Results table saved to {results_dir}/output.csv.")
    else:
        print("No valid data was found to write to the CSV. Check your directory path and file formats.")