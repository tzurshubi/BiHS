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
base_dir = "/home/tzur-shubi/Documents/Programming/BiHS/results/2026_05_10"

ALGS = ['A*', 'XMM', 'XDFBnB', 'BiXDFBnB', 'XIDA', 'BiXIDA']

DOMAIN_ORDER = [
    ("grid",  "LSP Grids"),
    ("snake", "Snake Grids"),
    ("maze",  "LSP Mazes"),
    ("cib",   "CIB"),
]

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

def get_lookahead(directory):
    for part in Path(directory).parts:
        m = re.match(r'(\d+)_lookahead', part)
        if m:
            return int(m.group(1))
    return None

def get_run_algorithm(directory):
    """Return the run-algorithm folder name (IDA, DFBnB, XMM, A) from the path, or None."""
    for part in Path(directory).parts:
        if part in ("IDA", "DFBnB", "XMM", "A"):
            return part
    return None

def parse_timestamp_ms(ts_str):
    """Parse [d:h:m:s:ms] timestamp string to total milliseconds."""
    m = re.match(r'\[(\d+):(\d+):(\d+):(\d+):(\d+)\]', ts_str)
    if m:
        d, h, mn, s, ms = (int(x) for x in m.groups())
        return d * 86400000 + h * 3600000 + mn * 60000 + s * 1000 + ms
    return 0

def parse_and_check_results(directory):
    bug_reports = []

    # data[row_key][algorithm][col_key] = (expansions, time_ms)
    # For CIB: col_key is "find" or "prove". For others: col_key is percent/blocked variant.
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
            row_key = match.group(1)
            col_key = match.group(2) + "blocked"
        elif dir_type == "cib":
            match = re.search(r"results_(\d+)d_cube_", filename)
            if not match:
                continue
            row_key = match.group(1)
        else:
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

        # CIB-specific state for finding stats
        if dir_type == "cib":
            find_acc = defaultdict(list)  # alg -> [(exp, time_ms), ...]
            last_finding = None    # (exp, time_ms) from last "New longest path found"
            pending_finding = None  # saved at "Path of length L found"
            uni_alg = "XDFBnB" if "DFBnB" in filename else "XIDA"
            bi_alg  = "BiXDFBnB" if "DFBnB" in filename else "BiXIDA"

        for line in lines:
            # 1. Identify the current graph block
            header_match = re.search(r"^----------\s*(.*?)\s*----------", line)
            if header_match:
                current_graph_id = header_match.group(1)
                if dir_type == "cib":
                    last_finding = None
                    pending_finding = None
                continue

            # 2. Extract path lengths for the BUG CHECKER
            length_match = re.search(r"path length:\s*([\d,]+)\s*\[edges\]", line)
            if length_match and current_graph_id:
                val = int(length_match.group(1).replace(',', ''))
                lengths_for_graph[current_graph_id].append(val)

            if dir_type == "cib":
                # 3a. Track finding stats from "New longest path found" lines
                new_path_match = re.search(
                    r'(\[\d+:\d+:\d+:\d+:\d+\]) Expansion ([\d,]+): New longest path found', line)
                if new_path_match:
                    ts_ms = parse_timestamp_ms(new_path_match.group(1))
                    exp = int(new_path_match.group(2).replace(',', ''))
                    last_finding = (exp, ts_ms)

                # 3b. "Path of length L found" — lock in the finding for whoever just ran
                if re.search(r'Path of length \d+ found:', line):
                    pending_finding = last_finding
                    last_finding = None

                # 3c. "! Unidirectional s-t." → attribute pending finding to the uni algorithm
                if re.search(r'! Unidirectional s-t\.', line) and pending_finding:
                    find_acc[uni_alg].append(pending_finding)
                    pending_finding = None

                # 3d. "! Bidirectional." → attribute pending finding to the bi algorithm
                if re.search(r'! Bidirectional\.', line) and pending_finding:
                    find_acc[bi_alg].append(pending_finding)
                    pending_finding = None

            # 4. Extract final summary stats
            summary_match = re.search(
                r"(A\*|XMM|XDFBnB|BiXDFBnB|XIDA|BiXIDA):\s*([\d,]+)\s*,\s*([\d,]+)\s*\(expansions", line)
            if summary_match:
                alg = summary_match.group(1)
                expansions = summary_match.group(2).replace(',', '')
                time_ms = summary_match.group(3).replace(',', '')
                if dir_type == "cib":
                    data[row_key][alg]["prove"] = (expansions, time_ms)
                else:
                    data[row_key][alg][col_key] = (expansions, time_ms)

        # Bug checker
        for graph_id, lengths in lengths_for_graph.items():
            if len(set(lengths)) > 1:
                bug_reports.append(
                    f"BUG in {filename} -> Graph '{graph_id}': Conflicting lengths found {lengths}")

        # CIB: average finding stats across all graph instances in this file
        if dir_type == "cib":
            for alg, findings in find_acc.items():
                if findings:
                    avg_exp = round(sum(e for e, _ in findings) / len(findings))
                    avg_time = round(sum(t for _, t in findings) / len(findings))
                    data[row_key][alg]["find"] = (str(avg_exp), str(avg_time))

    return data, bug_reports

def write_csv(data, directory):
    dir_type = get_dir_type(directory)
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
                for alg in ALGS:
                    cells = []
                    for key, _ in variants:
                        cells.extend(data[grid][alg][key])
                    if any(cells):
                        writer.writerow([grid, alg] + cells)

        elif dir_type == "cib":
            writer.writerow(["Dimension", "Algorithm",
                             "Finding Expansions", "Finding Time [ms]",
                             "Proving Expansions", "Proving Time [ms]"])

            for dim in ['4', '5', '6', '7']:
                for alg in ALGS:
                    cells = list(data[dim][alg]["find"]) + list(data[dim][alg]["prove"])
                    if any(cells):
                        writer.writerow([dim, alg] + cells)

        else:
            grids = ['7x7', '7x8', '7x9', '8x8', '8x9', '9x9'] if dir_type == "snake" else ['6x6', '6x7', '6x8', '7x7', '7x8', '8x8']
            percents = ['20%', '16%', '12%', '8%'] #, '4%']

            header = ["Grid", "Algorithm"]
            for p in percents:
                header.extend([f"{p} Expansions", f"{p} Time [ms]"])
            writer.writerow(header)

            for grid in grids:
                for alg in ALGS:
                    cells = []
                    for p in percents:
                        cells.extend(data[grid][alg][p])
                    if any(cells):
                        writer.writerow([grid, alg] + cells)

def _write_domain_table(writer, dir_type, type_data, first_table, domain_name):
    """Write one domain table into an open csv writer. Returns updated first_table flag."""
    sorted_lookaheads = sorted(type_data.keys())

    if not first_table:
        writer.writerow([])
    writer.writerow([f"--- {domain_name} ---"])

    if dir_type == "maze":
        variants = [("0blocked", "0 blocks"), ("1blocked", "1 block"), ("2blocked", "2 blocks")]
        header = ["Grid", "Algorithm", "Lookahead"]
        for _, label in variants:
            header.extend([f"{label} Expansions", f"{label} Time [ms]"])
        writer.writerow(header)

        for grid in ['13x13']:
            for alg in ALGS:
                for la in sorted_lookaheads:
                    cells = []
                    for key, _ in variants:
                        cells.extend(type_data[la][grid][alg][key])
                    if any(cells):
                        writer.writerow([grid, alg, la] + cells)

    elif dir_type == "cib":
        writer.writerow(["Dimension", "Algorithm", "Lookahead",
                         "Finding Expansions", "Finding Time [ms]",
                         "Proving Expansions", "Proving Time [ms]"])

        for dim in ['4', '5', '6', '7']:
            for alg in ALGS:
                for la in sorted_lookaheads:
                    cells = list(type_data[la][dim][alg]["find"]) + list(type_data[la][dim][alg]["prove"])
                    if any(cells):
                        writer.writerow([dim, alg, la] + cells)

    else:
        grids = ['7x7', '7x8', '7x9', '8x8', '8x9', '9x9'] if dir_type == "snake" else ['6x6', '6x7', '6x8', '7x7', '7x8', '8x8']
        percents = ['20%', '16%', '12%', '8%'] #, '4%']

        header = ["Grid", "Algorithm", "Lookahead"]
        for p in percents:
            header.extend([f"{p} Expansions", f"{p} Time [ms]"])
        writer.writerow(header)

        for grid in grids:
            for alg in ALGS:
                for la in sorted_lookaheads:
                    cells = []
                    for p in percents:
                        cells.extend(type_data[la][grid][alg][p])
                    if any(cells):
                        writer.writerow([grid, alg, la] + cells)

    return False  # first_table is now False after the first write

def write_combined_csv(all_data, output_path):
    """Write 4 domain tables (one per domain) into one CSV, merging all run algorithms.
    Column order: primary key | Algorithm | Lookahead | data...
    Empty rows are skipped.
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        first_table = True
        for dir_type, domain_name in DOMAIN_ORDER:
            if dir_type not in all_data:
                continue
            first_table = _write_domain_table(writer, dir_type, all_data[dir_type], first_table, domain_name)

if __name__ == "__main__":
    leaf_dirs = get_leaf_dirs(base_dir)

    if not leaf_dirs:
        print(f"No leaf subdirectories found in '{base_dir}'.")

    # all_data[dir_type][lookahead] = merged parsed_data (across all run algorithms)
    all_data = defaultdict(dict)

    for directory in leaf_dirs:
        print(f"\n{'='*60}")
        print(f"Scanning: {directory}\n")

        parsed_data, bugs = parse_and_check_results(directory)

        dir_type = get_dir_type(directory)
        lookahead = get_lookahead(directory)
        if lookahead is not None:
            if lookahead not in all_data[dir_type]:
                all_data[dir_type][lookahead] = defaultdict(
                    lambda: defaultdict(lambda: defaultdict(lambda: ("", ""))))
            target = all_data[dir_type][lookahead]
            for row_key in parsed_data:
                for alg in parsed_data[row_key]:
                    for col_key, val in parsed_data[row_key][alg].items():
                        if val != ("", ""):
                            target[row_key][alg][col_key] = val

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

    if all_data:
        combined_path = os.path.join(base_dir, "bigoutput.csv")
        write_combined_csv(all_data, combined_path)
        print(f"\nCombined results saved to {combined_path}.")
