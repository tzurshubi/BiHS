import os
import re
import csv
from collections import defaultdict

# Set the directory containing the txt files
directory = "/home/tzur-shubi/Documents/Programming/BiHS/results/2025_07_25"

# Regex patterns
result_pattern = re.compile(r"(A\*|XMM):\s*(\d+)\s*,\s*(\d+)\s*\(expansions\s*,\s*time\[ms\]\)")
size_pattern = re.compile(r"\s*(\d+)\s*x\s*(\d+)", re.IGNORECASE)
blocks_pattern = re.compile(r"_\s*(\d+)per", re.IGNORECASE)

# Data structure: {grid_size: {block_percent: {"A*": (exp, time), "XMM": (exp, time)}}}
data = defaultdict(lambda: defaultdict(dict))
block_percents = set()
grid_sizes = set()

for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), "r") as f:
        lines = f.readlines()
    
    # Find grid size and block percent in the file
    grid_size = None
    block_percent = None
    size_match = size_pattern.search(filename)
    if size_match:
        grid_size = f"{size_match.group(1)}x{size_match.group(2)}"
    blocks_match = blocks_pattern.search(filename)
    if blocks_match:
        block_percent = f"{blocks_match.group(1)}%"
    
    if not grid_size or not block_percent:
        continue
    
    grid_sizes.add(grid_size)
    block_percents.add(block_percent)
    
    # Get last 10 lines for results
    for line in lines[-10:]:
        result_match = result_pattern.match(line.strip())
        if result_match:
            algo, expansions, time = result_match.groups()
            data[grid_size][block_percent][algo] = (expansions, time)

# Sort for consistent CSV output
block_percents = sorted(block_percents, key=lambda x: int(x.rstrip('%')), reverse=True)
grid_sizes = sorted(grid_sizes, key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))

# Prepare CSV header
header = ["Grid Size", "Algorithm"]
for bp in block_percents:
    header.extend([f"{bp} Expansion", f"{bp} Time"])

# Write to CSV
with open(directory+"/table_of_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for grid_size in grid_sizes:
        # Collect A* and XMM row values for ratio calculation
        astar_row = [grid_size, "A*"]
        xmm_row = [grid_size, "XMM"]
        ratio_row = [grid_size, "A*/XMM"]
        astar_vals = []
        xmm_vals = []
        for bp in block_percents:
            if "A*" in data[grid_size][bp]:
                expansions, time = data[grid_size][bp]["A*"]
                astar_row.extend([expansions, time])
                astar_vals.append((expansions, time))
            else:
                astar_row.extend(["", ""])
                astar_vals.append((None, None))
            if "XMM" in data[grid_size][bp]:
                expansions, time = data[grid_size][bp]["XMM"]
                xmm_row.extend([expansions, time])
                xmm_vals.append((expansions, time))
            else:
                xmm_row.extend(["", ""])
                xmm_vals.append((None, None))
        # Calculate ratios
        for (a_exp, a_time), (x_exp, x_time) in zip(astar_vals, xmm_vals):
            # Expansion ratio
            try:
                ratio_exp = float(a_exp) / float(x_exp) if a_exp and x_exp and float(x_exp) != 0 else ""
            except Exception:
                ratio_exp = ""
            # Time ratio
            try:
                ratio_time = float(a_time) / float(x_time) if a_time and x_time and float(x_time) != 0 else ""
            except Exception:
                ratio_time = ""
            ratio_row.extend([f"{ratio_exp:.2f}" if ratio_exp != "" else "", f"{ratio_time:.2f}" if ratio_time != "" else ""])
        writer.writerow(astar_row)
        writer.writerow(xmm_row)
        writer.writerow(ratio_row)