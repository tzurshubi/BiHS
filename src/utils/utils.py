import networkx as nx
import json
import random
import numpy as np
import matplotlib.patches as patches
import os, math
import matplotlib.pyplot as plt
import re
import csv

def ms2str(time_ms):
    """
    Convert elapsed time in milliseconds to a formatted string: 
    - DD:HH:MM:SS (omit DD if 0, omit HH if 0)
    - Always include MM:SS
    
    Args:
        time_ms (float): Elapsed time in milliseconds.
    
    Returns:
        str: A string representing the formatted elapsed time.
    """
    # Convert from milliseconds to seconds
    elapsed_time = time_ms / 1000.0

    # Convert to DD:HH:MM:SS
    days = int(elapsed_time // 86400)
    hours = int((elapsed_time % 86400) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Build the formatted string
    if days > 0:
        return f"{days}:{hours:02}:{minutes:02}:{seconds:02}"
    elif hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"

def time2str(start_time, end_time):
    """
    Calculate elapsed time formatted as DD:HH:MM:SS, but remove days if 00 
    and hours if 00. Always include MM:SS.

    Args:
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        str: A string representing the formatted elapsed time.
    """
    elapsed_time = end_time - start_time

    # Convert to DD:HH:MM:SS
    days = int(elapsed_time // 86400)
    hours = int((elapsed_time % 86400) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Build the formatted string
    if days > 0:
        return f"{days}:{hours:02}:{minutes:02}:{seconds:02}"
    elif hours > 0:
        return f"{hours}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"

def time_ms(start_time, end_time):
    """
    Calculate elapsed time formatted as DD:HH:MM:SS, but remove days if 00 
    and hours if 00. Always include MM:SS.

    Args:
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        str: A string representing the formatted elapsed time.
    """
    elapsed_time = end_time - start_time

    return int(1000*elapsed_time)
    
def calculate_averages(avgs, log_file_name=None):
    """
    Calculate and print the averages for each metric across categories.

    Parameters:
    avgs (dict): A dictionary where keys are categories (e.g., "uni_st", "uni_ts", "bi"),
                 and values are dictionaries with metrics (e.g., "expansions", "time") as keys and lists of numbers as values.
    """
    metrics = list(next(iter(avgs.values())).keys())  # Get metric names from the first category
    averages_per_metric = {}

    for metric in metrics:
        avgs_list = []
        for category in avgs:
            values = avgs[category].get(metric, [])
            avg = round(sum(values) / len(values)) if values else 0
            avgs_list.append(avg)
        averages_per_metric[metric] = avgs_list

        formatted = ", ".join(str(x) for x in avgs_list)
        line = f"average {metric}: ({formatted})"
        print(line)
        if log_file_name:
            with open(log_file_name, 'a') as f:
                f.write(line + "\n")
                

    # Calculate and print average expansions per second
    expansions_per_second = []
    for category in avgs:
        expansions = avgs[category]["expansions"]
        times = avgs[category]["time"]
        avg_expansions_per_sec = round(sum(expansions) / (sum(times) / 1000)) if times else 0
        expansions_per_second.append(avg_expansions_per_sec)
    formatted_expansions_per_second = ", ".join(f"{eps}" for eps in expansions_per_second)
    # print(f"average expansions per second: ({formatted_expansions_per_second})")
    # if log_file_name:
    #     with open(log_file_name, 'a') as file:
    #         file.write(f"average expansions per second: ({formatted_expansions_per_second})")

    # Summary two lines
    exp_avgs = averages_per_metric.get("expansions", [])
    time_avgs = averages_per_metric.get("time", [])
    if exp_avgs[1]==0: exp_avgs[1]=exp_avgs[0]
    if time_avgs[1]==0: time_avgs[1]=time_avgs[0]
    if len(exp_avgs) >= 2 and len(time_avgs) >= 2:
        first_min_exp = min(exp_avgs[0], exp_avgs[1])
        first_min_time = min(time_avgs[0], time_avgs[1])
        last_exp = exp_avgs[-1]
        last_time = time_avgs[-1]

        line1 = f"A*: {first_min_exp} , {first_min_time} (expansions , time[ms])"
        line2 = f"XMM: {last_exp} , {last_time} (expansions , time[ms])"
        print()
        print(line1)
        print(line2)
        if log_file_name:
            with open(log_file_name, 'a') as f:
                f.write("\n" + line1 + "\n")
                f.write(line2 + "\n")

# Coils utilities

def node_num_to_bits_on(dim, numbers):
    """
    Convert a list of node numbers to their corresponding bit representations.

    Parameters
    ----------
    dim : int
        The dimension of the cube (number of bits).
    numbers : list of int
        List of node numbers to convert.

    Returns
    -------
    list of list of int
        A list where each element is a list of bit positions that are '1' in the binary representation of the corresponding node number.
    """
    bits_on_list = []
    for n in numbers:
        bits_on = [i for i, b in enumerate(f"{n:0{dim}b}"[::-1], start=1) if b == '1']
        bits_on_list.append(bits_on)
    return bits_on_list

def print_bit_statistics(l):
    # Find the maximum index that appears
    max_index = max((max(sublist) for sublist in l if sublist), default=0)

    for i in range(1, max_index + 1):
        count_on = sum(1 for sublist in l if i in sublist)
        count_off = len(l) - count_on
        print(f"bit {i} on: {count_on} times, bit {i} off: {count_off} times")

def print_bits_pattern(bit_lists):
    for bits in bit_lists:
        if len(bits) == 0:
            print("X")
            continue
        line = " "
        for i in range(1, max(max(sub) if sub else 0 for sub in bit_lists) + 1):
            line += str(i) if i in bits else " "
        print(line.rstrip())  # remove trailing spaces for neatness
    print("X")  # bottom boundary

def bits_to_moves(bit_lists):
    # Convert list of bits turned on to list of moves (bit changes)
    moves = []
    for prev, curr in zip(bit_lists, bit_lists[1:]):
        # XOR logic: the changed bit is the one that’s in one list but not the other
        diff = set(curr) ^ set(prev)
        if len(diff) == 1:
            moves.append(next(iter(diff)))  # extract the single changed bit
        else:
            moves.append(None)  # if something’s wrong (shouldn’t happen in valid coils)
    if len(bit_lists[-1]) > 1:
        print("Warning: Last element has multiple bits on; cannot determine move.")
    return moves + bit_lists[-1]

def coil_dim_crossed_to_vertices(coil_str):
    """
    Convert a coil string into a list of vertex integers.
    coil_str: a string where each char is a digit '0'..'6' (or higher for bigger cubes)
    Returns: list of integer vertex values, starting with vertex 0.
    """

    vertices = [0]               # start at vertex 0
    current = 0

    for ch in coil_str:
        d = int(ch)             # dimension to flip
        bit = 1 << d            # value of that bit

        # flip the bit
        current ^= bit          # XOR toggles the bit
        vertices.append(current)

    return vertices


# Function to display the graph
def display_graph(graph, title="Graph", filename=None):
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=500,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# Function to parse results file and write to CSV
def parse_results_file(input_file, output_file):
    """
    Parse a results .txt file and write a structured .csv file.

    Parameters
    ----------
    input_file : str
        Path to the input .txt file with experiment results.
    output_file : str
        Path where the .csv file will be written.
    """

    # Regex patterns
    uni_pattern = re.compile(r"! unidirectional (s-t|t-s)\. expansions: ([\d,]+), time: ([\d,]+) \[ms\]")
    bi_pattern = re.compile(r"! bidirectional\. expansions: ([\d,]+), time: ([\d,]+) \[ms\]")

    rows = []
    grid_index = 0  # increments when we hit a new "SM_Grids/..." header

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # # Detect grid header
            # if line.startswith("SM_Grids/"):
            #     grid_index += 1
            #     continue

            # Match unidirectional
            m_uni = uni_pattern.search(line)
            if m_uni:
                direction, expansions, time = m_uni.groups()
                direction = "s -> t" if direction == "s-t" else "t -> s"
                rows.append([grid_index, direction, expansions, time])
                continue

            # Match bidirectional (XMM)
            m_bi = bi_pattern.search(line)
            if m_bi:
                expansions, time = m_bi.groups()
                rows.append([grid_index, "XMM", expansions, time])
                continue

    # Write to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["grid number", "Direction", "Expansions", "Time [ms]"])
        writer.writerows(rows)