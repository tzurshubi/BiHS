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