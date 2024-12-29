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


def calculate_averages(avgs):
    """
    Calculate and print the averages for each metric across categories.

    Parameters:
    avgs (dict): A dictionary where keys are categories (e.g., "uni_st", "uni_ts", "bi"),
                 and values are dictionaries with metrics (e.g., "expansions", "time") as keys and lists of numbers as values.
    """
    metrics = list(next(iter(avgs.values())).keys())  # Get metric names from the first category

    for metric in metrics:
        averages = []
        for category in avgs:
            values = avgs[category][metric]
            avg = round(sum(values) / len(values)) if values else 0
            averages.append(avg)
        formatted_averages = ", ".join(f"{avg}" for avg in averages)
        print(f"average {metric}: ({formatted_averages})")

    # Calculate and print average expansions per second
    expansions_per_second = []
    for category in avgs:
        expansions = avgs[category]["expansions"]
        times = avgs[category]["time"]
        avg_expansions_per_sec = round(sum(expansions) / (sum(times) / 1000)) if times else 0
        expansions_per_second.append(avg_expansions_per_sec)
    formatted_expansions_per_second = ", ".join(f"{eps}" for eps in expansions_per_second)
    print(f"average expansions per second: ({formatted_expansions_per_second})")