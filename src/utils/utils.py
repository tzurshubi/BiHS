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
