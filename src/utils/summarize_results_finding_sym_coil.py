#!/usr/bin/env python3
"""
Scan a folder of result logs (ignore *.csv) and report:

- total number of files scanned
- how many contain "no path found" (case-insensitive)
- stats (count/min/median/mean/max) for expansions and time_ms parsed from summary lines like:
  "... expansions: 2,980,571, time: 1,388,207 [ms] ..."

Usage:
  python scan_results.py /path/to/folder
  python scan_results.py /path/to/folder --details
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple


SUMMARY_RE = re.compile(
    r"expansions:\s*([\d,]+)\s*,\s*time:\s*([\d,]+)\s*\[ms\]",
    re.IGNORECASE,
)

NO_PATH_RE = re.compile(r"\bno\s+path\s+found\b", re.IGNORECASE)


def parse_int_commas(s: str) -> int:
    return int(s.replace(",", "").strip())


@dataclass
class FileResult:
    path: Path
    has_no_path_found: bool
    expansions: Optional[int]
    time_ms: Optional[int]
    summary_line: Optional[str]


def scan_file(p: Path) -> FileResult:
    # We parse "no path found" anywhere in the file,
    # and parse expansions/time from the LAST matching summary line (if any).
    has_no_path = False
    last_match: Optional[Tuple[int, int, str]] = None

    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not has_no_path and NO_PATH_RE.search(line):
                    has_no_path = True

                m = SUMMARY_RE.search(line)
                if m:
                    exp = parse_int_commas(m.group(1))
                    tms = parse_int_commas(m.group(2))
                    last_match = (exp, tms, line.rstrip("\n"))
    except Exception as e:
        # If a file is unreadable, treat as scanned but with missing stats.
        return FileResult(
            path=p,
            has_no_path_found=False,
            expansions=None,
            time_ms=None,
            summary_line=f"ERROR reading file: {e}",
        )

    if last_match is None:
        return FileResult(
            path=p,
            has_no_path_found=has_no_path,
            expansions=None,
            time_ms=None,
            summary_line=None,
        )

    exp, tms, sline = last_match
    return FileResult(
        path=p,
        has_no_path_found=has_no_path,
        expansions=exp,
        time_ms=tms,
        summary_line=sline,
    )


def fmt_int(n: Optional[float]) -> str:
    if n is None:
        return "NA"
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{int(n):,}"


def fmt_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "NA"
    return f"{ms:,.0f} ms"


def fmt_s(ms: Optional[float]) -> str:
    if ms is None:
        return "NA"
    return f"{ms/1000:,.2f} s"


def describe(values: List[int]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def main() -> int:
    DEFAULT_FOLDER = "/home/tzur-shubi/Documents/Programming/BiHS/results/2025_12_30"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "folder",
        nargs="?",
        default=DEFAULT_FOLDER,
        help="Folder to scan (default: %(default)s)",
    )
    ap.add_argument("--details", action="store_true", help="Print per-file parsed summary")
    args = ap.parse_args()


    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: not a folder: {root}")
        return 2

    # Collect all non-csv files (non-recursive). Change to rglob("*") if you want recursive.
    files: List[Path] = []
    for entry in sorted(root.iterdir()):
        if entry.is_file():
            if entry.suffix.lower() == ".csv":
                continue
            files.append(entry)

    results: List[FileResult] = [scan_file(p) for p in files]

    total_files = len(results)
    no_path_files = sum(1 for r in results if r.has_no_path_found)

    exp_values = [r.expansions for r in results if r.expansions is not None]
    time_values = [r.time_ms for r in results if r.time_ms is not None]

    exp_stats = describe(exp_values)
    time_stats = describe(time_values)

    # Basic counts about stats extraction
    parsed_count = sum(1 for r in results if r.expansions is not None and r.time_ms is not None)
    missing_count = total_files - parsed_count

    print("=== Scan summary ===")
    print(f"Folder: {root}")
    print(f"Files scanned (excluding *.csv): {total_files}")
    print(f'Files containing "no path found": {no_path_files}')
    print(f"Files with parsed expansions+time: {parsed_count}")
    print(f"Files missing parsed expansions+time: {missing_count}")
    print()

    print("=== Expansions stats (parsed from summary lines) ===")
    print(f"count : {exp_stats['count']}")
    print(f"min   : {fmt_int(exp_stats['min'])}")
    print(f"median: {fmt_int(exp_stats['median'])}")
    print(f"mean  : {fmt_int(exp_stats['mean'])}")
    print(f"max   : {fmt_int(exp_stats['max'])}")
    print()

    print("=== Time stats (ms) (parsed from summary lines) ===")
    print(f"count : {time_stats['count']}")
    print(f"min   : {fmt_ms(time_stats['min'])} ({fmt_s(time_stats['min'])})")
    print(f"median: {fmt_ms(time_stats['median'])} ({fmt_s(time_stats['median'])})")
    print(f"mean  : {fmt_ms(time_stats['mean'])} ({fmt_s(time_stats['mean'])})")
    print(f"max   : {fmt_ms(time_stats['max'])} ({fmt_s(time_stats['max'])})")
    print()

    if args.details:
        print("=== Per-file details ===")
        for r in results:
            exp_s = fmt_int(r.expansions) if r.expansions is not None else "NA"
            t_s = fmt_ms(r.time_ms) if r.time_ms is not None else "NA"
            tag = "NO_PATH" if r.has_no_path_found else "OK"
            print(f"- {r.path.name}  [{tag}]  expansions={exp_s}  time={t_s}")
            if r.summary_line:
                print(f"  summary: {r.summary_line}")
            elif r.expansions is None or r.time_ms is None:
                print("  summary: (no matching 'expansions: ..., time: ... [ms]' line found)")
        print()

    # Optional: show top 5 slowest / highest expansions (only if parsed)
    if parsed_count:
        parsed = [r for r in results if r.time_ms is not None and r.expansions is not None]
        slowest = sorted(parsed, key=lambda x: x.time_ms, reverse=True)[:5]
        biggest = sorted(parsed, key=lambda x: x.expansions, reverse=True)[:5]

        print("=== Top 5 by time (ms) ===")
        for r in slowest:
            print(f"- {r.path.name}: {fmt_ms(r.time_ms)}  expansions={fmt_int(r.expansions)}")
        print()

        print("=== Top 5 by expansions ===")
        for r in biggest:
            print(f"- {r.path.name}: expansions={fmt_int(r.expansions)}  time={fmt_ms(r.time_ms)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
