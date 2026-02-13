#!/usr/bin/env python3
from __future__ import annotations
import os
os.system("clear")
import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

DEFAULT_FOLDER = "/home/tzur-shubi/Documents/Programming/BiHS/results/2026_02_11/sym_coil_7d_prove_no_48_simexpncheck_expand_both_frontiers_together"


SUMMARY_RE = re.compile(
    r"expansions:\s*([\d,]+)\s*,\s*time:\s*([\d,]+)\s*\[ms\]",
    re.IGNORECASE,
)

NO_PATH_RE = re.compile(r"\bno\s+path\s+found\b", re.IGNORECASE)

PATHS_FOUND_RE = re.compile(
    r"Total\s+number\s+of\s+paths\s+with\s+g\s*==\s*g_cutoff\([^)]*\)\s*found:\s*([\d,]+)",
    re.IGNORECASE,
)

VALID_MEETING_CHECKS_RE = re.compile(
    r"valid_meeting_checks'\s*:\s*([\d,]+)",
    re.IGNORECASE,
)


def parse_int_commas(s: str) -> int:
    return int(s.replace(",", "").strip())


@dataclass
class FileResult:
    path: Path
    has_no_path_found: bool
    has_sym_coil_found: bool
    expansions: Optional[int]
    time_ms: Optional[int]
    paths_found: Optional[int]
    valid_meeting_checks: Optional[int]
    summary_line: Optional[str]
    


def scan_file(p: Path) -> FileResult:
    has_no_path = False
    has_sym_coil = False      # <--- NEW TRACKER
    last_summary: Optional[Tuple[int, int, str]] = None
    last_paths_found: Optional[int] = None
    last_valid_meeting_checks: Optional[int] = None


    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not has_no_path and NO_PATH_RE.search(line):
                    has_no_path = True
                
                # <--- NEW CHECK
                if not has_sym_coil and "SYM_COIL_FOUND" in line:
                    has_sym_coil = True

                m = SUMMARY_RE.search(line)
                if m:
                    exp = parse_int_commas(m.group(1))
                    tms = parse_int_commas(m.group(2))
                    last_summary = (exp, tms, line.rstrip("\n"))

                pm = PATHS_FOUND_RE.search(line)
                if pm:
                    last_paths_found = parse_int_commas(pm.group(1))

                vm = VALID_MEETING_CHECKS_RE.search(line)
                if vm:
                    last_valid_meeting_checks = parse_int_commas(vm.group(1))


    except Exception as e:
        return FileResult(
            path=p,
            has_no_path_found=False,
            has_sym_coil_found=False,
            expansions=None,
            time_ms=None,
            paths_found=None,
            valid_meeting_checks=last_valid_meeting_checks,
            summary_line=f"ERROR reading file: {e}",
        )

    if last_summary is None:
        return FileResult(
            path=p,
            has_no_path_found=has_no_path,
            has_sym_coil_found=has_sym_coil,
            expansions=None,
            time_ms=None,
            paths_found=last_paths_found,
            valid_meeting_checks=last_valid_meeting_checks,
            summary_line=None,
        )

    exp, tms, sline = last_summary
    return FileResult(
        path=p,
        has_no_path_found=has_no_path,
        has_sym_coil_found=has_sym_coil,
        expansions=exp,
        time_ms=tms,
        paths_found=last_paths_found,
        valid_meeting_checks=last_valid_meeting_checks,
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
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def main() -> int:
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

    files: List[Path] = []
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix.lower() != ".csv":
            files.append(entry)

    results: List[FileResult] = [scan_file(p) for p in files]

    total_files = len(results)
    no_path_files = sum(1 for r in results if r.has_no_path_found)
    sym_coil_files = sum(1 for r in results if r.has_sym_coil_found) # <--- COUNTING

    exp_values = [r.expansions for r in results if r.expansions is not None]
    time_values = [r.time_ms for r in results if r.time_ms is not None]
    paths_values = [r.paths_found for r in results if r.paths_found is not None]

    exp_stats = describe(exp_values)
    time_stats = describe(time_values)
    paths_stats = describe(paths_values)

    parsed_count = sum(1 for r in results if r.expansions is not None and r.time_ms is not None)
    missing_count = total_files - parsed_count

    paths_parsed_count = sum(1 for r in results if r.paths_found is not None)
    paths_missing_count = total_files - paths_parsed_count
    total_paths_found = sum(paths_values) if paths_values else 0

    vmc_values = [r.valid_meeting_checks for r in results if r.valid_meeting_checks is not None]
    vmc_stats = describe(vmc_values)
    total_vmc = sum(vmc_values) if vmc_values else 0


    print("=== Scan summary ===")
    print(f"Folder: {root}")
    print(f"Files scanned (excluding *.csv): {total_files}")
    print(f'Files containing "no path found": {no_path_files}')
    print(f'Files containing "SYM_COIL_FOUND": {sym_coil_files}') # <--- PRINTING
    print(f"Files with parsed expansions+time: {parsed_count}")
    print(f"Files missing parsed expansions+time: {missing_count}")
    print(f"Files with parsed paths_found: {paths_parsed_count}")
    print(f"Files missing paths_found: {paths_missing_count}")
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

    print("=== Paths found stats (from 'Total number of paths ... found: N') ===")
    print(f"count : {paths_stats['count']}")
    print(f"min   : {fmt_int(paths_stats['min'])}")
    print(f"median: {fmt_int(paths_stats['median'])}")
    print(f"mean  : {fmt_int(paths_stats['mean'])}")
    print(f"max   : {fmt_int(paths_stats['max'])}")
    print(f"sum   : {fmt_int(total_paths_found)}")
    print()

    print("=== Valid meeting checks stats ===")
    print(f"count : {vmc_stats['count']}")
    print(f"min   : {fmt_int(vmc_stats['min'])}")
    print(f"median: {fmt_int(vmc_stats['median'])}")
    print(f"mean  : {fmt_int(vmc_stats['mean'])}")
    print(f"max   : {fmt_int(vmc_stats['max'])}")
    print(f"sum   : {fmt_int(total_vmc)}")
    print()

    if args.details:
        print("=== Per-file details ===")
        for r in results:
            exp_s = fmt_int(r.expansions) if r.expansions is not None else "NA"
            t_s = fmt_ms(r.time_ms) if r.time_ms is not None else "NA"
            pf_s = fmt_int(r.paths_found) if r.paths_found is not None else "NA"
            
            # Update tags to include COIL
            tags = []
            if r.has_no_path_found: tags.append("NO_PATH")
            if r.has_sym_coil_found: tags.append("COIL")
            if not tags: tags.append("OK")
            tag_str = ",".join(tags)
            
            print(f"- {r.path.name}  [{tag_str}]  expansions={exp_s}  time={t_s}  paths_found={pf_s}")
            if r.summary_line:
                print(f"  summary: {r.summary_line}")
            elif r.expansions is None or r.time_ms is None:
                print("  summary: (no matching 'expansions: ..., time: ... [ms]' line found)")
        print()

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

        print("=== Bottom 5 by time (ms) ===")
        fastest = sorted(parsed, key=lambda x: x.time_ms)[:5]
        for r in fastest:
            print(f"- {r.path.name}: {fmt_ms(r.time_ms)}  expansions={fmt_int(r.expansions)}")
        print()

        print("=== Bottom 5 by expansions ===")
        smallest = sorted(parsed, key=lambda x: x.expansions)[:5]
        for r in smallest:
            print(f"- {r.path.name}: expansions={fmt_int(r.expansions)}  time={fmt_ms(r.time_ms)}")
        print()


        print("=== 10 files closest to the mean (expansions + time) ===")
        mean_exp = exp_stats["mean"]
        mean_time = time_stats["mean"]

        # Use standard deviation to normalize (avoid time dominating expansions)
        exp_sd = statistics.pstdev(exp_values) if len(exp_values) >= 2 else 1.0
        time_sd = statistics.pstdev(time_values) if len(time_values) >= 2 else 1.0

        def dist_to_mean(r: FileResult) -> float:
            # normalized Euclidean distance in (expansions, time_ms)
            de = (r.expansions - mean_exp) / exp_sd
            dt = (r.time_ms - mean_time) / time_sd
            return (de * de + dt * dt) ** 0.5

        closest10 = sorted(parsed, key=dist_to_mean)[:10]

        print(f"mean expansions = {fmt_int(mean_exp)}")
        print(f"mean time       = {fmt_ms(mean_time)} ({fmt_s(mean_time)})")
        for r in closest10:
            d = dist_to_mean(r)
            print(
                f"- {r.path.name}: "
                f"expansions={fmt_int(r.expansions)}  "
                f"time={fmt_ms(r.time_ms)}  "
                f"dist={d:.3f}"
            )
        print()



    return 0


if __name__ == "__main__":
    raise SystemExit(main())