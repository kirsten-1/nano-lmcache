#!/usr/bin/env python3
"""
Code complexity comparison: nano-lmcache vs LMCache

Analyzes code size and structure to demonstrate nano-lmcache's simplicity.

Usage:
    python examples/code_complexity.py
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def count_lines(file_path: Path) -> Tuple[int, int, int]:
    """Count total lines, code lines, and comment lines."""
    total = 0
    code = 0
    comments = 0
    in_multiline = False

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total += 1
                stripped = line.strip()

                # Track multiline strings/comments
                if '"""' in stripped or "'''" in stripped:
                    count = stripped.count('"""') + stripped.count("'''")
                    if count == 1:
                        in_multiline = not in_multiline
                    if in_multiline or count >= 2:
                        comments += 1
                        continue

                if in_multiline:
                    comments += 1
                    continue

                if not stripped:
                    continue
                elif stripped.startswith("#"):
                    comments += 1
                else:
                    code += 1
    except Exception:
        pass

    return total, code, comments


def analyze_directory(path: Path, extensions: List[str] = [".py"]) -> Dict:
    """Analyze all Python files in a directory."""
    stats = {
        "files": 0,
        "total_lines": 0,
        "code_lines": 0,
        "comment_lines": 0,
        "by_file": {},
    }

    if not path.exists():
        return stats

    for file_path in path.rglob("*"):
        if file_path.suffix in extensions and "__pycache__" not in str(file_path):
            total, code, comments = count_lines(file_path)
            rel_path = str(file_path.relative_to(path))
            stats["files"] += 1
            stats["total_lines"] += total
            stats["code_lines"] += code
            stats["comment_lines"] += comments
            stats["by_file"][rel_path] = {
                "total": total,
                "code": code,
                "comments": comments,
            }

    return stats


def get_lmcache_stats() -> Dict:
    """Try to get LMCache stats if installed."""
    try:
        import lmcache
        lmcache_path = Path(lmcache.__file__).parent
        return analyze_directory(lmcache_path)
    except ImportError:
        # Estimate based on GitHub repo structure
        return {
            "files": "~50+",
            "code_lines": "~8000-10000",
            "note": "Estimated - install lmcache for exact count",
        }


def main():
    print("=" * 64)
    print("Code Complexity Comparison")
    print("=" * 64)
    print()

    # Find nano-lmcache
    script_dir = Path(__file__).parent
    nano_path = script_dir.parent / "nano_lmcache"

    if not nano_path.exists():
        print(f"Error: nano_lmcache not found at {nano_path}")
        sys.exit(1)

    # Analyze nano-lmcache
    print("--- nano-lmcache ---")
    nano_stats = analyze_directory(nano_path)
    print(f"  Files:         {nano_stats['files']}")
    print(f"  Total lines:   {nano_stats['total_lines']}")
    print(f"  Code lines:    {nano_stats['code_lines']}")
    print(f"  Comment lines: {nano_stats['comment_lines']}")
    print()

    print("  Breakdown by module:")
    for file, info in sorted(nano_stats["by_file"].items()):
        print(f"    {file:40s}  {info['code']:4d} lines")
    print()

    # Analyze LMCache
    print("--- LMCache ---")
    lmcache_stats = get_lmcache_stats()
    if "note" in lmcache_stats:
        print(f"  {lmcache_stats['note']}")
        print(f"  Estimated files: {lmcache_stats['files']}")
        print(f"  Estimated code:  {lmcache_stats['code_lines']}")
    else:
        print(f"  Files:         {lmcache_stats['files']}")
        print(f"  Total lines:   {lmcache_stats['total_lines']}")
        print(f"  Code lines:    {lmcache_stats['code_lines']}")
    print()

    # Summary
    print("--- Summary ---")
    nano_code = nano_stats["code_lines"]
    print(f"  nano-lmcache: {nano_code} lines of code")

    if isinstance(lmcache_stats.get("code_lines"), int):
        lmcache_code = lmcache_stats["code_lines"]
        ratio = lmcache_code / nano_code if nano_code > 0 else 0
        print(f"  LMCache:      {lmcache_code} lines of code")
        print(f"  Ratio:        LMCache is {ratio:.1f}x larger")
    else:
        print(f"  LMCache:      ~8000-10000 lines (estimated)")
        print(f"  Ratio:        LMCache is ~8-10x larger (estimated)")

    print()
    print("  Key simplifications in nano-lmcache:")
    print("    - Single-process design (no IPC complexity)")
    print("    - Radix tree instead of distributed hash")
    print("    - Variable segments instead of fixed chunks")
    print("    - Direct tensor storage (no serialization)")

    print()
    print("=" * 64)


if __name__ == "__main__":
    main()
