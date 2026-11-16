"""Compare C++ result files with the saved Python results.

Run from the repository root:
    python3 cpp/tools/compare_results.py [cpp/results] [results]
For every CSV and JSON file below the C++ root that also exists below the
Python root, prints, per numeric column (CSV) or key (JSON), the largest
difference relative to the largest Python value of that column, and lists
non-numeric cells that differ. Timing fields are skipped.
"""
import csv
import json
import math
import sys
from pathlib import Path

TIMING = ("seconds", "timestamp", "time_", "_time", "commit")


def is_timing(name):
    return any(t in name for t in TIMING)


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, value


def column_key(path):
    """Group JSON leaves like a[3].b into column a[].b."""
    out, depth = [], 0
    for ch in path:
        if ch == "[":
            depth += 1
            out.append("[]")
        elif ch == "]":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def compare_pairs(pairs):
    columns, text_diffs = {}, []
    for key, a, b in pairs:
        if is_timing(key):
            continue
        fa, fb = number(a), number(b)
        if isinstance(a, bool) or isinstance(b, bool) or fa is None or fb is None:
            if str(a) != str(b):
                text_diffs.append((key, a, b))
            continue
        col = columns.setdefault(column_key(key), [0.0, 0.0])
        if math.isnan(fa) and math.isnan(fb):
            continue
        col[0] = max(col[0], abs(fa - fb) if math.isfinite(fa - fb) else math.inf)
        col[1] = max(col[1], abs(fb) if math.isfinite(fb) else 0.0)
    return columns, text_diffs


def csv_pairs(a_path, b_path):
    a = list(csv.DictReader(open(a_path, newline="")))
    b = list(csv.DictReader(open(b_path, newline="")))
    notes = []
    if len(a) != len(b):
        notes.append(f"row count {len(a)} (C++) vs {len(b)} (Python)")
    if a and b and list(a[0]) != list(b[0]):
        notes.append(f"columns differ: {list(a[0])} vs {list(b[0])}")
    pairs = [(k, ra.get(k), rb.get(k)) for ra, rb in zip(a, b) for k in rb]
    return pairs, notes


def json_pairs(a_path, b_path):
    a = dict(flatten(json.load(open(a_path))))
    b = dict(flatten(json.load(open(b_path))))
    notes = []
    missing = [k for k in b if k not in a and not is_timing(k)]
    extra = [k for k in a if k not in b and not is_timing(k)]
    if missing:
        notes.append(f"missing in C++: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if extra:
        notes.append(f"only in C++: {extra[:8]}{' ...' if len(extra) > 8 else ''}")
    return [(k, a.get(k), b[k]) for k in b if k in a], notes


def main(cpp_root="cpp/results", py_root="results", floor="1e-13"):
    """floor: absolute differences at or below it count as roundoff and are not listed."""
    cpp_root, py_root, floor = Path(cpp_root), Path(py_root), float(floor)
    for a_path in sorted(cpp_root.rglob("*")):
        if a_path.suffix not in (".csv", ".json") or not a_path.is_file():
            continue
        rel = a_path.relative_to(cpp_root)
        b_path = py_root / rel
        if not b_path.exists():
            print(f"\n## {rel}: no Python counterpart")
            continue
        pairs, notes = (csv_pairs if a_path.suffix == ".csv" else json_pairs)(a_path, b_path)
        columns, text_diffs = compare_pairs(pairs)
        above = {k: v for k, v in columns.items() if v[0] > floor}
        worst_abs = max((d for d, _ in columns.values()), default=0.0)
        worst = max((d / s if s > 0 else d for d, s in above.values()), default=0.0)
        print(f"\n## {rel}: {len(columns)} numeric fields, largest |diff| {worst_abs:.1e}; "
              f"above {floor:g}: {len(above)} fields, worst relative {worst:.1e}")
        for note in notes:
            print(f"   note: {note}")
        for name, (d, s) in sorted(above.items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else kv[1][0])):
            r = d / s if s > 0 else d
            if r > 1e-12:
                print(f"   {name:45s} {r:.2e}  (max |diff| {d:.2e}, max |value| {s:.2e})")
        for key, a, b in text_diffs[:10]:
            print(f"   text differs at {key}: C++ {a!r} vs Python {b!r}")
        if len(text_diffs) > 10:
            print(f"   ... {len(text_diffs) - 10} more text differences")


if __name__ == "__main__":
    main(*sys.argv[1:])
