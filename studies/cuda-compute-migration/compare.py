"""
Compare two benchmark JSON files produced by `bench_public_apis.py`.

Usage:
    python compare.py bench_before.json bench_after.json
    python compare.py bench_before.json bench_after.json --markdown > report.md
"""

from __future__ import annotations

import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def key(case):
    return (case["backend"], case["api"], case["dtype"],
            case["n_events"], case["avg_items"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    before = load(args.before)
    after = load(args.after)

    before_map = {key(c): c for c in before["cases"]}
    after_map = {key(c): c for c in after["cases"]}

    keys = sorted(set(before_map) | set(after_map))

    if args.markdown:
        print(f"# Benchmark comparison\n")
        print(f"- **Before**: `{before['meta'].get('label', args.before)}` "
              f"(commit `{before['meta'].get('git_sha', '?')[:8]}`)")
        print(f"- **After** : `{after['meta'].get('label', args.after)}` "
              f"(commit `{after['meta'].get('git_sha', '?')[:8]}`)")
        print(f"- Device: {after['meta'].get('device', '?')}")
        print(f"- ak={after['meta'].get('ak_version', '?')}, "
              f"cupy={after['meta'].get('cupy_version', '?')}, "
              f"cuda.compute={after['meta'].get('cuda_compute_version', '?')}\n")
        print("| backend | api | dtype | n_events | avg_items | "
              "before min (ms) | after min (ms) | speedup | notes |")
        print("|---|---|---|---:|---:|---:|---:|---:|---|")
        for k in keys:
            b = before_map.get(k)
            a = after_map.get(k)
            be = (b and b["error"]) or ""
            ae = (a and a["error"]) or ""
            backend, api, dtype, ne, ai = k
            if b and a and not be and not ae:
                speedup = b["min_ms"] / a["min_ms"] if a["min_ms"] > 0 else float("inf")
                speed_s = f"{speedup:.2f}x"
                notes = ""
            else:
                speed_s = "-"
                notes = []
                if not b: notes.append("missing in before")
                elif be: notes.append(f"before: ERR ({be[:30]}...)")
                if not a: notes.append("missing in after")
                elif ae: notes.append(f"after: ERR ({ae[:30]}...)")
                notes = "; ".join(notes)
            bm = f"{b['min_ms']:.3f}" if b and not be else "-"
            am = f"{a['min_ms']:.3f}" if a and not ae else "-"
            print(f"| {backend} | {api} | {dtype} | {ne} | {ai} | "
                  f"{bm} | {am} | {speed_s} | {notes} |")
    else:
        # Plain text table
        hdr = ("backend", "api", "dtype", "n_events", "avg_items",
               "before_min_ms", "after_min_ms", "speedup", "notes")
        print(" | ".join(f"{h:>14}" if i >= 5 else f"{h:>10}" for i, h in enumerate(hdr)))
        for k in keys:
            b = before_map.get(k)
            a = after_map.get(k)
            be = (b and b["error"]) or ""
            ae = (a and a["error"]) or ""
            backend, api, dtype, ne, ai = k
            bm = (b["min_ms"] if b and not be else None)
            am = (a["min_ms"] if a and not ae else None)
            if bm is not None and am is not None and am > 0:
                speedup = bm / am
                speed_s = f"{speedup:.2f}x"
            else:
                speed_s = "-"
            notes = []
            if be: notes.append(f"before:ERR")
            if ae: notes.append(f"after:ERR")
            if not b: notes.append("no-before")
            if not a: notes.append("no-after")
            bm_s = f"{bm:.3f}" if bm is not None else "-"
            am_s = f"{am:.3f}" if am is not None else "-"
            row = (backend, api, dtype, str(ne), str(ai),
                   bm_s, am_s, speed_s, ",".join(notes))
            print(" | ".join(f"{c:>14}" if i >= 5 else f"{c:>10}"
                              for i, c in enumerate(row)))


if __name__ == "__main__":
    sys.exit(main())
