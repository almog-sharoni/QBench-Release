#!/usr/bin/env python3
"""Emit the ordered scene list referenced by the SuperGlue paper pairs file.

Scenes are deduplicated and kept in first-seen order, matching the scene order
used by extract_paper_frames.sh and make_paper_pairs.sh. Optionally filters to
scenes not already present on disk (useful for feeding download_scannet_test.sh).
"""
import argparse
import os
import sys

DEFAULT_PAIRS = "/home/spark1/yarden/QBench2/QBench-Release/custom/SuperGluePretrainedNetwork/assets/scannet_test_pairs_with_gt.txt"
DEFAULT_ROOT = "/data/scannet/scans_test"


def ordered_scenes(pairs_file):
    order, seen = [], set()
    with open(pairs_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            for p in (parts[0], parts[1]):
                s = p.split("/")[1]
                if s not in seen:
                    seen.add(s)
                    order.append(s)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="scans_test root; used with --missing-only")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep first N scenes (0 = all)")
    ap.add_argument("--missing-only", action="store_true",
                    help="emit only scenes whose <scene>.sens is absent under --root")
    ap.add_argument("--out", default="-",
                    help="output path (default stdout)")
    args = ap.parse_args()

    scenes = ordered_scenes(args.pairs)
    if args.limit > 0:
        scenes = scenes[:args.limit]
    if args.missing_only:
        scenes = [s for s in scenes
                  if not os.path.isfile(os.path.join(args.root, s, f"{s}.sens"))]

    body = "\n".join(scenes) + ("\n" if scenes else "")
    if args.out == "-":
        sys.stdout.write(body)
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as g:
            g.write(body)
        print(f"wrote {len(scenes)} scenes -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
