"""
Generate a ScanNet-GT-format pairs file from a local posed_images dataset.

Layout expected under --root:
    sceneXXXX_YY/
        NNNNN.jpg            color image (1296x968 typical)
        NNNNN.png            depth map, uint16 millimeters (640x480 typical)
        NNNNN.txt            4x4 camera-to-world pose
        intrinsic.txt        4x4 color-camera intrinsic (top-left 3x3 used)
        depth_intrinsic.txt  4x4 depth-camera intrinsic

Output format (38 fields per line), consumed by runspace/src/datasets/scannet_pairs.py:
    img0 img1 rot0 rot1 K0(9) K1(9) T_0to1(16)

Two-stage:
  Stage A: per-scene, compute overlap-filtered candidate pairs via
           depth-based reprojection on a sparse grid. Cached per scene
           to <cache-dir>/<scene>__<paramhash>.npz.
  Stage B: stratified sample from cached candidates, write pairs file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageAParams:
    min_stride: int
    max_stride: int
    grid_size: int
    overlap_min: float
    overlap_max: float
    depth_min: float
    depth_max: float

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Scene IO
# ---------------------------------------------------------------------------

def _list_frames(scene_dir: Path) -> list[int]:
    frames = []
    for p in scene_dir.iterdir():
        if p.suffix.lower() == ".jpg" and p.stem.isdigit():
            frames.append(int(p.stem))
    frames.sort()
    return frames


def _load_pose(path: Path) -> np.ndarray | None:
    try:
        arr = np.loadtxt(path).reshape(4, 4)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float64)


def _load_K(path: Path) -> np.ndarray:
    return np.loadtxt(path).reshape(4, 4)[:3, :3].astype(np.float64)


def _load_depth(path: Path) -> np.ndarray:
    im = Image.open(path)
    return np.asarray(im, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------

def _grid_uv(w: int, h: int, n: int) -> np.ndarray:
    us = np.linspace(0.5, w - 0.5, n)
    vs = np.linspace(0.5, h - 0.5, n)
    uu, vv = np.meshgrid(us, vs)
    return np.stack([uu.ravel(), vv.ravel()], axis=-1)  # (N, 2)


def _unproject(uv: np.ndarray, depth_at_uv: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    """uv: (N,2), depth_at_uv: (N,), K_inv: (3,3) -> (N,3) camera-frame points."""
    ones = np.ones((uv.shape[0], 1))
    pix_h = np.concatenate([uv, ones], axis=-1)  # (N, 3)
    rays = pix_h @ K_inv.T  # (N, 3)
    return rays * depth_at_uv[:, None]


def _project(pts_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((pts_cam.shape[0], 2), np.nan)
    uv[valid] = (pts_cam[valid] @ K.T)[:, :2] / z[valid, None]
    return uv, valid


def _compute_overlap(
    depth_i_mm: np.ndarray,
    pose_i: np.ndarray,
    pose_j: np.ndarray,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    W_color: int,
    H_color: int,
    grid_size: int,
    depth_min: float,
    depth_max: float,
) -> float:
    Hd, Wd = depth_i_mm.shape
    uv = _grid_uv(Wd, Hd, grid_size)  # (N, 2) in depth image
    uv_int = np.round(uv).astype(int)
    uv_int[:, 0] = np.clip(uv_int[:, 0], 0, Wd - 1)
    uv_int[:, 1] = np.clip(uv_int[:, 1], 0, Hd - 1)
    d = depth_i_mm[uv_int[:, 1], uv_int[:, 0]].astype(np.float64) / 1000.0
    valid_d = (d > depth_min) & (d < depth_max)
    if valid_d.sum() < 4:
        return 0.0
    uv = uv[valid_d]
    d = d[valid_d]

    # Unproject in depth-camera frame (approximate color==depth extrinsic as identity)
    K_depth_inv = np.linalg.inv(K_depth)
    pts_cam_i = _unproject(uv, d, K_depth_inv)  # (M, 3)

    # To world, then to j camera
    ones = np.ones((pts_cam_i.shape[0], 1))
    pts_h = np.concatenate([pts_cam_i, ones], axis=-1)  # (M, 4)
    pts_w = pts_h @ pose_i.T  # (M, 4)
    pts_j = pts_w @ np.linalg.inv(pose_j).T  # (M, 4)
    pts_j = pts_j[:, :3]

    uv_j, z_valid = _project(pts_j, K_color)
    in_bounds = (
        z_valid
        & (uv_j[:, 0] >= 0) & (uv_j[:, 0] < W_color)
        & (uv_j[:, 1] >= 0) & (uv_j[:, 1] < H_color)
    )
    return float(in_bounds.sum()) / float(len(d))


# ---------------------------------------------------------------------------
# Per-scene stage A
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, scene: str, params: StageAParams) -> Path:
    return cache_dir / f"{scene}__{params.hash()}.npz"


def process_scene(args: tuple) -> tuple[str, int, int, str]:
    """Stage A for one scene. Returns (scene, candidates_kept, frames_examined, status)."""
    scene_dir, cache_file, params, color_size = args
    scene_dir = Path(scene_dir)
    cache_file = Path(cache_file)

    if cache_file.exists():
        return (scene_dir.name, -1, -1, "cached")

    try:
        K_color = _load_K(scene_dir / "intrinsic.txt")
        K_depth = _load_K(scene_dir / "depth_intrinsic.txt")
    except Exception as e:
        return (scene_dir.name, 0, 0, f"error: {e}")

    frames = _list_frames(scene_dir)
    poses: dict[int, np.ndarray] = {}
    for f in frames:
        p = _load_pose(scene_dir / f"{f:05d}.txt")
        if p is not None:
            poses[f] = p

    valid_frames = sorted(poses.keys())
    if len(valid_frames) < 2:
        np.savez(cache_file, frame_i=np.array([], dtype=np.int32),
                 frame_j=np.array([], dtype=np.int32),
                 overlap=np.array([], dtype=np.float32))
        return (scene_dir.name, 0, len(valid_frames), "ok")

    W_color, H_color = color_size

    # Cache depth reads: for each i we load its depth once and test all j's
    kept_i: list[int] = []
    kept_j: list[int] = []
    kept_o: list[float] = []

    frame_to_idx = {f: k for k, f in enumerate(valid_frames)}

    for i_idx, i in enumerate(valid_frames):
        js = [f for f in valid_frames[i_idx + 1 :]
              if params.min_stride <= (f - i) <= params.max_stride]
        if not js:
            continue
        try:
            depth_i = _load_depth(scene_dir / f"{i:05d}.png")
        except Exception:
            continue

        for j in js:
            ov = _compute_overlap(
                depth_i, poses[i], poses[j], K_depth, K_color,
                W_color, H_color, params.grid_size,
                params.depth_min, params.depth_max,
            )
            if params.overlap_min <= ov <= params.overlap_max:
                kept_i.append(i)
                kept_j.append(j)
                kept_o.append(ov)

    np.savez(
        cache_file,
        frame_i=np.asarray(kept_i, dtype=np.int32),
        frame_j=np.asarray(kept_j, dtype=np.int32),
        overlap=np.asarray(kept_o, dtype=np.float32),
    )
    return (scene_dir.name, len(kept_i), len(valid_frames), "ok")


# ---------------------------------------------------------------------------
# Stage B: assemble pairs file
# ---------------------------------------------------------------------------

def _stratified_sample(
    frame_i: np.ndarray, frame_j: np.ndarray, overlap: np.ndarray,
    per_scene: int, rng: random.Random,
) -> list[int]:
    if len(overlap) == 0:
        return []
    if per_scene <= 0 or len(overlap) <= per_scene:
        return list(range(len(overlap)))

    buckets: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for idx, o in enumerate(overlap):
        if o < 0.4:
            buckets[0].append(idx)
        elif o < 0.6:
            buckets[1].append(idx)
        else:
            buckets[2].append(idx)

    for b in buckets.values():
        rng.shuffle(b)

    chosen: list[int] = []
    # Round-robin across buckets to preserve variety
    quotas = [per_scene // 3] * 3
    for r in range(per_scene - sum(quotas)):
        quotas[r] += 1
    remaining = per_scene
    for _ in range(per_scene):
        progressed = False
        for b_idx in (0, 1, 2):
            if remaining == 0:
                break
            if quotas[b_idx] > 0 and buckets[b_idx]:
                chosen.append(buckets[b_idx].pop())
                quotas[b_idx] -= 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    # If under quota (one bucket empty), fill from others
    if remaining > 0:
        leftovers = [i for b in buckets.values() for i in b]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:remaining])

    return chosen


def _fmt_floats(arr: np.ndarray) -> str:
    return " ".join(f"{x:.6f}" for x in arr.ravel())


def assemble_pairs(
    root: Path, cache_dir: Path, scenes: list[str], params: StageAParams,
    pairs_per_scene: int, max_total: int, seed: int, out_path: Path,
) -> int:
    rng = random.Random(seed)
    lines: list[str] = []

    for scene in sorted(scenes):
        cf = _cache_path(cache_dir, scene, params)
        if not cf.exists():
            continue
        data = np.load(cf)
        fi = data["frame_i"]
        fj = data["frame_j"]
        ov = data["overlap"]
        if len(fi) == 0:
            continue

        scene_dir = root / scene
        try:
            K = _load_K(scene_dir / "intrinsic.txt")
        except Exception:
            continue
        K_flat = _fmt_floats(K)

        picks = _stratified_sample(fi, fj, ov, pairs_per_scene, rng)

        for idx in picks:
            i, j = int(fi[idx]), int(fj[idx])
            pose_i = _load_pose(scene_dir / f"{i:05d}.txt")
            pose_j = _load_pose(scene_dir / f"{j:05d}.txt")
            if pose_i is None or pose_j is None:
                continue
            T_0to1 = np.linalg.inv(pose_j) @ pose_i
            img0 = f"{scene}/{i:05d}.jpg"
            img1 = f"{scene}/{j:05d}.jpg"
            line = (
                f"{img0} {img1} 0 0 "
                f"{K_flat} {K_flat} "
                f"{_fmt_floats(T_0to1)}"
            )
            lines.append(line)

    rng.shuffle(lines)
    if max_total > 0:
        lines = lines[:max_total]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    return len(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _discover_scenes(root: Path, scenes_file: Path | None) -> list[str]:
    if scenes_file is not None:
        with open(scenes_file) as f:
            wanted = [ln.strip() for ln in f if ln.strip()]
        return [s for s in wanted if (root / s).is_dir()]
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--scenes-file", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Default: <root>/../.overlap_cache")
    ap.add_argument("--min-stride", type=int, default=10)
    ap.add_argument("--max-stride", type=int, default=120)
    ap.add_argument("--grid-size", type=int, default=16)
    ap.add_argument("--overlap-min", type=float, default=0.2)
    ap.add_argument("--overlap-max", type=float, default=0.8)
    ap.add_argument("--depth-min", type=float, default=0.1)
    ap.add_argument("--depth-max", type=float, default=10.0)
    ap.add_argument("--color-width", type=int, default=1296)
    ap.add_argument("--color-height", type=int, default=968)
    ap.add_argument("--pairs-per-scene", type=int, default=5)
    ap.add_argument("--max-total", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(list(argv) if argv is not None else None)

    root: Path = args.root.resolve()
    cache_dir: Path = (args.cache_dir or (root.parent / ".overlap_cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    params = StageAParams(
        min_stride=args.min_stride,
        max_stride=args.max_stride,
        grid_size=args.grid_size,
        overlap_min=args.overlap_min,
        overlap_max=args.overlap_max,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )

    scenes = _discover_scenes(root, args.scenes_file)
    print(f"[stage A] scenes={len(scenes)} param-hash={params.hash()} cache={cache_dir}")

    if args.rebuild_cache:
        for s in scenes:
            cf = _cache_path(cache_dir, s, params)
            if cf.exists():
                cf.unlink()

    tasks = [
        (str(root / s), str(_cache_path(cache_dir, s, params)),
         params, (args.color_width, args.color_height))
        for s in scenes
    ]

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(process_scene, tasks), 1):
                results.append(res)
                if i % 10 == 0 or i == len(tasks):
                    print(f"[stage A] {i}/{len(tasks)}  last={res}")
    else:
        results = []
        for i, t in enumerate(tasks, 1):
            res = process_scene(t)
            results.append(res)
            if i % 10 == 0 or i == len(tasks):
                print(f"[stage A] {i}/{len(tasks)}  last={res}")

    cached = sum(1 for r in results if r[3] == "cached")
    ok = sum(1 for r in results if r[3] == "ok")
    errs = [r for r in results if r[3].startswith("error")]
    total_kept = sum(r[1] for r in results if r[1] >= 0)
    print(f"[stage A] done. cached={cached} computed={ok} errors={len(errs)} "
          f"new_candidates={total_kept}")
    for r in errs[:5]:
        print(f"  err: {r[0]}: {r[3]}")

    print(f"[stage B] assembling pairs file -> {args.out}")
    n = assemble_pairs(
        root=root, cache_dir=cache_dir, scenes=scenes, params=params,
        pairs_per_scene=args.pairs_per_scene, max_total=args.max_total,
        seed=args.seed, out_path=args.out.resolve(),
    )
    print(f"[stage B] wrote {n} pairs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
