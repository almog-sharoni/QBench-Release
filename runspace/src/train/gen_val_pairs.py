"""Generate a disjoint ScanNet validation pairs file (38-field format).

Output: runspace/inputs/scannet/pairs_val_disjoint_1500.txt

The Phase 2 / V2 training run accidentally used the same paper-100 test set for
both per-epoch validation and final test, making "best epoch" selection a
selection-on-test (see memory: project_qbench_design_c_val_test_overlap.md).
This script builds a clean validation set drawn from the official ScanNet val
scenes (val_scenes_all.txt), with no scene overlap with pairs_train.txt or
pairs_test_paper_100scenes.txt.

Pair format (38 fields, mirrors pairs_train.txt):
    img0 img1 0 0 K0(9) K1(9) T_0to1(16)
where img paths are `<scene>/<frame>.jpg` relative to /data/scannet/posed_images.

Run inside qbench container with no CLI args:
    docker exec qbench python runspace/src/train/gen_val_pairs.py
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/app")
INPUTS_DIR = REPO_ROOT / "runspace" / "inputs" / "scannet"
SCANNET_ROOT = Path("/data/scannet/posed_images")

VAL_SCENES_LIST = INPUTS_DIR / "val_scenes_all.txt"
TRAIN_PAIRS = INPUTS_DIR / "pairs_train.txt"
TEST_PAIRS = INPUTS_DIR / "pairs_test_paper_100scenes.txt"
OUT_FILE = INPUTS_DIR / "pairs_val_disjoint_1500.txt"

LOG_DIR = REPO_ROOT / "runspace" / "outputs" / "logs"

TARGET_PAIRS = 1500
GAP_CHOICES = list(range(10, 130, 10))   # {10, 20, ..., 120}
SEED = 42
MAX_TRIES_PER_SCENE = 200


def _scene_of(path: str) -> str:
    """Extract `sceneXXXX_YY` from any of the three pair-path conventions used in this repo."""
    parts = path.strip().split("/")
    for p in parts:
        if p.startswith("scene") and "_" in p:
            return p
    raise ValueError(f"no scene token in path: {path}")


def _scenes_in_pairs_file(path: Path) -> set[str]:
    out: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 2:
                continue
            out.add(_scene_of(cols[0]))
            out.add(_scene_of(cols[1]))
    return out


def _load_intrinsic(scene_dir: Path) -> np.ndarray:
    """Read intrinsic.txt (4×4 with K in upper-left 3×3) and return the 3×3 K."""
    M = np.loadtxt(scene_dir / "intrinsic.txt")
    return M[:3, :3].astype(np.float64)


def _load_pose(frame_txt: Path) -> np.ndarray | None:
    """Read a per-frame cam-to-world 4×4. Return None on -inf/nan/sensor-failure poses."""
    try:
        T = np.loadtxt(frame_txt)
    except Exception:
        return None
    if T.shape != (4, 4):
        return None
    if not np.all(np.isfinite(T)):
        return None
    return T.astype(np.float64)


def _list_frames(scene_dir: Path) -> list[int]:
    """Frames present as `<NNNNN>.jpg` AND `<NNNNN>.txt` AND `<NNNNN>.png`."""
    frames: list[int] = []
    for jpg in scene_dir.glob("*.jpg"):
        stem = jpg.stem
        if not stem.isdigit():
            continue
        if not (scene_dir / f"{stem}.txt").exists():
            continue
        if not (scene_dir / f"{stem}.png").exists():
            continue
        frames.append(int(stem))
    frames.sort()
    return frames


def _format_38(img0: str, img1: str, K0: np.ndarray, K1: np.ndarray, T_0to1: np.ndarray) -> str:
    flat_K0 = " ".join(f"{x:.6f}" for x in K0.reshape(-1))
    flat_K1 = " ".join(f"{x:.6f}" for x in K1.reshape(-1))
    flat_T = " ".join(f"{x:.6f}" for x in T_0to1.reshape(-1))
    return f"{img0} {img1} 0 0 {flat_K0} {flat_K1} {flat_T}"


def _pair_for_scene(scene: str, rng: random.Random) -> str | None:
    scene_dir = SCANNET_ROOT / scene
    if not scene_dir.is_dir():
        return None
    try:
        K = _load_intrinsic(scene_dir)
    except Exception:
        return None
    frames = _list_frames(scene_dir)
    if len(frames) < 2:
        return None

    for _ in range(MAX_TRIES_PER_SCENE):
        gap = rng.choice(GAP_CHOICES)
        f0_candidates = [f for f in frames if (f + gap) in frames]
        if not f0_candidates:
            continue
        f0 = rng.choice(f0_candidates)
        f1 = f0 + gap

        T0 = _load_pose(scene_dir / f"{f0:05d}.txt")
        T1 = _load_pose(scene_dir / f"{f1:05d}.txt")
        if T0 is None or T1 is None:
            continue
        T_0to1 = np.linalg.inv(T1) @ T0
        if not np.all(np.isfinite(T_0to1)):
            continue
        img0 = f"{scene}/{f0:05d}.jpg"
        img1 = f"{scene}/{f1:05d}.jpg"
        return _format_38(img0, img1, K, K, T_0to1)
    return None


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[gen_val_pairs] reading val scene list: {VAL_SCENES_LIST}")
    with open(VAL_SCENES_LIST) as f:
        val_scenes = [s.strip() for s in f if s.strip()]
    print(f"[gen_val_pairs] {len(val_scenes)} scenes in val list")

    print("[gen_val_pairs] computing scene-level disjointness against train and test")
    train_scenes = _scenes_in_pairs_file(TRAIN_PAIRS)
    test_scenes = _scenes_in_pairs_file(TEST_PAIRS)
    print(f"[gen_val_pairs]   train scenes: {len(train_scenes)}")
    print(f"[gen_val_pairs]   test scenes:  {len(test_scenes)}")
    val_set = set(val_scenes)
    overlap_vt = val_set & train_scenes
    overlap_vT = val_set & test_scenes
    overlap_tT = train_scenes & test_scenes
    print(f"[gen_val_pairs]   val ∩ train:  {len(overlap_vt)}")
    print(f"[gen_val_pairs]   val ∩ test:   {len(overlap_vT)}")
    print(f"[gen_val_pairs]   train ∩ test: {len(overlap_tT)}")
    if overlap_vt or overlap_vT:
        print(f"[gen_val_pairs] FAIL: val scene list overlaps train or test", file=sys.stderr)
        return 2

    print(f"[gen_val_pairs] filtering val scenes by on-disk presence under {SCANNET_ROOT}")
    available = [s for s in val_scenes if (SCANNET_ROOT / s).is_dir()]
    missing = [s for s in val_scenes if not (SCANNET_ROOT / s).is_dir()]
    print(f"[gen_val_pairs]   available on disk: {len(available)}; missing: {len(missing)}")
    if missing[:5]:
        print(f"[gen_val_pairs]   first missing: {missing[:5]}")
    if len(available) < 50:
        print(f"[gen_val_pairs] FAIL: too few val scenes available ({len(available)})", file=sys.stderr)
        return 2

    rng = random.Random(SEED)
    rng.shuffle(available)

    print(f"[gen_val_pairs] sampling {TARGET_PAIRS} pairs across {len(available)} scenes")
    pairs: list[str] = []
    used_scenes: set[str] = set()
    scene_idx = 0
    while len(pairs) < TARGET_PAIRS:
        if scene_idx >= len(available):
            scene_idx = 0
            rng.shuffle(available)
        scene = available[scene_idx]
        scene_idx += 1
        line = _pair_for_scene(scene, rng)
        if line is None:
            continue
        pairs.append(line)
        used_scenes.add(scene)
        if len(pairs) % 250 == 0:
            print(f"[gen_val_pairs]   progress: {len(pairs)}/{TARGET_PAIRS} pairs, "
                  f"{len(used_scenes)} unique scenes")

    print(f"[gen_val_pairs] writing {len(pairs)} pairs from {len(used_scenes)} scenes -> {OUT_FILE}")
    with open(OUT_FILE, "w") as f:
        for line in pairs:
            f.write(line + "\n")

    print("[gen_val_pairs] re-verifying scene-level disjointness on the WRITTEN file")
    out_scenes = _scenes_in_pairs_file(OUT_FILE)
    print(f"[gen_val_pairs]   scenes in output: {len(out_scenes)}")
    bad_t = out_scenes & train_scenes
    bad_T = out_scenes & test_scenes
    print(f"[gen_val_pairs]   output ∩ train:   {len(bad_t)}")
    print(f"[gen_val_pairs]   output ∩ test:    {len(bad_T)}")
    if bad_t or bad_T:
        print("[gen_val_pairs] FAIL: written file has scene overlap", file=sys.stderr)
        return 3

    print("[gen_val_pairs] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
