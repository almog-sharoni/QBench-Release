"""V.1 Equivalence gate — ObservedLearnedSinkhorn vs ObservedLearnedSinkhornBase2.

Loads a natural-base Design C head checkpoint (Phase 2 best.pt), instantiates
both the natural-base and the Base2 heads with the same weights, runs both on
the 300 ScanNet val pairs, and reports:

  max | log_T - log2_T / log2(e) |  across all (M+1)(N+1) cells, per pair.

Pass criterion (plan §5.1): max across all pairs < 1e-3.
Kill gate   (plan §7 R1): fail on more than 5 percent of pairs -> widen LUT.

Usage (run inside the qbench container):
  docker exec qbench bash -c "PYTHONPATH=/app python runspace/src/scripts/verify_base2_equivalence.py \
    --ckpt /app/runspace/outputs/design_c_t3/checkpoints/best.pt \
    --pairs /app/runspace/inputs/scannet/pairs_test_paper_100scenes.txt \
    --root /data/scannet \
    --max-pairs 300"
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

# Ensure runspace modules are importable when run directly. Mirrors the sys.path
# setup in runspace/train.py so both `src.xxx` and `runspace.src.xxx` resolve.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
RUNSPACE_ROOT = os.path.join(PROJECT_ROOT, 'runspace')
for p in (PROJECT_ROOT, RUNSPACE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import src.adapters.pipelines  # noqa: F401 — triggers pipeline registration
import src.datasets  # noqa: F401 — triggers val dataset registration
from src.adapters.feature_matching_adapter import FeatureMatchingPipeline  # noqa: F401
from src.adapters.pipeline_registry import load_pipeline
from src.datasets.dataset_registry import build_data_loader


_LOG2E = math.log2(math.e)


def _build_model(pipeline_name: str, repo_path: str, sg_weights: str = 'indoor') -> torch.nn.Module:
    cfg = {
        'repo_path': repo_path,
        'sg_weights': sg_weights,
        'sp_config': {'max_keypoints': 512},
        'sg_config': {
            'learned_sinkhorn_iterations': 3,
            'learned_proj_dim': 8,
        },
    }
    return load_pipeline(pipeline_name, cfg)


def _load_head(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['head_state_dict'] if isinstance(ckpt, dict) and 'head_state_dict' in ckpt else ckpt
    model.superglue.sinkhorn.load_state_dict(state)
    if isinstance(ckpt, dict) and 'bin_score' in ckpt:
        with torch.no_grad():
            model.superglue.bin_score.copy_(ckpt['bin_score'])
    print(f"  loaded head weights from {ckpt_path}")


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='/app/runspace/outputs/design_c_t3/checkpoints/best.pt')
    p.add_argument('--pairs', default='/app/runspace/inputs/scannet/pairs_test_paper_100scenes.txt')
    p.add_argument('--root', default='/data/scannet')
    p.add_argument('--repo-path', default='/app/custom/SuperGluePretrainedNetwork')
    p.add_argument('--max-pairs', type=int, default=300)
    p.add_argument('--pass-thr', type=float, default=1e-3)
    p.add_argument('--fail-frac', type=float, default=0.05)
    p.add_argument('--dtype', choices=['float32', 'float64'], default='float64',
                   help="Precision for the equivalence check. FP64 isolates the "
                        "algorithmic correctness from FP32 cancellation; FP32 "
                        "reflects the actual training-time precision. Default FP64.")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    print(f"device: {device}  dtype: {args.dtype}")

    print("building natural-base model (ObservedLearnedSinkhorn)")
    nat = _build_model('superpoint_superglue_learned', args.repo_path).to(device).eval()
    _load_head(nat, args.ckpt)

    print("building base2 model (ObservedLearnedSinkhornBase2)")
    b2 = _build_model('superpoint_superglue_learned_base2', args.repo_path).to(device).eval()
    _load_head(b2, args.ckpt)

    # Cast only the learned heads + bin_score to the evaluation dtype. The
    # SuperPoint/SuperGlue backbones stay FP32; we'll cast the scores/desc
    # tensors at the sinkhorn boundary so both heads operate in the target
    # precision on identical inputs.
    nat.superglue.sinkhorn.to(dtype)
    b2.superglue.sinkhorn.to(dtype)
    nat.superglue.bin_score.data = nat.superglue.bin_score.data.to(dtype)
    b2.superglue.bin_score.data = b2.superglue.bin_score.data.to(dtype)

    dataset_cfg = {
        'name': 'scannet_pairs',
        'path': args.root,
        'pairs_file': args.pairs,
        'image_size': [480, 640],
        'batch_size': 1,
        'num_workers': 2,
        'max_pairs': args.max_pairs,
    }
    loader = build_data_loader(dataset_cfg['name'], dataset_cfg)

    sg_nat = nat.superglue
    sg_b2 = b2.superglue
    sp = nat.superpoint  # same SuperPoint weights in both models

    max_errs: list[float] = []
    n_fail = 0
    n_total = 0
    for batch in loader:
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        inputs = {k: v for k, v in inputs.items()
                  if torch.is_tensor(v) and k not in ('T_0to1', 'K0', 'K1')}

        with torch.no_grad():
            pred0 = sp({'image': inputs['image0']})
            pred1 = sp({'image': inputs['image1']})
            data = {
                'image0': inputs['image0'], 'image1': inputs['image1'],
                **{f'{k}0': v for k, v in pred0.items()},
                **{f'{k}1': v for k, v in pred1.items()},
            }
            for k in data:
                if isinstance(data[k], (list, tuple)) and len(data[k]) > 0 and torch.is_tensor(data[k][0]):
                    data[k] = torch.stack(data[k])

            # Run the FP32 backbone through to descriptor matmul (shared by
            # both heads). Then cast scores + mdesc to target dtype and run
            # each sinkhorn manually in that precision.
            kpts0 = sg_nat.normalize_kpts0(data['keypoints0'], data['image0'].shape)
            kpts1 = sg_nat.normalize_kpts1(data['keypoints1'], data['image1'].shape)
            desc0 = sg_nat.kenc_add0(data['descriptors0'], sg_nat.kenc(kpts0, data['scores0']))
            desc1 = sg_nat.kenc_add1(data['descriptors1'], sg_nat.kenc(kpts1, data['scores1']))
            desc0, desc1 = sg_nat.gnn(desc0, desc1)
            mdesc0, mdesc1 = sg_nat.final_proj(desc0), sg_nat.final_proj(desc1)
            scores = sg_nat.desc_matmul(mdesc0, mdesc1)

            scores_t = scores.to(dtype)
            mdesc0_t = mdesc0.to(dtype)
            mdesc1_t = mdesc1.to(dtype)

            log_T_nat, _, _, _ = sg_nat.sinkhorn(
                scores_t, sg_nat.bin_score, mdesc0_t, mdesc1_t, return_trace=True
            )
            log2_T, _, _, _ = sg_b2.sinkhorn(
                scores_t, sg_b2.bin_score, mdesc0_t, mdesc1_t, return_trace=True
            )

        converted = log2_T / _LOG2E
        diff = (log_T_nat - converted).abs()
        max_err = float(diff.max().item())
        max_errs.append(max_err)
        n_total += 1
        if max_err > args.pass_thr:
            n_fail += 1

    print("")
    print(f"pairs evaluated: {n_total}")
    print(f"max-error distribution (5 bins):")
    if max_errs:
        import numpy as np
        arr = np.asarray(max_errs)
        bins = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])
        for i in range(4):
            cnt = int(((arr >= bins[i]) & (arr <= bins[i + 1])).sum())
            print(f"  [{bins[i]:.6e} .. {bins[i+1]:.6e}]  n={cnt}")
        print(f"  max across all pairs: {arr.max():.6e}")
        print(f"  median:               {np.median(arr):.6e}")
    print("")
    frac_fail = n_fail / max(n_total, 1)
    print(f"pass threshold: {args.pass_thr} ; failing pairs: {n_fail}/{n_total} ({100*frac_fail:.2f}%)")
    if frac_fail > args.fail_frac:
        print(f"FAIL — failing fraction {frac_fail:.3f} exceeds {args.fail_frac:.3f}.")
        print("Per plan §R1: widen LUT to 32 entries or switch to minimax polynomial.")
        return 1
    print("PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
