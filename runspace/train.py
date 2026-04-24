"""Design C Phase 2 training entrypoint.

Runs the head-only training loop for `ObservedLearnedSinkhorn` with all
defaults coming from the YAML config. Optional CLI overrides are available
for smoke-testing (e.g. `--max-train-pairs`, `--epochs`).

Invocation (inside container):
    docker exec qbench bash -lc \
        'cd /app && PYTHONPATH=/app python runspace/train.py \
            --config runspace/inputs/train_configs/design_c_t3_scannet.yaml'
"""
import argparse
import os
import sys

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
RUNSPACE_ROOT = os.path.join(PROJECT_ROOT, 'runspace')
if RUNSPACE_ROOT not in sys.path:
    sys.path.insert(0, RUNSPACE_ROOT)

if 'TORCH_HOME' not in os.environ:
    os.environ['TORCH_HOME'] = os.path.join(PROJECT_ROOT, '.cache', 'torch')

from runspace.src.train.trainer import Trainer  # noqa: E402


def _parse_args():
    ap = argparse.ArgumentParser(description="Design C Phase 2 head-only training")
    ap.add_argument('--config', required=True, type=str, help='path to training YAML')
    ap.add_argument('--epochs', type=int, default=None, help='override schedule.epochs')
    ap.add_argument('--max-train-pairs', type=int, default=None,
                    help='override train.dataset.max_pairs (smoke tests)')
    ap.add_argument('--max-val-pairs', type=int, default=None,
                    help='override train.val_dataset.max_pairs')
    return ap.parse_args()


def main():
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg.setdefault('train', {}).setdefault('schedule', {})['epochs'] = args.epochs
    if args.max_train_pairs is not None:
        cfg['train']['dataset']['max_pairs'] = args.max_train_pairs
    if args.max_val_pairs is not None:
        cfg['train']['val_dataset']['max_pairs'] = args.max_val_pairs

    trainer = Trainer(cfg)
    trainer.fit()
    print(f"[done] best {cfg['checkpoint'].get('select_metric', 'pose_auc_10')} = {trainer.best_score:.4f}")


if __name__ == '__main__':
    main()
