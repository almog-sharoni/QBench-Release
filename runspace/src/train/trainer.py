"""Design C Phase 2 — head-only training loop for ObservedLearnedSinkhorn.

Freezes SuperPoint and the SuperGlue backbone (kenc / gnn / final_proj /
bin_score); trains only the learned Sinkhorn head. Validates against
ScanNet-1500 via the existing MatchingMetrics; tracks `pose_auc_10` as the
selection metric. Checkpoints the head's state_dict (< 100 KB each) to
`checkpoint.out_dir`, plus a `best.pt` symlink-style copy for the top score.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.adapters.pipelines  # noqa: F401 — triggers pipeline registration
import src.datasets  # noqa: F401 — triggers val dataset registration
# Side-effect imports: register dataset + pipeline
from src.adapters.feature_matching_adapter import FeatureMatchingPipeline
from src.adapters.pipeline_registry import load_pipeline
from src.datasets.dataset_registry import build_data_loader
from src.eval.metrics.matching import MatchingMetrics
from src.train import scannet_training_pairs  # noqa: F401 — registers dataset
from src.train.gt_correspondences import batch_compute_gt_matches
from src.train.losses import design_c_prime_base2_total_loss, design_c_total_loss


def _pad_kpts(kpt_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length SuperPoint outputs into a dense (B, K_max, 2) tensor.

    Returns (padded, valid_count_per_image).
    """
    counts = [k.shape[0] for k in kpt_list]
    K_max = max(counts) if counts else 0
    B = len(kpt_list)
    if K_max == 0:
        device = kpt_list[0].device if kpt_list else torch.device('cpu')
        return torch.zeros(B, 0, 2, device=device), torch.tensor(counts)
    device = kpt_list[0].device
    out = torch.zeros(B, K_max, 2, device=device, dtype=kpt_list[0].dtype)
    for b, k in enumerate(kpt_list):
        out[b, : k.shape[0]] = k
    return out, torch.tensor(counts)


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_dir = cfg['checkpoint']['out_dir']
        os.makedirs(self.out_dir, exist_ok=True)

        self.model = self._build_model()
        self._freeze_non_head()

        # Warm-start from a prior checkpoint if requested (plan §3.7). Restores
        # head_state_dict + bin_score only; optimizer and scheduler state are
        # fresh — Phase 2b begins a new cosine schedule.
        resume_from = cfg['train'].get('resume_from') if 'train' in cfg else None
        if resume_from:
            self.load_checkpoint(resume_from)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.train_loader = build_data_loader(
            cfg['train']['dataset']['name'], cfg['train']['dataset'])
        self.val_loader = build_data_loader(
            cfg['train']['val_dataset']['name'], cfg['train']['val_dataset'])

        # Loss dispatch: Base2 pipelines use the base-2 variant of the loss.
        pipeline_name = cfg.get('target_pipeline', 'superpoint_superglue_learned')
        self._use_base2_loss = pipeline_name.endswith('_base2')

        self.global_step = 0
        self.best_score = -float('inf')
        self.history: list[dict] = []

    def load_checkpoint(self, path: str) -> None:
        """Restore head_state_dict + bin_score from a Phase 2 / Phase 2b checkpoint.

        Skips optimizer and scheduler state intentionally — Phase 2b begins a
        fresh cosine schedule per plan §3.7.
        """
        ckpt = torch.load(path, map_location=self.device)
        head_state = ckpt['head_state_dict'] if isinstance(ckpt, dict) else ckpt
        self.model.backbone.superglue.sinkhorn.load_state_dict(head_state)
        if isinstance(ckpt, dict) and 'bin_score' in ckpt:
            with torch.no_grad():
                self.model.backbone.superglue.bin_score.copy_(
                    ckpt['bin_score'].to(self.model.backbone.superglue.bin_score.device)
                )
        print(f"[resume] loaded head weights + bin_score from {path}", flush=True)

    # ---------- setup ----------

    def _build_model(self) -> nn.Module:
        model_cfg = dict(self.cfg['model'])
        pipeline_name = self.cfg.get('target_pipeline', 'superpoint_superglue_learned')
        backbone = load_pipeline(pipeline_name, model_cfg)
        model = FeatureMatchingPipeline(backbone).to(self.device)
        return model

    def _freeze_non_head(self) -> None:
        sg = self.model.backbone.superglue
        sp = self.model.backbone.superpoint

        for p in sp.parameters():
            p.requires_grad = False
        for name, p in sg.named_parameters():
            p.requires_grad = name.startswith('sinkhorn.')

        # Put frozen submodules in eval mode (stops BN buffer drift).
        sp.eval()
        for attr in ('kenc', 'gnn', 'final_proj', 'kenc_add0', 'kenc_add1',
                     'desc_matmul', 'match_select',
                     'normalize_kpts0', 'normalize_kpts1'):
            if hasattr(sg, attr):
                getattr(sg, attr).eval()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[freeze] trainable params: {trainable:,} / {total:,}")

    def _set_training_mode(self) -> None:
        """Keep frozen parts in eval; put only `sinkhorn` in train mode."""
        self.model.backbone.superpoint.eval()
        sg = self.model.backbone.superglue
        for attr in ('kenc', 'gnn', 'final_proj', 'kenc_add0', 'kenc_add1',
                     'desc_matmul', 'match_select',
                     'normalize_kpts0', 'normalize_kpts1'):
            if hasattr(sg, attr):
                getattr(sg, attr).eval()
        sg.sinkhorn.train()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.cfg['train']['optimizer']
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=float(opt_cfg['lr']),
            weight_decay=float(opt_cfg.get('weight_decay', 0.0)),
        )

    def _build_scheduler(self):
        sch_cfg = self.cfg['train'].get('schedule', {})
        epochs = int(sch_cfg.get('epochs', 1))
        warmup = int(sch_cfg.get('warmup_epochs', 0))
        if sch_cfg.get('cosine', False):
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(epochs - warmup, 1)
            )
        return None

    # ---------- training ----------

    def _lambda_marg(self, epoch: int) -> float:
        loss_cfg = self.cfg['train']['loss']
        start = float(loss_cfg.get('lambda_marg_initial', 0.1))
        end = float(loss_cfg.get('lambda_marg_final', 0.01))
        anneal_ep = int(loss_cfg.get('lambda_marg_anneal_epoch', 10))
        return end if epoch >= anneal_ep else start

    def _forward_with_superpoint(self, batch: dict) -> tuple[dict, dict]:
        """Run SuperPoint on both images, then SuperGlueLearned with trace.

        Returns (superpoint_outputs, superglue_training_outputs).
        """
        sp = self.model.backbone.superpoint
        sg = self.model.backbone.superglue

        with torch.no_grad():  # SuperPoint frozen
            pred0 = sp({'image': batch['image0']})
            pred1 = sp({'image': batch['image1']})

        data = {
            'image0': batch['image0'], 'image1': batch['image1'],
            **{f'{k}0': v for k, v in pred0.items()},
            **{f'{k}1': v for k, v in pred1.items()},
        }
        for k in data:
            if isinstance(data[k], (list, tuple)) and len(data[k]) > 0 and torch.is_tensor(data[k][0]):
                data[k] = torch.stack(data[k])

        sg_out = sg.forward_with_sinkhorn_trace(data)
        sp_out = {
            'keypoints0': pred0['keypoints'],
            'keypoints1': pred1['keypoints'],
            'scores0': pred0['scores'],
            'scores1': pred1['scores'],
        }
        return sp_out, sg_out

    def _gt_matches(self, sp_out: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GT match indices for the batch."""
        kp0 = sp_out['keypoints0']
        kp1 = sp_out['keypoints1']
        kp0_p, _ = _pad_kpts(kp0)
        kp1_p, _ = _pad_kpts(kp1)

        reproj_thr = float(self.cfg['train']['loss'].get('gt_reproj_threshold_px', 3.0))
        gt0, gt1 = batch_compute_gt_matches(
            kp0_p, kp1_p,
            batch['depth0'].to(self.device),
            batch['depth1'].to(self.device),
            batch['K0'].to(self.device),
            batch['K1'].to(self.device),
            batch['T_0to1'].to(self.device),
            reproj_threshold_px=reproj_thr,
        )
        return gt0, gt1

    def train_one_epoch(self, epoch: int) -> dict:
        self._set_training_mode()
        loss_cfg = self.cfg['train']['loss']
        lambda_marg = self._lambda_marg(epoch)
        lambda_aux = float(loss_cfg.get('lambda_aux', 0.3))

        running = {'total': 0.0, 'match': 0.0, 'marg': 0.0, 'aux': 0.0}
        n_steps = 0
        t0 = time.time()

        for batch in self.train_loader:
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

            sp_out, sg_out = self._forward_with_superpoint(batch)
            gt0, gt1 = self._gt_matches(sp_out, batch)

            if self._use_base2_loss:
                losses = design_c_prime_base2_total_loss(
                    final_log2_T=sg_out['log_T'],
                    trace=sg_out['trace'],
                    log2_mu=sg_out['log_mu'],
                    log2_nu=sg_out['log_nu'],
                    gt_matches0=gt0,
                    gt_matches1=gt1,
                    lambda_marg=lambda_marg,
                    lambda_aux=lambda_aux,
                )
            else:
                losses = design_c_total_loss(
                    final_log_T=sg_out['log_T'],
                    trace=sg_out['trace'],
                    log_mu=sg_out['log_mu'],
                    log_nu=sg_out['log_nu'],
                    gt_matches0=gt0,
                    gt_matches1=gt1,
                    lambda_marg=lambda_marg,
                    lambda_aux=lambda_aux,
                )

            self.optimizer.zero_grad(set_to_none=True)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
            )
            self.optimizer.step()

            running['total'] += float(losses['total'].item())
            running['match'] += float(losses['match'].item())
            running['marg'] += float(losses['marg'].item())
            running['aux'] += float(losses['aux'].item())
            n_steps += 1
            self.global_step += 1

            if self.global_step % 10 == 0:
                print(
                    f"[train] ep {epoch} step {self.global_step} "
                    f"total={losses['total'].item():.4f} "
                    f"match={losses['match'].item():.4f} "
                    f"marg={losses['marg'].item():.4f} "
                    f"aux={losses['aux'].item():.4f}",
                    flush=True,
                )

        if self.scheduler is not None:
            self.scheduler.step()

        avg = {k: v / max(n_steps, 1) for k, v in running.items()}
        avg['epoch_seconds'] = time.time() - t0
        avg['lambda_marg'] = lambda_marg
        print(f"[train] ep {epoch} avg {json.dumps(avg)}", flush=True)
        return avg

    # ---------- validation ----------

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        metrics = MatchingMetrics()
        for batch in self.val_loader:
            batch_dev = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            inputs = {k: v for k, v in batch_dev.items()
                      if torch.is_tensor(v) and k not in ('T_0to1', 'K0', 'K1')}
            out = self.model(inputs)
            # Align with MatchingMetrics.update contract.
            out.setdefault('keypoints0', inputs.get('keypoints0'))
            out.setdefault('keypoints1', inputs.get('keypoints1'))
            targets = {k: batch_dev[k] for k in ('T_0to1', 'K0', 'K1') if k in batch_dev}
            metrics.update(out, targets)

        res = metrics.compute()
        print(f"[val] ep {epoch} {json.dumps({k: float(v) for k, v in res.items()})}", flush=True)
        return res

    # ---------- checkpointing ----------

    def save_checkpoint(self, epoch: int, val_metrics: dict, train_metrics: dict) -> str:
        select_metric = self.cfg['checkpoint'].get('select_metric', 'pose_auc_10')
        score = float(val_metrics.get(select_metric, 0.0))

        head = self.model.backbone.superglue.sinkhorn.state_dict()
        ckpt = {
            'epoch': epoch,
            'step': self.global_step,
            'val_metrics': {k: float(v) for k, v in val_metrics.items()},
            'train_metrics': {k: float(v) for k, v in train_metrics.items()},
            'config': self.cfg,
            'head_state_dict': head,
            'bin_score': self.model.backbone.superglue.bin_score.detach().cpu(),
        }
        path = os.path.join(self.out_dir, f"ckpt_ep{epoch:03d}.pt")
        torch.save(ckpt, path)
        print(f"[ckpt] saved {path}  score ({select_metric}) = {score:.4f}", flush=True)

        if score > self.best_score:
            self.best_score = score
            best_path = os.path.join(self.out_dir, "best.pt")
            shutil.copyfile(path, best_path)
            print(f"[ckpt] new best → {best_path}", flush=True)
        return path

    # ---------- main loop ----------

    def fit(self) -> None:
        epochs = int(self.cfg['train']['schedule'].get('epochs', 1))
        val_every = int(self.cfg['train'].get('val_every_epochs', 1))

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = {}
            if epoch % val_every == 0 or epoch == epochs:
                val_metrics = self.validate(epoch)
            ckpt_path = self.save_checkpoint(epoch, val_metrics, train_metrics)
            self.history.append({
                'epoch': epoch,
                'ckpt': ckpt_path,
                'train': train_metrics,
                'val': val_metrics,
            })
            with open(os.path.join(self.out_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f, indent=2)
