import os
import numpy as np


def _to_rgb(img_tensor):
    """Convert [1,H,W] or [H,W] float tensor in [0,1] to uint8 numpy [H,W,3]."""
    t = img_tensor
    if hasattr(t, 'cpu'):
        t = t.cpu()
    arr = t.numpy() if hasattr(t, 'numpy') else t
    if arr.ndim == 3:
        arr = arr[0]
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def _extract_matches(outputs, idx=0):
    """Return (kpts0, kpts1, mkpts0, mkpts1) numpy arrays for one item in the batch."""
    kpts0 = outputs['keypoints0'][idx].cpu().numpy()
    kpts1 = outputs['keypoints1'][idx].cpu().numpy()
    matches0 = outputs['matches0'][idx].cpu().numpy()
    valid = matches0 > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]
    return kpts0, kpts1, mkpts0, mkpts1


def _side_by_side(img0, img1):
    H, W = img0.shape[:2]
    canvas = np.zeros((H, 2 * W, 3), dtype=np.uint8)
    canvas[:, :W] = img0
    canvas[:, W:] = img1
    return canvas, W


def _info_header(quant_info, batch_metrics):
    """Build a multi-line annotation string from quant_info and batch_metrics."""
    if not quant_info and not batch_metrics:
        return None

    lines = []

    if quant_info:
        lines.append(
            f"Quantization: {quant_info.get('quant_type', '?')}  |  "
            f"Quantized: {quant_info.get('components', '?')}"
        )

    if batch_metrics:
        m = batch_metrics
        if 'ref_matches' in m:
            ref_prec = f"  prec: {m['ref_precision']:.3f}" if 'ref_precision' in m else ''
            q_prec = f"  prec: {m['quant_precision']:.3f}" if 'quant_precision' in m else ''
            delta_m = m['quant_matches'] - m['ref_matches']
            delta_str = f"{delta_m:+d}"

            lines.append(
                f"Ref:   {m['ref_matches']} matches  "
                f"({m.get('ref_kpts0','?')} / {m.get('ref_kpts1','?')} kpts){ref_prec}"
            )
            lines.append(
                f"Quant: {m['quant_matches']} matches  "
                f"({m.get('quant_kpts0','?')} / {m.get('quant_kpts1','?')} kpts){q_prec}  "
                f"Δ matches: {delta_str}"
            )
            if 'ref_precision' in m and 'quant_precision' in m:
                delta_prec = m['quant_precision'] - m['ref_precision']
                lines.append(f"Δ precision vs ref: {delta_prec:+.3f}")

        elif 'ref_kpts0' in m:
            lines.append(
                f"Ref kpts (img0/img1): {m['ref_kpts0']} / {m['ref_kpts1']}"
            )
            lines.append(
                f"Quant kpts (img0/img1): {m['quant_kpts0']} / {m['quant_kpts1']}  "
                f"Δ: {m['quant_kpts0']-m['ref_kpts0']:+d} / {m['quant_kpts1']-m['ref_kpts1']:+d}"
            )

        elif 'ref_kpts' in m:
            delta_k = m['quant_kpts'] - m['ref_kpts']
            lines.append(
                f"Ref: {m['ref_kpts']} kpts  |  Quant: {m['quant_kpts']} kpts  |  Δ: {delta_k:+d}"
            )

    return '\n'.join(lines)


def _add_info_box(fig, text):
    """Add a styled info box below the figure (bbox_inches='tight' will include it)."""
    if not text:
        return
    fig.text(
        0.5, -0.005, text,
        ha='center', va='top',
        fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e1e2e', edgecolor='#444', alpha=0.92),
        color='#cdd6f4',
        transform=fig.transFigure,
    )


def _draw_matches_on_ax(ax, img0, img1, kpts0, kpts1, mkpts0, mkpts1,
                         line_color, kpt_color, title):
    canvas, W = _side_by_side(img0, img1)
    ax.imshow(canvas)
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis('off')

    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        ax.plot([x0, x1 + W], [y0, y1], color=line_color, linewidth=0.7, alpha=0.65)

    if len(kpts0):
        ax.scatter(kpts0[:, 0], kpts0[:, 1], c=kpt_color, s=3, linewidths=0, zorder=3, alpha=0.6)
    if len(kpts1):
        ax.scatter(kpts1[:, 0] + W, kpts1[:, 1], c=kpt_color, s=3, linewidths=0, zorder=3, alpha=0.6)
    if len(mkpts0):
        ax.scatter(mkpts0[:, 0], mkpts0[:, 1], c=line_color, s=7, linewidths=0, zorder=4)
    if len(mkpts1):
        ax.scatter(mkpts1[:, 0] + W, mkpts1[:, 1], c=line_color, s=7, linewidths=0, zorder=4)


def save_matching_viz(batch, ref_outputs, quant_outputs, output_path,
                      pair_id="", quant_info=None, batch_metrics=None):
    """
    Save a 3-row figure comparing reference vs quantized SuperGlue matches.
    Row 0: reference matches (green)
    Row 1: quantized matches (blue)
    Row 2: overlay of both
    Header: quant type, quantized components, per-pair accuracy vs ref and GT.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img0 = _to_rgb(batch['image0'][0])
    img1 = _to_rgb(batch['image1'][0])
    _, W = img0.shape[:2]

    ref_kpts0, ref_kpts1, ref_mkpts0, ref_mkpts1 = _extract_matches(ref_outputs)
    q_kpts0, q_kpts1, q_mkpts0, q_mkpts1 = _extract_matches(quant_outputs)

    ref_prec_str = (f"  prec: {batch_metrics['ref_precision']:.3f}"
                    if batch_metrics and 'ref_precision' in batch_metrics else '')
    q_prec_str = (f"  prec: {batch_metrics['quant_precision']:.3f}"
                  if batch_metrics and 'quant_precision' in batch_metrics else '')

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    _draw_matches_on_ax(
        axes[0], img0, img1, ref_kpts0, ref_kpts1, ref_mkpts0, ref_mkpts1,
        line_color='lime', kpt_color='lime',
        title=f"Reference  |  {len(ref_mkpts0)} matches  "
              f"({len(ref_kpts0)} / {len(ref_kpts1)} kpts){ref_prec_str}",
    )
    _draw_matches_on_ax(
        axes[1], img0, img1, q_kpts0, q_kpts1, q_mkpts0, q_mkpts1,
        line_color='deepskyblue', kpt_color='deepskyblue',
        title=f"Quantized  |  {len(q_mkpts0)} matches  "
              f"({len(q_kpts0)} / {len(q_kpts1)} kpts){q_prec_str}",
    )

    canvas, _ = _side_by_side(img0, img1)
    axes[2].imshow(canvas)
    axes[2].set_title(
        f"Overlay  |  Ref (green): {len(ref_mkpts0)}  |  Quant (blue): {len(q_mkpts0)}",
        fontsize=9, pad=3,
    )
    axes[2].axis('off')
    for (x0, y0), (x1, y1) in zip(ref_mkpts0, ref_mkpts1):
        axes[2].plot([x0, x1 + W], [y0, y1], color='lime', linewidth=0.7, alpha=0.55)
    for (x0, y0), (x1, y1) in zip(q_mkpts0, q_mkpts1):
        axes[2].plot([x0, x1 + W], [y0, y1], color='deepskyblue', linewidth=0.7, alpha=0.55)

    label = pair_id if pair_id else "unknown"
    fig.suptitle(f"Match Comparison — {label}", fontsize=11, y=1.0)
    _add_info_box(fig, _info_header(quant_info, batch_metrics))
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_superglue_keypoints_viz(batch, ref_outputs, quant_outputs, output_path,
                                  pair_id="", quant_info=None, batch_metrics=None):
    """
    Save a 2-row × 3-col figure comparing keypoints for SuperPoint+SuperGlue outputs.
    Row 0: image0 keypoints — ref | quant | overlay
    Row 1: image1 keypoints — ref | quant | overlay
    Used when SuperPoint is quantized (keypoints differ between ref and quant).
    Header: quant type, quantized components, per-image keypoint delta.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img0 = _to_rgb(batch['image0'][0])
    img1 = _to_rgb(batch['image1'][0])

    ref_kpts0 = ref_outputs['keypoints0'][0].cpu().numpy()
    ref_kpts1 = ref_outputs['keypoints1'][0].cpu().numpy()
    q_kpts0 = quant_outputs['keypoints0'][0].cpu().numpy()
    q_kpts1 = quant_outputs['keypoints1'][0].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for row, (img, ref_kpts, q_kpts, lbl) in enumerate([
        (img0, ref_kpts0, q_kpts0, "Image 0"),
        (img1, ref_kpts1, q_kpts1, "Image 1"),
    ]):
        ref_set = set(map(tuple, ref_kpts.round().astype(int).tolist()))
        q_set = set(map(tuple, q_kpts.round().astype(int).tolist()))
        common = ref_set & q_set
        ref_only = ref_set - q_set
        q_only = q_set - ref_set

        def _pts(s):
            return np.array(list(s)) if s else np.empty((0, 2))

        for col, (kpts, color, title) in enumerate([
            (ref_kpts, 'lime',      f"{lbl} — Reference  |  {len(ref_kpts)} kpts"),
            (q_kpts,   'deepskyblue', f"{lbl} — Quantized  |  {len(q_kpts)} kpts  "
                                    f"(Δ {len(q_kpts)-len(ref_kpts):+d})"),
        ]):
            ax = axes[row, col]
            ax.imshow(img)
            if len(kpts):
                ax.scatter(kpts[:, 0], kpts[:, 1], c=color, s=5, linewidths=0)
            ax.set_title(title, fontsize=9)
            ax.axis('off')

        pts_both = _pts(common)
        pts_ref_only = _pts(ref_only)
        pts_q_only = _pts(q_only)
        ax = axes[row, 2]
        ax.imshow(img)
        if len(pts_both):
            ax.scatter(pts_both[:, 0], pts_both[:, 1], c='yellow', s=5,
                       linewidths=0, label=f'Both ({len(pts_both)})')
        if len(pts_ref_only):
            ax.scatter(pts_ref_only[:, 0], pts_ref_only[:, 1], c='lime', s=5,
                       linewidths=0, label=f'Ref only ({len(pts_ref_only)})')
        if len(pts_q_only):
            ax.scatter(pts_q_only[:, 0], pts_q_only[:, 1], c='deepskyblue', s=5,
                       linewidths=0, label=f'Quant only ({len(pts_q_only)})')
        ax.legend(fontsize=7, markerscale=2, loc='upper right')
        ax.set_title(
            f"{lbl} — Overlay  |  Both: {len(pts_both)}  "
            f"Ref-only: {len(pts_ref_only)}  Quant-only: {len(pts_q_only)}",
            fontsize=9,
        )
        ax.axis('off')

    label = pair_id if pair_id else "unknown"
    fig.suptitle(f"Keypoint Comparison (SuperPoint) — {label}", fontsize=11, y=1.0)
    _add_info_box(fig, _info_header(quant_info, batch_metrics))
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_keypoints_viz(batch, ref_outputs, quant_outputs, output_path,
                       pair_id="", quant_info=None, batch_metrics=None):
    """
    Save a 3-panel figure comparing reference vs quantized SuperPoint keypoints.
    Panel 0: reference keypoints (green)
    Panel 1: quantized keypoints (blue)
    Panel 2: overlay — green=ref only, blue=quant only, yellow=both
    Header: quant type, quantized components, keypoint count delta.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img_key = 'image0' if 'image0' in batch else 'image'
    img = _to_rgb(batch[img_key][0])

    ref_kpts = ref_outputs['keypoints'][0].cpu().numpy()
    q_kpts = quant_outputs['keypoints'][0].cpu().numpy()

    ref_set = set(map(tuple, ref_kpts.round().astype(int).tolist()))
    q_set = set(map(tuple, q_kpts.round().astype(int).tolist()))
    common = ref_set & q_set
    ref_only = ref_set - q_set
    q_only = q_set - ref_set

    def _pts(s):
        return np.array(list(s)) if s else np.empty((0, 2))

    pts_both = _pts(common)
    pts_ref_only = _pts(ref_only)
    pts_q_only = _pts(q_only)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, kpts, color, title in [
        (axes[0], ref_kpts, 'lime',
         f"Reference  |  {len(ref_kpts)} keypoints"),
        (axes[1], q_kpts, 'deepskyblue',
         f"Quantized  |  {len(q_kpts)} keypoints  (Δ {len(q_kpts)-len(ref_kpts):+d})"),
    ]:
        ax.imshow(img)
        if len(kpts):
            ax.scatter(kpts[:, 0], kpts[:, 1], c=color, s=5, linewidths=0)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    axes[2].imshow(img)
    if len(pts_both):
        axes[2].scatter(pts_both[:, 0], pts_both[:, 1], c='yellow', s=5,
                        linewidths=0, label=f'Both ({len(pts_both)})')
    if len(pts_ref_only):
        axes[2].scatter(pts_ref_only[:, 0], pts_ref_only[:, 1], c='lime', s=5,
                        linewidths=0, label=f'Ref only ({len(pts_ref_only)})')
    if len(pts_q_only):
        axes[2].scatter(pts_q_only[:, 0], pts_q_only[:, 1], c='deepskyblue', s=5,
                        linewidths=0, label=f'Quant only ({len(pts_q_only)})')
    axes[2].legend(fontsize=8, markerscale=2, loc='upper right')
    axes[2].set_title(
        f"Overlay  |  Both: {len(pts_both)}  "
        f"Ref-only: {len(pts_ref_only)}  Quant-only: {len(pts_q_only)}",
        fontsize=9,
    )
    axes[2].axis('off')

    label = pair_id if pair_id else "unknown"
    fig.suptitle(f"Keypoint Comparison — {label}", fontsize=11, y=1.0)
    _add_info_box(fig, _info_header(quant_info, batch_metrics))
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
