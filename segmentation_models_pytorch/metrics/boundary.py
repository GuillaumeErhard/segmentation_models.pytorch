# DO BF1C
Then loss
https://arxiv.org/pdf/1905.07852v1
import torch
import torch, torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

# TODO: No distance transform edt plz

def _extract_boundary(mask: torch.Tensor) -> torch.Tensor:
    """
    Thin binary boundary using 3×3 max‑pool (same idea as Canny gradient≠0).
    mask : (H,W) bool / 0‑1 tensor
    """
    mask = mask.float().unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
    dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
    return (dilated - mask).squeeze(0).squeeze(0).bool()

def _bf1_single(gt_bin: torch.Tensor,
                pred_bin: torch.Tensor,
                theta: int = 3) -> float:
    """Scalar BF1 for *one* class on *one* image."""
    gt_b   = _extract_boundary(gt_bin)
    pred_b = _extract_boundary(pred_bin)

    gt_np, pred_np = gt_b.cpu().numpy().astype(np.uint8), \
                     pred_b.cpu().numpy().astype(np.uint8)

    dt_gt   = distance_transform_edt(1 - gt_np)
    dt_pred = distance_transform_edt(1 - pred_np)

    matched_pred = (pred_np & (dt_gt   < theta)).sum()
    matched_gt   = (gt_np   & (dt_pred < theta)).sum()

    Pc = matched_pred / (pred_np.sum() + 1e‑8)
    Rc = matched_gt   / (gt_np.sum()   + 1e‑8)
    return 0. if Pc + Rc == 0 else 2 * Pc * Rc / (Pc + Rc)


def boundary_f1_multiclass(
        gt:   torch.Tensor,           # (B,H,W)   or (H,W)
        pred: torch.Tensor,           # same shape, already argmaxed
        num_classes: int,
        theta: int = 3,
        ignore_index: int | None = None,
        reduction: str = 'mean'       # 'none' | 'mean' | 'sum'
) -> torch.Tensor | float:
    """
    Compute BF1 over a batch & K classes.
    Returns tensor of shape (B,K) if reduction='none'.
    """
    if gt.dim() == 2:       # promote to batch of 1
        gt, pred = gt.unsqueeze(0), pred.unsqueeze(0)

    B, H, W = gt.shape
    device  = gt.device
    out     = torch.zeros(B, num_classes, device=device)

    for b in range(B):
        for c in range(num_classes):
            if c == ignore_index:
                out[b, c] = float('nan')          # will be ignored in reduction
                continue
            out[b, c] = _bf1_single(gt[b] == c, pred[b] == c, theta)

    if reduction == 'none':
        return out
    mask = ~torch.isnan(out)
    if reduction == 'mean':
        return out[mask].mean().item() if mask.any() else 0.0
    if reduction == 'sum':
        return out[mask].sum().item()
    raise ValueError(f"Unknown reduction: {reduction}")
