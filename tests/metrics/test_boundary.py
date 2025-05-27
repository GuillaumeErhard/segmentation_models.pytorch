import torch, pytest

from segmentation_models_pytorch.metrics.boundary import boundary_f1_multiclass

def label_grid(classes, size=32):
    """Return a (H,W) tensor where each quadrant has a different label."""
    h = w = size
    y = torch.arange(h).unsqueeze(1).repeat(1,w)
    x = torch.arange(w).unsqueeze(0).repeat(h,1)
    return ((y >= h//2).long()*2 + (x >= w//2).long()) % classes

def test_multiclass_identity():
    gt   = label_grid(4)
    pred = gt.clone()
    score = boundary_f1_multiclass(gt, pred, num_classes=4, reduction='mean')
    assert abs(score - 1.0) < 1e-6

def test_ignore_index():
    gt   = label_grid(3)
    pred = gt.clone()
    pred[10:20, 10:20] = 255             # noise with ignore label
    s = boundary_f1_multiclass(gt, pred, 3, ignore_index=255)
    assert s < 1.0                       # disturbed boundaries ↓ score
