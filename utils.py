import random

import numpy as np
import torch
from scipy import ndimage as ndi


def fix_randseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _surface(mask):
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    return mask & ~ndi.binary_erosion(mask, structure=structure, border_value=0)


def hd95_mm(prediction, truth, spacing=None):
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if prediction.shape != truth.shape:
        raise ValueError('prediction and truth must have identical shapes')
    spacing = np.ones(prediction.ndim) if spacing is None else np.asarray(spacing, dtype=float)
    if not prediction.any() and not truth.any():
        return 0.0
    if not prediction.any() or not truth.any():
        return float(np.linalg.norm(np.asarray(prediction.shape) * spacing))
    p = _surface(prediction)
    t = _surface(truth)
    d_to_t = ndi.distance_transform_edt(~t, sampling=spacing)[p]
    d_to_p = ndi.distance_transform_edt(~p, sampling=spacing)[t]
    return float(np.percentile(np.concatenate((d_to_t, d_to_p)), 95))


class Scores:
    def __init__(self):
        self.patient_dice = []
        self.patient_iou = []
        self.patient_hd95 = []
        self.patient_asd = []
        self.patient_dice1 = []
        self.TP = self.TN = self.FP = self.FN = 0

    def record(self, preds, label, spacing=None):
        p = preds.detach().cpu().numpy().astype(bool)
        y = label.detach().cpu().numpy().astype(bool)
        if p.shape != y.shape:
            raise ValueError(f'Shape mismatch: prediction {p.shape}, target {y.shape}')
        tp = int(np.count_nonzero(p & y))
        fp = int(np.count_nonzero(p & ~y))
        fn = int(np.count_nonzero(~p & y))
        tn = int(np.count_nonzero(~p & ~y))
        dice = float(2 * tp / (2 * tp + fp + fn)) if 2*tp + fp + fn else 1.0
        iou = float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0
        hd95 = hd95_mm(p, y, spacing=spacing)
        self.patient_dice.append(dice)
        self.patient_iou.append(iou)
        self.patient_hd95.append(hd95)
        self.patient_dice1.append(dice)
        self.patient_asd.append(float('nan'))
        self.TP += tp
        self.FP += fp
        self.FN += fn
        self.TN += tn
        return dice, iou, hd95

    def compute_dice(self):
        denom = 2*self.TP + self.FP + self.FN
        return 2*self.TP/denom if denom else 1.0

    def compute_iou(self):
        denom = self.TP+self.FP+self.FN
        return self.TP/denom if denom else 1.0
