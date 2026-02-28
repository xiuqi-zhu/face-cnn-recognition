"""
Metrics: running average, top-k accuracy, and verification (EER, AUC, TPR@FPR).
"""
import torch
from sklearn import metrics as mt
from scipy.optimize import brentq
from scipy.interpolate import interp1d


class AverageMeter:
    """Running average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy (percent). output: logits [N,C], target: labels [N]."""
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def get_ver_metrics(labels, scores, FPRs):
    """
    Verification metrics from binary labels and similarity scores.
    Returns ACC, EER (%), AUC (%), and TPR at given FPRs.
    """
    fpr, tpr, _ = mt.roc_curve(labels, scores, pos_label=1)
    roc_curve = interp1d(fpr, tpr)
    EER = 100.0 * brentq(lambda x: 1.0 - x - roc_curve(x), 0.0, 1.0)
    AUC = 100.0 * mt.auc(fpr, tpr)

    tnr = 1.0 - fpr
    pos_num = labels.count(1)
    neg_num = labels.count(0)
    ACC = 100.0 * max(tpr * pos_num + tnr * neg_num) / len(labels)

    if isinstance(FPRs, list):
        TPRs = [
            ("TPR@FPR={}".format(FPR), 100.0 * float(roc_curve(float(FPR))))
            for FPR in FPRs
        ]
    else:
        TPRs = []

    return {
        "ACC": ACC,
        "EER": EER,
        "AUC": AUC,
        "TPRs": TPRs,
    }
