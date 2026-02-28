"""
Evaluation: classification accuracy/loss and verification metrics (EER, AUC, etc.).
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .metrics import AverageMeter, accuracy, get_ver_metrics


@torch.no_grad()
def valid_epoch_cls(model, dataloader, device, config, criterion):
    """Run validation for classification; return mean accuracy and loss."""
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Val",
        ncols=5,
    )
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.inference_mode():
            outputs = model(images)
            logits = outputs["out"]
            loss = criterion(logits, labels)
            acc = accuracy(logits, labels)[0].item()
        loss_m.update(loss.item())
        acc_m.update(acc)
        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
        )
        batch_bar.update()
    batch_bar.close()
    return acc_m.avg, loss_m.avg


def valid_epoch_ver(model, pair_data_loader, device, config):
    """Evaluate verification on paired data with labels; compute EER, AUC, ACC; return ACC."""
    model.eval()
    scores = []
    match_labels = []
    batch_bar = tqdm(
        total=len(pair_data_loader),
        dynamic_ncols=True,
        position=0,
        leave=False,
        desc="Val Veri.",
    )
    for i, (images1, images2, labels) in enumerate(pair_data_loader):
        images = torch.cat([images1, images2], dim=0).to(device)
        with torch.inference_mode():
            outputs = model(images)
        feats = F.normalize(outputs["feats"], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.append(similarity.cpu().numpy())
        match_labels.append(labels.cpu().numpy())
        batch_bar.update()

    scores = np.concatenate(scores)
    match_labels = np.concatenate(match_labels)
    FPRs = ["1e-4", "5e-4", "1e-3", "5e-3", "5e-2"]
    metric_dict = get_ver_metrics(match_labels.tolist(), scores.tolist(), FPRs)
    print(metric_dict)
    return metric_dict["ACC"]


def test_epoch_ver(model, pair_data_loader, device, config):
    """Run verification on test pairs (no labels); return list of similarity scores."""
    model.eval()
    scores = []
    batch_bar = tqdm(
        total=len(pair_data_loader),
        dynamic_ncols=True,
        position=0,
        leave=False,
        desc="Test Veri.",
    )
    for i, (images1, images2) in enumerate(pair_data_loader):
        images = torch.cat([images1, images2], dim=0).to(device)
        with torch.inference_mode():
            outputs = model(images)
        feats = F.normalize(outputs["feats"], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.extend(similarity.cpu().numpy().tolist())
        batch_bar.update()
    return scores
