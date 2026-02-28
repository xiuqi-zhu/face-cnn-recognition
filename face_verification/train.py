"""
One-epoch training for face classification (supports CutMix targets).
"""
import torch
from tqdm import tqdm
from .metrics import AverageMeter, accuracy


def train_epoch(model, dataloader, optimizer, lr_scheduler, scaler, device, config, criterion):
    """Run one training epoch; return mean accuracy and loss."""
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Train",
        ncols=5,
    )

    for i, (images, targets) in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device, non_blocking=True)

        # CutMix returns (y_a, y_b, lam); otherwise targets are class indices
        if isinstance(targets, (tuple, list)):
            y_a, y_b, lam = targets
            y_a = y_a.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
        else:
            targets = targets.to(device, non_blocking=True)
            y_a = y_b = lam = None

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            logits = outputs["out"]
            if y_a is not None:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                acc = (
                    lam * accuracy(logits, y_a)[0].item()
                    + (1 - lam) * accuracy(logits, y_b)[0].item()
                )
            else:
                loss = criterion(logits, targets)
                acc = accuracy(logits, targets)[0].item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_m.update(loss.item())
        acc_m.update(acc)
        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
            lr="{:.06f}".format(float(optimizer.param_groups[0]["lr"])),
        )
        batch_bar.update()

    batch_bar.close()
    if lr_scheduler and not isinstance(
        lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        lr_scheduler.step()
    return acc_m.avg, loss_m.avg
