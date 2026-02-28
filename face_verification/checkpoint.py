"""
Save and load training checkpoints (model, optimizer, scheduler, metrics, epoch).
"""
import torch


def save_model(model, optimizer, scheduler, metrics, epoch, path):
    """Save checkpoint to path."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metric": metrics,
            "epoch": epoch,
        },
        path,
    )


def load_model(model, path, optimizer=None, scheduler=None):
    """Load checkpoint from path; optionally restore optimizer and scheduler."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metric", {})
    return model, optimizer, scheduler, epoch, metrics
