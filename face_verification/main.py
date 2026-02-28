"""
Main entry: train face classification and verification model with optional resume.
"""
import os
import argparse
import torch
import gc
from .config import get_config
from .dataset import create_dataloaders
from .model import Network
from .train import train_epoch
from .evaluate import valid_epoch_cls, valid_epoch_ver
from .checkpoint import save_model, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cls_data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.cls_data_dir is not None:
        overrides["cls_data_dir"] = args.cls_data_dir
        base = os.path.dirname(args.cls_data_dir.rstrip("/"))
        overrides["ver_data_dir"] = os.path.join(base, "ver_data")
        overrides["val_pairs_file"] = os.path.join(base, "val_pairs.txt")
        overrides["test_pairs_file"] = os.path.join(base, "test_pairs.txt")
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir

    config = get_config(**overrides)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    loaders = create_dataloaders(config)
    cls_train_loader = loaders["cls_train_loader"]
    cls_val_loader = loaders["cls_val_loader"]
    ver_val_loader = loaders["ver_val_loader"]

    model = Network(num_classes=config["num_classes"]).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    best_valid_cls_acc = 0.0
    best_valid_ret_acc = 0.0
    eval_cls = config.get("eval_cls", True)

    if args.resume and os.path.isfile(args.resume):
        model, optimizer, scheduler, start_epoch, metrics = load_model(
            model, args.resume, optimizer=optimizer, scheduler=scheduler
        )
        start_epoch += 1
        best_valid_cls_acc = metrics.get("valid_cls_acc", 0.0)
        best_valid_ret_acc = metrics.get("valid_ret_acc", 0.0)
        print(
            "Resumed from epoch {}, best_cls={:.4f}%, best_ret={:.4f}%".format(
                start_epoch, best_valid_cls_acc, best_valid_ret_acc
            )
        )

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    for epoch in range(start_epoch, config["epochs"]):
        print("\nEpoch {}/{}".format(epoch + 1, config["epochs"]))

        train_cls_acc, train_loss = train_epoch(
            model,
            cls_train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            criterion,
        )
        curr_lr = float(optimizer.param_groups[0]["lr"])
        print(
            "Train Cls. Acc {:.04f}% | Train Loss {:.04f} | LR {:.06f}".format(
                train_cls_acc, train_loss, curr_lr
            )
        )

        metrics = {
            "epoch": epoch + 1,
            "train_cls_acc": train_cls_acc,
            "train_loss": train_loss,
            "lr": curr_lr,
        }

        if eval_cls:
            valid_cls_acc, valid_loss = valid_epoch_cls(
                model, cls_val_loader, device, config, criterion
            )
            print("Val Cls. Acc {:.04f}% | Val Loss {:.04f}".format(valid_cls_acc, valid_loss))
            metrics["valid_cls_acc"] = valid_cls_acc
            metrics["valid_loss"] = valid_loss
            scheduler.step(valid_loss)

        valid_ret_acc = valid_epoch_ver(model, ver_val_loader, device, config)
        print("Val Ret. Acc {:.04f}%".format(valid_ret_acc))
        metrics["valid_ret_acc"] = valid_ret_acc

        save_model(
            model, optimizer, scheduler, metrics, epoch,
            os.path.join(config["checkpoint_dir"], "last.pth"),
        )
        print("Saved last.pth")

        if eval_cls and valid_cls_acc >= best_valid_cls_acc:
            best_valid_cls_acc = valid_cls_acc
            save_model(
                model, optimizer, scheduler, metrics, epoch,
                os.path.join(config["checkpoint_dir"], "best_cls.pth"),
            )
            print("Saved best_cls.pth")
        if valid_ret_acc >= best_valid_ret_acc:
            best_valid_ret_acc = valid_ret_acc
            save_model(
                model, optimizer, scheduler, metrics, epoch,
                os.path.join(config["checkpoint_dir"], "best_ret.pth"),
            )
            print("Saved best_ret.pth")

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print("Training done.")


if __name__ == "__main__":
    main()
