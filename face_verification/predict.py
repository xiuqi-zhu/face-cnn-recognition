"""
Generate verification submission CSV from test pairs (similarity scores per pair).
"""
import os
import argparse
import torch
from .config import get_config
from .dataset import create_dataloaders
from .model import Network
from .evaluate import test_epoch_ver
from .checkpoint import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (e.g. best_ret.pth)")
    parser.add_argument("--config", type=str, default=None, help="Override config key=value,key=value")
    parser.add_argument("--output", type=str, default="verification_submission.csv", help="Output CSV path")
    args = parser.parse_args()

    config = get_config()
    if args.config:
        for pair in args.config.split(","):
            k, v = pair.split("=", 1)
            config[k.strip()] = v.strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Network(num_classes=config["num_classes"]).to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        model, _, _, _, _ = load_model(model, args.checkpoint)
        print("Loaded checkpoint:", args.checkpoint)
    else:
        print("No checkpoint given or file missing; using untrained weights (structure only).")

    loaders = create_dataloaders(config)
    ver_test_loader = loaders["ver_test_loader"]

    scores = test_epoch_ver(model, ver_test_loader, device, config)

    with open(args.output, "w") as f:
        f.write("ID,Label\n")
        for i, s in enumerate(scores):
            f.write("{},{}\n".format(i, s))
    print("Wrote {} rows to {}".format(len(scores), args.output))


if __name__ == "__main__":
    main()
