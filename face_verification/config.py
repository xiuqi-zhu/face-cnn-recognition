"""
Configuration for face classification and verification.
Adjust data paths for your environment (Colab / Kaggle / PSC).
"""
# Colab: "/content/data/hw2p2_puru_aligned"
# Kaggle: "/kaggle/input/11785-hw-2-p-2-face-verification-fall-2025/hw2p2_puru_aligned"
# PSC: "/ocean/projects/cis250019p/mzhang23/TA/HW2P2/hw2p2_data/hw2p2_puru_aligned"

DEFAULT_CONFIG = {
    "batch_size": 1024,
    "lr": 1e-3,
    "epochs": 50,
    "num_classes": 8631,
    "cls_data_dir": "/content/data/hw2p2_puru_aligned/cls_data",
    "ver_data_dir": "/content/data/hw2p2_puru_aligned/ver_data",
    "val_pairs_file": "/content/data/hw2p2_puru_aligned/val_pairs.txt",
    "test_pairs_file": "/content/data/hw2p2_puru_aligned/test_pairs.txt",
    "checkpoint_dir": "/content/data/checkpoint",
    "augument": True,
    "eval_cls": True,
    "use_wandb": False,
}


def get_config(**overrides):
    """Return config dict with optional overrides applied."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg
