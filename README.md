# 11785 HW2P2: Face Recognition and Verification

Face classification and verification with CNN (CMU 11-785).

## Project Structure

```
.
├── README.md
├── requirements.txt
├── acknowledgement.txt
├── face_verification/       # Face classification + verification (PyTorch)
│   ├── config.py            # Config and data paths
│   ├── dataset.py           # Datasets and augmentation
│   ├── model.py             # ConvNeXt backbone and head
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Validation and verification eval
│   ├── metrics.py           # ACC, EER, AUC, etc.
│   ├── checkpoint.py        # Save/load checkpoints
│   ├── predict.py           # Test-set prediction and submission CSV
│   └── main.py              # Entry script
├── mytorch/                 # NumPy-based deep learning components
│   ├── nn/                  # Conv, BN, activation, pool, linear, loss, etc.
│   └── flatten.py
└── models/                  # Models built on mytorch (CNN-1D, MLP, ResNet)
    ├── layers.py            # Re-exports for mlp (Linear, ReLU, etc.)
    ├── cnn.py
    ├── mlp.py
    ├── mlp_scan.py
    └── resnet.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Face Verification (PyTorch)

- **Train**: Edit data paths in `face_verification/config.py`, then:
  ```bash
  python -m face_verification.main
  ```
- **Predict & submit**: Load checkpoint and write similarity scores to CSV:
  ```bash
  python -m face_verification.predict --checkpoint path/to/best_ret.pth --output verification_submission.csv
  ```

Adjust paths in `face_verification/config.py` for your environment (Colab / Kaggle / PSC).

## MyTorch and models

- `mytorch/`: NumPy implementations of conv, BN, activation, pool, linear, loss, etc.
- `models/cnn.py`: 1D CNN (uses mytorch).
- `models/resnet.py`: 2D ResNet (uses mytorch Conv2d, BatchNorm2d, etc.).

## Licence and acknowledgement

See `acknowledgement.txt`.
