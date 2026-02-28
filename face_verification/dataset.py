"""
Datasets and transforms for face classification and verification.
"""
import os
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from PIL import Image
from tqdm.auto import tqdm
import numpy as np


def create_transforms(image_size: int = 112, augment: bool = True) -> T.Compose:
    """Build transform pipeline for face images (resize, normalize, optional augmentation)."""
    transform_list = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.ToDtype(torch.float32, scale=True),
    ]
    if augment:
        strong_augmentations = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ])
        transform_list.append(T.RandomApply([strong_augmentations], p=0.8))
    transform_list.extend([
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return T.Compose(transform_list)


class ImageDataset(torch.utils.data.Dataset):
    """Dataset of single face images with identity labels (for classification)."""

    def __init__(self, root, transform, num_classes=8631):
        self.root = root
        self.labels_file = os.path.join(self.root, "labels.txt")
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = set()

        with open(self.labels_file, "r") as f:
            lines = f.readlines()
        lines = sorted(lines, key=lambda x: int(x.strip().split(" ")[-1]))
        all_labels = sorted(set(int(line.strip().split(" ")[1]) for line in lines))
        selected_classes = set(all_labels[:num_classes]) if num_classes else set(all_labels)

        for line in tqdm(lines, desc="Loading dataset"):
            img_path, label = line.strip().split(" ")
            label = int(label)
            if label in selected_classes:
                self.image_paths.append(os.path.join(self.root, "images", img_path))
                self.labels.append(label)
                self.classes.add(label)

        assert len(self.image_paths) == len(self.labels)
        self.classes = sorted(self.classes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]


class ImagePairDataset(torch.utils.data.Dataset):
    """Dataset of image pairs with binary match labels (for verification validation)."""

    def __init__(self, root, pairs_file, transform):
        self.root = root
        self.transform = transform
        self.matches = []
        self.image1_list = []
        self.image2_list = []

        with open(pairs_file, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="Loading image pairs"):
            img_path1, img_path2, match = line.strip().split(" ")
            img1 = Image.open(os.path.join(self.root, img_path1)).convert("RGB")
            img2 = Image.open(os.path.join(self.root, img_path2)).convert("RGB")
            self.image1_list.append(img1)
            self.image2_list.append(img2)
            self.matches.append(int(match))
        assert len(self.image1_list) == len(self.image2_list) == len(self.matches)

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, idx):
        img1 = self.transform(self.image1_list[idx])
        img2 = self.transform(self.image2_list[idx])
        return img1, img2, self.matches[idx]


class TestImagePairDataset(torch.utils.data.Dataset):
    """Dataset of image pairs without labels (for verification test / submission)."""

    def __init__(self, root, pairs_file, transform):
        self.root = root
        self.transform = transform
        self.image1_list = []
        self.image2_list = []

        with open(pairs_file, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="Loading image pairs"):
            img_path1, img_path2 = line.strip().split(" ")
            img1 = Image.open(os.path.join(self.root, img_path1)).convert("RGB")
            img2 = Image.open(os.path.join(self.root, img_path2)).convert("RGB")
            self.image1_list.append(img1)
            self.image2_list.append(img2)
        assert len(self.image1_list) == len(self.image2_list)

    def __len__(self):
        return len(self.image1_list)

    def __getitem__(self, idx):
        return self.transform(self.image1_list[idx]), self.transform(self.image2_list[idx])


def rand_bbox(size, lam):
    """Sample random bounding box for CutMix (lambda controls area ratio)."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_collate_fn(batch, alpha=1.0):
    """Collate batch with CutMix: mix two images and interpolate labels by area."""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    indices = torch.randperm(imgs.size(0))
    shuffled_imgs = imgs[indices]
    shuffled_labels = labels[indices]
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
    imgs[:, :, bby1:bby2, bbx1:bbx2] = shuffled_imgs[:, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2)))
    targets = (labels, shuffled_labels, lam)
    return imgs, targets


def cutmix_collate(batch):
    """Default CutMix collate with alpha=1.0."""
    return cutmix_collate_fn(batch, alpha=1.0)


def create_dataloaders(config):
    """Create train/val/test dataloaders for classification and verification."""
    train_tf = create_transforms(image_size=112, augment=config.get("augument", True))
    val_tf = create_transforms(image_size=112, augment=False)
    num_classes = config.get("num_classes", 8631)
    cls_data = config["cls_data_dir"]
    ver_data = config["ver_data_dir"]

    cls_train = ImageDataset(
        root=os.path.join(cls_data, "train"),
        transform=train_tf,
        num_classes=num_classes,
    )
    cls_val = ImageDataset(
        root=os.path.join(cls_data, "dev"),
        transform=val_tf,
        num_classes=num_classes,
    )
    cls_test = ImageDataset(
        root=os.path.join(cls_data, "test"),
        transform=val_tf,
        num_classes=num_classes,
    )
    assert cls_train.classes == cls_val.classes == cls_test.classes

    cls_train_loader = DataLoader(
        cls_train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=cutmix_collate,
    )
    cls_val_loader = DataLoader(
        cls_val,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    cls_test_loader = DataLoader(
        cls_test,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    ver_val = ImagePairDataset(
        root=ver_data,
        pairs_file=config["val_pairs_file"],
        transform=val_tf,
    )
    ver_test = TestImagePairDataset(
        root=ver_data,
        pairs_file=config["test_pairs_file"],
        transform=val_tf,
    )
    ver_val_loader = DataLoader(
        ver_val,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    ver_test_loader = DataLoader(
        ver_test,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return {
        "cls_train_loader": cls_train_loader,
        "cls_val_loader": cls_val_loader,
        "cls_test_loader": cls_test_loader,
        "ver_val_loader": ver_val_loader,
        "ver_test_loader": ver_test_loader,
    }
