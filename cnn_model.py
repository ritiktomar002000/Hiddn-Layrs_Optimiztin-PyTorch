import math
import pickle
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    data_path: str = "data/data_dog_nondog.pickle"
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0  # keep 0 for Windows reliability
    seed: int = 42
    save_path: str = "cnn_dog_nondog_best.pt"


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_image_shape_from_flat(n_features: int) -> Tuple[int, int, int]:
    """
    Tries to infer (C, H, W) from a flat vector length.
    Assumes RGB if divisible by 3; otherwise assumes single-channel.
    """
    if n_features % 3 == 0:
        c = 3
        hw = n_features // 3
    else:
        c = 1
        hw = n_features

    side = int(math.isqrt(hw))
    if side * side != hw:
        raise ValueError(
            f"Cannot infer square image from n_features={n_features}. "
            f"Expected C*H*W where H==W and H*W is perfect square. Got hw={hw}."
        )
    return c, side, side


def to_image_tensor(X: np.ndarray) -> torch.Tensor:
    """
    Accepts X in common shapes:
      - (n_features, m)  [Andrew Ng style]
      - (m, n_features)
      - (m, H, W, C) or (m, C, H, W)

    Returns torch.Tensor float32 in shape (m, C, H, W), scaled to [0,1] if needed.
    """
    X = np.asarray(X)

    # Case: (n_features, m) -> (m, n_features)
    if X.ndim == 2:
        if X.shape[0] < X.shape[1]:
            # could be (n_features, m)
            # but ambiguous; we’ll treat it as (n_features, m) if it looks like that
            # Typical: n_features = 12288, m ~ 200
            pass

        # Decide orientation by checking which dimension is "features-like"
        # Heuristic: features dimension usually > 1000 for flattened images
        if X.shape[0] > 1000 and X.shape[1] <= 10000:
            # (n_features, m)
            X = X.T  # (m, n_features)
        # else assume already (m, n_features)

        m, n_features = X.shape
        c, h, w = infer_image_shape_from_flat(n_features)
        X = X.reshape(m, c, h, w)

    # Case: images already present
    elif X.ndim == 4:
        # (m, H, W, C) -> (m, C, H, W)
        if X.shape[-1] in (1, 3):
            X = np.transpose(X, (0, 3, 1, 2))
        # else assume already (m, C, H, W)
    else:
        raise ValueError(f"Unsupported X shape: {X.shape}")

    X = X.astype(np.float32)

    # Normalize if it looks like 0..255
    if X.max() > 1.5:
        X = X / 255.0

    return torch.from_numpy(X)


def to_label_tensor(Y: np.ndarray) -> torch.Tensor:
    """
    Accepts Y shapes:
      - (1, m)
      - (m, 1)
      - (m,)
    Returns float32 tensor shape (m, 1)
    """
    Y = np.asarray(Y)
    if Y.ndim == 2:
        if Y.shape[0] == 1:
            Y = Y.T
    elif Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported Y shape: {Y.shape}")

    return torch.from_numpy(Y.astype(np.float32))


# -------------------------
# Dataset
# -------------------------
class DogNonDogDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# -------------------------
# CNN Model
# -------------------------
class SimpleCNN(nn.Module):
    """
    Compact CNN for small image datasets.
    Works well for 64x64 RGB flattened datasets (common in classic dog/cat demos).
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits  # raw logits


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds.eq(y).float().mean().item()) * 100.0


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits.detach(), y)
        n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    data = load_pickle(cfg.data_path)

    X_train = to_image_tensor(data["X_train"])
    Y_train = to_label_tensor(data["Y_train"])
    X_test = to_image_tensor(data["X_test"])
    Y_test = to_label_tensor(data["Y_test"])

    in_channels = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DogNonDogDataset(X_train, Y_train)
    test_ds = DogNonDogDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = SimpleCNN(in_channels=in_channels).to(device)

    # BCEWithLogitsLoss expects raw logits (no sigmoid in model output)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_test_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "best_test_acc": best_test_acc,
                    "config": cfg.__dict__,
                },
                cfg.save_path,
            )

    print(f"\n✅ Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"💾 Saved best model to: {cfg.save_path}")


if __name__ == "__main__":
    main()
