"""
Phase 2: Fine-tune a pre-trained FSL-Net using supervised feature importance vectors.

Usage
-----
    python experiments/phase2_finetune_fsl.py --target {logreg, mlp, xgb}

This script:
  - Loads MNIST from `./notebooks/data` (same split and preprocessing as Phase 1).
  - Loads a precomputed, normalized (784,) feature-importance vector from
    `notebooks/feature_importance_classifiers/feature_importance_<Model>.npy`.
  - Fine-tunes only the final parameters of a pre-trained FSL-Net so that its
    shift-based importance aligns with the supervised importance.
  - Saves fine-tuned weights to `models/fsl_finetuned_<target>.pth`.
"""

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fslnet.fslnet import FSLNet


SEED = 42
DATA_ROOT = Path("./notebooks/data")
FI_ROOT = Path("notebooks/feature_importance_classifiers")
MODELS_DIR = Path("models")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mnist_flat(split: str = "train"):
    """
    Load MNIST and return flattened [0,1] tensors and labels.
    """
    assert split in {"train", "test"}

    transform = transforms.ToTensor()  # gives [0,1]

    ds = datasets.MNIST(
        root=DATA_ROOT,
        train=(split == "train"),
        download=True,
        transform=transform,
    )

    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    xs, ys = [], []
    for imgs, labels in loader:
        xs.append(imgs.view(imgs.size(0), -1))  # (B, 784)
        ys.append(labels)
    X = torch.cat(xs, dim=0)  # (N, 784)
    y = torch.cat(ys, dim=0)  # (N,)
    return X, y


def load_target_importance(target: str) -> np.ndarray:
    """
    Load the precomputed supervised feature-importance vector for the chosen model.
    """
    target_map: Dict[str, str] = {
        "logreg": "feature_importance_LogisticRegression.npy",
        "mlp": "feature_importance_MLPClassifier.npy",
        "xgb": "feature_importance_XGBoost.npy",
    }
    fname = target_map[target]
    path = FI_ROOT / fname
    if not path.exists():
        raise FileNotFoundError(f"Importance file not found: {path}")
    imp = np.load(path)
    if imp.shape != (784,):
        raise ValueError(f"Expected importance vector shape (784,), got {imp.shape}")
    return imp.astype(np.float32)


def build_one_vs_all_indices(y: torch.Tensor):
    """
    Precompute indices for one-vs-all splits for each digit c in {0,...,9}.
    Returns dict: c -> (idx_c, idx_not_c), as numpy arrays of indices.
    """
    y_np = y.cpu().numpy()
    indices = {}
    all_idx = np.arange(len(y_np))
    for c in range(10):
        idx_c = np.where(y_np == c)[0]
        idx_not_c = np.where(y_np != c)[0]
        indices[c] = (idx_c, idx_not_c)
    return indices


def fine_tune_fsl(
    target: str,
    epochs: int = 20,
    lr: float = 1e-4,
    max_ref_per_class: int = 1000,
    max_not_ref_per_class: int = 1000,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data and supervision
    # ------------------------------------------------------------------
    print("Loading MNIST train/test splits ...")
    X_train, y_train = load_mnist_flat(split="train")
    X_test, y_test = load_mnist_flat(split="test")

    # Ensure on CPU for indexing; moved to device later per batch.
    X_train_np = X_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()

    train_indices = build_one_vs_all_indices(y_train)
    test_indices = build_one_vs_all_indices(y_test)

    target_importance_np = load_target_importance(target)  # (784,)
    target_importance = torch.tensor(target_importance_np, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Load pre-trained FSL-Net and freeze all but final parameters
    # ------------------------------------------------------------------
    print("Loading pre-trained FSL-Net ...")
    fslnet = FSLNet.from_pretrained(device=str(device))
    fslnet.to(device)
    fslnet.train()

    # Freeze all params, then unfreeze the last few parameter tensors as "final layers".
    for p in fslnet.parameters():
        p.requires_grad = False

    # Unfreeze last K parameter tensors (heuristic for final layers).
    all_params = list(fslnet.parameters())
    K = min(4, len(all_params))
    for p in all_params[-K:]:
        p.requires_grad = True

    trainable_params = [p for p in fslnet.parameters() if p.requires_grad]
    print(f"Number of trainable parameter tensors: {len(trainable_params)}")

    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.MSELoss()

    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def epoch_loss(indices_dict, X_np, desc: str) -> float:
        fslnet.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for c in range(10):
                idx_c, idx_not_c = indices_dict[c]

                # Subsample to control memory
                if len(idx_c) > max_ref_per_class:
                    idx_c_sub = rng.choice(idx_c, size=max_ref_per_class, replace=False)
                else:
                    idx_c_sub = idx_c
                if len(idx_not_c) > max_not_ref_per_class:
                    idx_not_c_sub = rng.choice(idx_not_c, size=max_not_ref_per_class, replace=False)
                else:
                    idx_not_c_sub = idx_not_c

                ref_np = X_np[idx_c_sub]
                que_np = X_np[idx_not_c_sub]

                ref = torch.tensor(ref_np, dtype=torch.float32, device=device)
                que = torch.tensor(que_np, dtype=torch.float32, device=device)

                soft_predictions, _ = fslnet(ref, que)
                P = soft_predictions.squeeze(0)  # (784,)
                loss = criterion(P, target_importance)

                total_loss += float(loss.item())
                count += 1
        avg = total_loss / max(count, 1)
        print(f"{desc} loss (per-class avg): {avg:.6f}")
        return avg

    print("Starting fine-tuning ...")
    for epoch in range(1, epochs + 1):
        fslnet.train()
        running_loss = 0.0
        class_count = 0

        for c in range(10):
            idx_c, idx_not_c = train_indices[c]

            if len(idx_c) > max_ref_per_class:
                idx_c_sub = rng.choice(idx_c, size=max_ref_per_class, replace=False)
            else:
                idx_c_sub = idx_c
            if len(idx_not_c) > max_not_ref_per_class:
                idx_not_c_sub = rng.choice(idx_not_c, size=max_not_ref_per_class, replace=False)
            else:
                idx_not_c_sub = idx_not_c

            ref_np = X_train_np[idx_c_sub]
            que_np = X_train_np[idx_not_c_sub]

            ref = torch.tensor(ref_np, dtype=torch.float32, device=device)
            que = torch.tensor(que_np, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            soft_predictions, _ = fslnet(ref, que)
            P = soft_predictions.squeeze(0)  # (784,)
            loss = criterion(P, target_importance)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            class_count += 1

        avg_train_loss = running_loss / max(class_count, 1)
        print(f"Epoch {epoch:02d}/{epochs} - train loss (per-class avg): {avg_train_loss:.6f}")

    # Final train / test loss (without gradient)
    final_train_loss = epoch_loss(train_indices, X_train_np, desc="Final train")
    final_test_loss = epoch_loss(test_indices, X_test_np, desc="Final test")

    # ------------------------------------------------------------------
    # Save fine-tuned weights
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"fsl_finetuned_{target}.pth"
    torch.save(fslnet.state_dict(), out_path)
    print(f"Saved fine-tuned FSL-Net weights to: {out_path}")
    print(f"Final train loss: {final_train_loss:.6f}, final test loss: {final_test_loss:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Fine-tune FSL-Net using supervised feature importance.")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["logreg", "mlp", "xgb"],
        help="Which classifier's importance vector to align with (logreg, mlp, xgb).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(SEED)
    fine_tune_fsl(target=args.target, epochs=args.epochs, lr=args.lr)


