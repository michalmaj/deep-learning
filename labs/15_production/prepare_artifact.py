"""Train the EuroSAT head and save the model artifact (scaffold, given).

Run once, locally: `uv run python labs/15_production/prepare_artifact.py`.
Writes to `artifacts/` (gitignored), never committed. Downloads EuroSAT +
ResNet18 IMAGENET1K_V1 weights on first run (see data/registry.yml).
"""

import json
import os
import time

import torch
import torch.nn as nn
import torchvision.datasets as tvd
from torchvision.models import ResNet18_Weights, resnet18

ARTIFACT_DIR = os.environ.get("MODULE15_ARTIFACT_DIR", "artifacts")
DATA_ROOT = os.environ.get("MODULE15_DATA_ROOT", "data_cache")


def load_subset(full_ds, indices):
    X = torch.stack([full_ds[i][0] for i in indices])
    y = torch.tensor([full_ds[i][1] for i in indices])
    return X, y


def extract_features(model, X, batch_size=64):
    feats = []
    orig_fc = model.fc
    model.fc = nn.Identity()
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            feats.append(model(X[i : i + batch_size]))
    model.fc = orig_fc
    return torch.cat(feats, dim=0)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)

    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    t0 = time.time()
    full_ds = tvd.EuroSAT(root=DATA_ROOT, download=True, transform=preprocess)
    print(f"dataset load time: {time.time() - t0:.2f}s, n={len(full_ds)}")
    class_names = full_ds.classes

    perm = torch.randperm(len(full_ds))
    train_idx, val_idx, test_idx = perm[:1800], perm[1800:2200], perm[2200:2600]

    t0 = time.time()
    X_train, y_train = load_subset(full_ds, train_idx.tolist())
    X_val, y_val = load_subset(full_ds, val_idx.tolist())
    X_test, y_test = load_subset(full_ds, test_idx.tolist())
    print(f"prep time: {time.time() - t0:.2f}s")

    backbone = resnet18(weights=weights)
    for p in backbone.parameters():
        p.requires_grad = False
    n_classes = len(class_names)
    backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)
    backbone.eval()

    t0 = time.time()
    feat_train = extract_features(backbone, X_train)
    feat_val = extract_features(backbone, X_val)
    feat_test = extract_features(backbone, X_test)
    print(f"feature extraction time: {time.time() - t0:.2f}s")

    torch.manual_seed(0)
    head = nn.Linear(feat_train.shape[1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(30):
        perm_h = torch.randperm(feat_train.shape[0])
        for i in range(0, feat_train.shape[0], 64):
            idx = perm_h[i : i + 64]
            opt.zero_grad()
            out = head(feat_train[idx])
            loss = loss_fn(out, y_train[idx])
            loss.backward()
            opt.step()
    print(f"head train time: {time.time() - t0:.2f}s")

    backbone.fc = head
    backbone.eval()
    with torch.no_grad():
        val_acc = (backbone.fc(feat_val).argmax(1) == y_val).float().mean().item()
        test_acc = (backbone.fc(feat_test).argmax(1) == y_test).float().mean().item()
    print(f"val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(backbone.state_dict(), f"{ARTIFACT_DIR}/weights.pt")
    metadata = {
        "model_version": "0.1.0",
        "architecture": "resnet18",
        "num_classes": n_classes,
        "class_names": class_names,
        "input_size": [224, 224],
        "input_channels": 3,
        "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
        "torch_version": torch.__version__,
        "trained_on": "EuroSAT subset (1800 train / 400 val / 400 test)",
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }
    with open(f"{ARTIFACT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"artifact saved to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
