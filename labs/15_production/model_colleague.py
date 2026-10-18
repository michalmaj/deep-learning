"""Diagnostic task: a colleague's inference code with a real training-
serving skew bug.

This module is a SEPARATE, deliberately buggy implementation of
preprocessing + prediction -- NOT your own `app/model.py`. It "works"
(runs without raising) but uses a different preprocessing pipeline than
the one the model was actually trained/evaluated with.

Your task (see README.md section 8): diagnose the discrepancy, measure its
impact, and confirm that your own `app/model.py` does NOT have this bug.
"""

import io

import torch
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image_bytes_colleague(data: bytes) -> torch.Tensor:
    """The colleague's version: resizes directly to 224x224 (no intermediate
    256-then-center-crop step) and manually re-implements normalization,
    instead of using ResNet18_Weights.IMAGENET1K_V1.transforms()."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((224, 224))
    tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).view(224, 224, 3) / 255.0
    tensor = tensor.permute(2, 0, 1)
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0)


def predict_colleague(loaded, image_bytes: bytes) -> dict:
    import torch.nn.functional as F

    x = preprocess_image_bytes_colleague(image_bytes)
    with torch.no_grad():
        logits = loaded.model(x)
        probs = F.softmax(logits, dim=1)
    pred_idx = probs.argmax(1).item()
    return {
        "predicted_class": loaded.class_names[pred_idx],
        "confidence": probs[0, pred_idx].item(),
        "model_version": loaded.model_version,
    }
