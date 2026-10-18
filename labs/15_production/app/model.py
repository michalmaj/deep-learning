"""Artifact loading and inference logic.

`load_artifact` is given (scaffold). `preprocess_image_bytes` and `predict`
are your task -- see README.md sections 2.
"""

import json

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

_PREPROCESS = ResNet18_Weights.IMAGENET1K_V1.transforms()


class LoadedModel:
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata

    @property
    def class_names(self):
        return self.metadata["class_names"]

    @property
    def model_version(self):
        return self.metadata["model_version"]

    @property
    def input_size(self):
        return self.metadata["input_size"]


def load_artifact(artifact_dir: str) -> LoadedModel:
    """Build the architecture fresh and load weights + metadata -- never
    `torch.save(model)` the whole object (see lecture.md)."""
    with open(f"{artifact_dir}/metadata.json") as f:
        metadata = json.load(f)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, metadata["num_classes"])
    state = torch.load(f"{artifact_dir}/weights.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return LoadedModel(model, metadata)


def preprocess_image_bytes(data: bytes) -> torch.Tensor:
    # TODO:
    # 1. decode `data` into a PIL Image (io.BytesIO + Image.open), convert to RGB
    # 2. apply _PREPROCESS (ResNet18_Weights.IMAGENET1K_V1.transforms()) --
    #    do NOT hand-roll resize/crop/normalize yourself, see README section 8
    #    for exactly why that goes wrong
    # 3. add a batch dimension (unsqueeze(0)) and return
    raise NotImplementedError("TODO: preprocess_image_bytes")


def predict(loaded: LoadedModel, image_bytes: bytes) -> dict:
    # TODO:
    # 1. x = preprocess_image_bytes(image_bytes)
    # 2. with torch.no_grad(): logits = loaded.model(x); probs = softmax(logits, dim=1)
    # 3. pred_idx = probs.argmax(1).item()
    # 4. return {"predicted_class": ..., "confidence": ..., "model_version": ...}
    raise NotImplementedError("TODO: predict")
