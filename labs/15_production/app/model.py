"""Artifact loading and inference logic.

`load_artifact` is given (scaffold). `preprocess_image_bytes` and `predict`
are your task -- see README.md sections 2.
"""

import io
import json
import warnings

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision.models import ResNet18_Weights, resnet18

from .artifact_contract import validate_artifact_metadata

_PREPROCESS = ResNet18_Weights.IMAGENET1K_V1.transforms()
MAX_DECODED_IMAGE_PIXELS = 25_000_000


class InvalidImageError(ValueError):
    """The uploaded bytes cannot be decoded as a supported image."""


class ImageTooLargeError(InvalidImageError):
    """The decoded image exceeds the service's explicit pixel budget."""


def decode_image_bytes(
    data: bytes,
    *,
    max_pixels: int = MAX_DECODED_IMAGE_PIXELS,
) -> Image.Image:
    """Decode one RGB image while bounding decompressed pixel count.

    A compressed-byte limit alone does not bound memory use: a small PNG or
    TIFF can expand into hundreds of millions of pixels. Pillow exposes the
    dimensions before decoding the complete pixel buffer, so enforce the
    service limit before ``convert`` materializes RGB data.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if width * height > max_pixels:
                    raise ImageTooLargeError(f"decoded image exceeds the {max_pixels}-pixel limit")
                return image.convert("RGB")
    except ImageTooLargeError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageTooLargeError("decoded image exceeds Pillow's safety limit") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("could not decode image") from exc


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
        metadata = validate_artifact_metadata(json.load(f))

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, metadata["num_classes"])
    state = torch.load(f"{artifact_dir}/weights.pt", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return LoadedModel(model, metadata)


def preprocess_image_bytes(data: bytes) -> torch.Tensor:
    # TODO:
    # 1. call `decode_image_bytes(data)` -- it converts to RGB and rejects
    #    compressed images whose decoded pixel buffer would be too large
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
