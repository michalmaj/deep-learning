"""Shared metadata contract for the module 15 model artifact."""

from typing import Any

ARCHITECTURE = "resnet18"
INPUT_SIZE = [224, 224]
INPUT_CHANNELS = 3
PREPROCESSING = "ResNet18_Weights.IMAGENET1K_V1.transforms()"


def validate_artifact_metadata(metadata: Any) -> dict[str, Any]:
    """Return valid metadata or raise one error listing every contract violation."""
    if not isinstance(metadata, dict):
        raise ValueError("invalid artifact metadata: root must be an object")

    errors = []

    model_version = metadata.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        errors.append("model_version must be a non-empty string")

    if metadata.get("architecture") != ARCHITECTURE:
        errors.append(f"architecture must be {ARCHITECTURE!r}")

    num_classes = metadata.get("num_classes")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
        errors.append("num_classes must be a positive integer")

    class_names = metadata.get("class_names")
    valid_class_names = isinstance(class_names, list) and all(
        isinstance(name, str) and name.strip() for name in class_names
    )
    if not valid_class_names:
        errors.append("class_names must be a list of non-empty strings")
    else:
        if len(class_names) != len(set(class_names)):
            errors.append("class_names must be unique")
        if isinstance(num_classes, int) and not isinstance(num_classes, bool):
            if len(class_names) != num_classes:
                errors.append("class_names length must match num_classes")

    if metadata.get("input_size") != INPUT_SIZE:
        errors.append(f"input_size must be {INPUT_SIZE!r}")
    if metadata.get("input_channels") != INPUT_CHANNELS:
        errors.append(f"input_channels must be {INPUT_CHANNELS}")
    if metadata.get("preprocessing") != PREPROCESSING:
        errors.append(f"preprocessing must be {PREPROCESSING!r}")

    if errors:
        raise ValueError("invalid artifact metadata: " + "; ".join(errors))
    return metadata
