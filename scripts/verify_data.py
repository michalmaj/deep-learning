"""Validate the dataset registry schema and any repository-stored files."""

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "data" / "registry.yml"
DATASET_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_TEXT_PATHS = (
    ("name",),
    ("source", "provider"),
    ("source", "author"),
    ("source", "url"),
    ("license", "name"),
    ("license", "url"),
    ("redistribution", "notes"),
    ("citation", "text"),
    ("citation", "bibtex"),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_value(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_registry(registry: Any) -> list[str]:
    """Return schema and semantic problems without touching the filesystem."""
    if not isinstance(registry, dict):
        return ["registry root must be a mapping"]

    datasets = registry.get("datasets")
    if not isinstance(datasets, list):
        return ["registry must contain a 'datasets' list"]

    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()

    for index, dataset in enumerate(datasets):
        location = f"datasets[{index}]"
        if not isinstance(dataset, dict):
            errors.append(f"{location} must be a mapping")
            continue

        dataset_id = dataset.get("id")
        if not isinstance(dataset_id, str) or not DATASET_ID_PATTERN.fullmatch(dataset_id):
            errors.append(f"{location}.id must be a lowercase kebab-case identifier")
            label = location
        else:
            label = f"[{dataset_id}]"
            if dataset_id in seen_ids:
                errors.append(f"{label} duplicate dataset id")
            seen_ids.add(dataset_id)

        for field_path in REQUIRED_TEXT_PATHS:
            value = _nested_value(dataset, field_path)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label} {'.'.join(field_path)} must be a non-empty string")

        allowed = _nested_value(dataset, ("redistribution", "allowed"))
        if not isinstance(allowed, bool):
            errors.append(f"{label} redistribution.allowed must be a boolean")

        course_usage = dataset.get("course_usage")
        if not isinstance(course_usage, dict):
            errors.append(f"{label} course_usage must be a mapping")
        else:
            usage_count = 0
            for kind in ("lectures", "labs"):
                modules = course_usage.get(kind)
                if not isinstance(modules, list) or any(
                    not isinstance(module, int) or isinstance(module, bool) or not 1 <= module <= 15
                    for module in modules or []
                ):
                    errors.append(f"{label} course_usage.{kind} must list module numbers 1-15")
                else:
                    usage_count += len(modules)
            if usage_count == 0:
                errors.append(f"{label} course_usage must reference at least one module")

        files = dataset.get("files", [])
        if not isinstance(files, list):
            errors.append(f"{label} files must be a list when present")
            continue

        for file_index, file_entry in enumerate(files):
            file_label = f"{label} files[{file_index}]"
            if not isinstance(file_entry, dict):
                errors.append(f"{file_label} must be a mapping")
                continue

            rel_path = file_entry.get("path")
            if not isinstance(rel_path, str) or not rel_path:
                errors.append(f"{file_label}.path must be a non-empty string")
            else:
                path = Path(rel_path)
                if path.is_absolute() or ".." in path.parts:
                    errors.append(f"{file_label}.path must stay within the repository")
                if rel_path in seen_paths:
                    errors.append(f"{file_label}.path duplicates another registry entry")
                seen_paths.add(rel_path)

            expected_sha256 = file_entry.get("sha256")
            if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(
                expected_sha256
            ):
                errors.append(f"{file_label}.sha256 must be 64 lowercase hexadecimal characters")

    return errors


def verify_registered_files(registry: dict[str, Any], repo_root: Path) -> tuple[list[str], int]:
    """Check existence and SHA-256 for file entries after schema validation."""
    errors: list[str] = []
    checked = 0

    for dataset in registry["datasets"]:
        dataset_id = dataset["id"]
        for file_entry in dataset.get("files", []):
            rel_path = file_entry["path"]
            expected_sha256 = file_entry["sha256"]
            file_path = repo_root / rel_path
            checked += 1

            if not file_path.is_file():
                errors.append(f"[{dataset_id}] missing file: {rel_path}")
                continue

            actual_sha256 = sha256_of(file_path)
            if actual_sha256 != expected_sha256:
                errors.append(
                    f"[{dataset_id}] checksum mismatch for {rel_path}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )

    return errors, checked


def main() -> int:
    registry = yaml.safe_load(REGISTRY_PATH.read_text())
    schema_errors = validate_registry(registry)
    if schema_errors:
        for error in schema_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"verify_data: {len(schema_errors)} registry schema error(s).")
        return 1

    errors, checked = verify_registered_files(registry, REPO_ROOT)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"verify_data: {len(errors)} error(s) out of {checked} file(s) checked.")
        return 1

    print(
        f"verify_data: OK (schema valid for {len(registry['datasets'])} dataset(s); "
        f"{checked} repository file(s) checked)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
