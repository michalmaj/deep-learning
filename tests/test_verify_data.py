"""Focused tests for the dataset registry validator."""

import copy
import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_data import validate_registry, verify_registered_files  # noqa: E402


def _dataset(**overrides):
    dataset = {
        "id": "example-dataset",
        "name": "Example Dataset",
        "source": {
            "provider": "Example Provider",
            "author": "Example Author",
            "url": "https://example.com/dataset",
        },
        "license": {
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
        "redistribution": {"allowed": True, "notes": "Redistribution is allowed."},
        "citation": {"text": "Example citation.", "bibtex": "@dataset{example}"},
        "course_usage": {"lectures": [1], "labs": []},
    }
    dataset.update(overrides)
    return dataset


def _registry(*datasets):
    return {"datasets": list(datasets) or [_dataset()]}


@pytest.mark.parametrize(
    "registry, expected_fragment",
    [
        (None, "root must be a mapping"),
        ({}, "must contain a 'datasets' list"),
        ({"datasets": []}, "must contain at least one dataset"),
        ({"datasets": ["not-a-mapping"]}, "must be a mapping"),
        (_registry(_dataset(id="Bad ID")), "lowercase kebab-case"),
        (_registry(_dataset(name="")), "name must be a non-empty string"),
        (
            _registry(_dataset(redistribution={"allowed": "yes", "notes": "documented"})),
            "redistribution.allowed must be a boolean",
        ),
        (
            _registry(_dataset(course_usage={"lectures": [], "labs": []})),
            "must reference at least one module",
        ),
        (
            _registry(_dataset(course_usage={"lectures": [16], "labs": []})),
            "must list module numbers 1-15",
        ),
    ],
)
def test_invalid_registry_shapes_are_reported(registry, expected_fragment):
    assert any(expected_fragment in error for error in validate_registry(registry))


def test_duplicate_dataset_ids_are_reported():
    duplicate = copy.deepcopy(_dataset())
    errors = validate_registry(_registry(_dataset(), duplicate))
    assert any("duplicate dataset id" in error for error in errors)


def test_valid_registry_without_repository_files_passes_schema_validation():
    assert validate_registry(_registry()) == []


@pytest.mark.parametrize(
    "file_entry, expected_fragment",
    [
        ({"path": "../outside.bin", "sha256": "0" * 64}, "stay within the repository"),
        ({"path": "/tmp/outside.bin", "sha256": "0" * 64}, "stay within the repository"),
        ({"path": "data/example.bin", "sha256": "not-a-hash"}, "64 lowercase hexadecimal"),
    ],
)
def test_unsafe_or_invalid_file_entries_are_reported(file_entry, expected_fragment):
    errors = validate_registry(_registry(_dataset(files=[file_entry])))
    assert any(expected_fragment in error for error in errors)


def test_duplicate_repository_paths_are_reported_across_datasets():
    file_entry = {"path": "data/shared.bin", "sha256": "0" * 64}
    second_dataset = _dataset(id="second-dataset", files=[copy.deepcopy(file_entry)])
    errors = validate_registry(
        _registry(_dataset(files=[copy.deepcopy(file_entry)]), second_dataset)
    )
    assert any("duplicates another registry entry" in error for error in errors)


def test_missing_registered_file_is_reported(tmp_path):
    registry = _registry(
        _dataset(files=[{"path": "data/missing.bin", "sha256": "0" * 64}])
    )
    errors, checked = verify_registered_files(registry, tmp_path)
    assert checked == 1
    assert errors == ["[example-dataset] missing file: data/missing.bin"]


def test_registered_file_checksum_is_verified(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = b"registered test payload"
    (data_dir / "example.bin").write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()
    registry = _registry(
        _dataset(files=[{"path": "data/example.bin", "sha256": checksum}])
    )

    assert validate_registry(registry) == []
    assert verify_registered_files(registry, tmp_path) == ([], 1)

    registry["datasets"][0]["files"][0]["sha256"] = "0" * 64
    errors, checked = verify_registered_files(registry, tmp_path)
    assert checked == 1
    assert any("checksum mismatch" in error for error in errors)
