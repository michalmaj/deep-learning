"""Validate data/registry.yml against the files actually on disk.

For every dataset file listed in the registry, checks that the file exists
and that its SHA-256 checksum matches the recorded value. Exits non-zero on
any missing file or checksum mismatch.
"""

import hashlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "data" / "registry.yml"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    registry = yaml.safe_load(REGISTRY_PATH.read_text()) or {}
    datasets = registry.get("datasets") or []

    errors: list[str] = []
    checked = 0

    for dataset in datasets:
        dataset_id = dataset.get("id", "<missing id>")
        for file_entry in dataset.get("files", []):
            rel_path = file_entry["path"]
            expected_sha256 = file_entry["sha256"]
            file_path = REPO_ROOT / rel_path
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

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"verify_data: {len(errors)} error(s) out of {checked} file(s) checked.")
        return 1

    print(f"verify_data: OK ({checked} file(s) checked, {len(datasets)} dataset(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
