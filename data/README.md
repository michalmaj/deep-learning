# Data policy

This course does not assume that a publicly downloadable dataset may be
redistributed. Before any dataset is used in a lecture or laboratory, it must
be checked and registered here.

## Required checks before adding a dataset

- dataset name
- original author/provider
- canonical source
- license or terms
- redistribution status
- citation requirements

For Kaggle specifically: *available on Kaggle* does not imply
*redistributable* — the dataset-specific terms must be checked.

## Registry

[`registry.yml`](registry.yml) is the machine-readable record of every
dataset used in the course: source, license, redistribution status,
citation, which lecture/lab uses it, and file checksums.

Dataset IDs must be unique lowercase kebab-case values. Lecture and lab
references must point to modules 1-15, and every entry must be used by at
least one module. A `files` entry, when present, must use a repository-relative
path without `..` and a lowercase 64-character SHA-256 digest.

## Verification

```bash
make verify-data
# or
uv run python scripts/verify_data.py
```

validates the complete schema and semantics of every registry entry. It also
checks that each repository-stored file listed under `files` exists and
matches its recorded SHA-256 checksum. Datasets downloaded into ignored
caches are schema-checked here; their download integrity remains the
responsibility of the pinned downloader/checksum mechanism documented in
their registry entry.

## Storage

- If redistribution is allowed: the dataset may be stored in the repository
  directly or via Git LFS (for large binaries).
- If redistribution is not allowed or unclear: the dataset is not committed;
  a reproducible download mechanism is provided instead (with any required
  manual acceptance/authentication step documented).
