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

## Verification

```bash
make verify-data
# or
uv run python scripts/verify_data.py
```

validates that every file listed in `registry.yml` exists locally and
matches its recorded SHA-256 checksum. With an empty registry this passes
trivially.

## Storage

- Downloaded datasets share the repository-level `data_cache/` directory
  across lectures, labs, and languages. Set `DEEP_LEARNING_DATA_CACHE` to an
  absolute path to override that location for every notebook. Module 15's
  preparation script additionally accepts the narrower `MODULE15_DATA_ROOT`,
  which takes precedence when both variables are set.
- If redistribution is allowed: the dataset may be stored in the repository
  directly or via Git LFS (for large binaries).
- If redistribution is not allowed or unclear: the dataset is not committed;
  a reproducible download mechanism is provided instead (with any required
  manual acceptance/authentication step documented).
