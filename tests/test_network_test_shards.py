"""The scheduled network CI manifest must cover every network test exactly once."""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_network_test_shard import (  # noqa: E402
    NETWORK_TEST_SHARDS,
    discover_network_test_files,
    validate_network_test_shards,
)


def test_network_shards_cover_every_marked_test_once():
    assert validate_network_test_shards(REPO_ROOT) == []

    discovered = discover_network_test_files(REPO_ROOT)
    assigned = [path for paths in NETWORK_TEST_SHARDS.values() for path in paths]
    assert len(discovered) == 30
    assert len(assigned) == len(discovered)
    assert set(assigned) == discovered


def test_validator_reports_missing_and_duplicate_assignments():
    duplicate = NETWORK_TEST_SHARDS["fashion"][0]
    incomplete = {
        "first": (duplicate,),
        "second": (duplicate,),
    }

    problems = validate_network_test_shards(REPO_ROOT, incomplete)

    assert any("assigned more than once" in problem for problem in problems)
    assert any("missing from shards" in problem for problem in problems)


def test_validator_reports_missing_and_unmarked_files():
    invalid = {
        **NETWORK_TEST_SHARDS,
        "invalid": (
            "tests/does_not_exist.py",
            "tests/test_network_test_shards.py",
        ),
    }

    problems = validate_network_test_shards(REPO_ROOT, invalid)

    assert any("missing files" in problem for problem in problems)
    assert any("without network tests" in problem for problem in problems)


def test_network_workflow_runs_every_shard_with_an_isolated_cache():
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "network-tests.yml").read_text()
    )
    job = workflow["jobs"]["validate-network-notebooks"]

    assert job["strategy"]["fail-fast"] is False
    assert set(job["strategy"]["matrix"]["shard"]) == set(NETWORK_TEST_SHARDS)

    cache_step = next(step for step in job["steps"] if step.get("uses") == "actions/cache@v4")
    assert "${{ matrix.shard }}" in cache_step["with"]["key"]

    run_step = next(
        step for step in job["steps"] if step.get("name") == "Run network-dependent notebook shard"
    )
    assert run_step["run"] == (
        "uv run python scripts/run_network_test_shard.py ${{ matrix.shard }}"
    )
