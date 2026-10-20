"""Keep GitHub Actions on supported runtimes with least-privilege tokens."""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATHS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
PINNED_ACTION_REFS = {
    "actions/cache": "55cc8345863c7cc4c66a329aec7e433d2d1c52a9",
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "astral-sh/setup-uv": "1c37ad07a6a961277cf70c0d37d6f313000f5884",
}


def load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("workflow_path", WORKFLOW_PATHS, ids=lambda path: path.name)
def test_workflow_uses_read_only_repository_permissions(workflow_path: Path):
    workflow = load_workflow(workflow_path)

    assert workflow["permissions"] == {"contents": "read"}


@pytest.mark.parametrize("workflow_path", WORKFLOW_PATHS, ids=lambda path: path.name)
def test_workflow_actions_use_supported_node_24_releases(workflow_path: Path):
    workflow = load_workflow(workflow_path)

    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            action_ref = step.get("uses")
            if action_ref is None:
                continue

            action, separator, ref = action_ref.partition("@")
            assert separator, f"Action is missing a version: {action_ref}"
            assert action in PINNED_ACTION_REFS, f"Unreviewed action: {action_ref}"
            assert ref == PINNED_ACTION_REFS[action], f"Unpinned action version: {action_ref}"
