"""Keep GitHub Actions on supported runtimes with least-privilege tokens."""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATHS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
SUPPORTED_ACTION_REFS = {
    "actions/cache": "v6",
    "actions/checkout": "v7",
    "astral-sh/setup-uv": "v10",
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
            assert action in SUPPORTED_ACTION_REFS, f"Unreviewed action: {action_ref}"
            assert ref == SUPPORTED_ACTION_REFS[action], f"Unsupported action version: {action_ref}"
