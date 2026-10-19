"""Foundation-level checks: project metadata, registries, repo hygiene."""

import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_is_valid_and_pinned():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert data["project"]["name"] == "deep-learning"
    assert data["project"]["version"] == "1.0.0"
    assert data["project"]["requires-python"].startswith(">=3.12")


def test_project_version_matches_lockfile():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    locked_project = next(
        package for package in lock["package"] if package["name"] == project["name"]
    )
    assert locked_project["version"] == project["version"]


def test_data_registry_is_valid_yaml_with_expected_schema():
    data = yaml.safe_load((REPO_ROOT / "data" / "registry.yml").read_text())
    assert "datasets" in data
    assert isinstance(data["datasets"], list)


def test_third_party_sources_is_valid_yaml_with_expected_schema():
    data = yaml.safe_load((REPO_ROOT / "third_party" / "SOURCES.yml").read_text())
    assert "assets" in data
    assert isinstance(data["assets"], list)


def test_verify_data_script_passes_on_current_registry():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_data.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_decisions_directory_is_not_tracked_by_git():
    result = subprocess.run(
        ["git", "ls-files", "decisions"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""


def test_gitignore_excludes_decisions():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "decisions/" in gitignore
