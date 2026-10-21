"""Style checks for course prose that has completed editorial review."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REVIEWED_PROSE_PATHS = (
    "lectures/01_intro/lecture.md",
    "lectures/01_intro/lecture.en.md",
    "lectures/01_intro/examples.ipynb",
    "lectures/01_intro/examples.en.ipynb",
    "labs/01_intro/README.md",
    "labs/01_intro/README.pl.md",
    "labs/01_intro/lab.ipynb",
    "labs/01_intro/lab.en.ipynb",
    "lectures/02_mlp/lecture.md",
    "lectures/02_mlp/lecture.en.md",
    "lectures/02_mlp/examples.ipynb",
    "lectures/02_mlp/examples.en.ipynb",
    "labs/02_mlp/README.md",
    "labs/02_mlp/README.pl.md",
    "labs/02_mlp/lab.ipynb",
    "labs/02_mlp/lab.en.ipynb",
)

FORBIDDEN_DASHES = {
    "\N{EM DASH}": "em dash",
    " -- ": "spaced double hyphen",
}


def _read_prose(path: Path) -> str:
    if path.suffix != ".ipynb":
        return path.read_text()

    notebook = json.loads(path.read_text())
    return "".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )


def test_reviewed_course_prose_does_not_use_long_dashes():
    problems = []

    for relative_path in REVIEWED_PROSE_PATHS:
        prose = _read_prose(REPO_ROOT / relative_path)
        for token, description in FORBIDDEN_DASHES.items():
            if token in prose:
                problems.append(f"{relative_path}: contains {description}")

    assert problems == []
