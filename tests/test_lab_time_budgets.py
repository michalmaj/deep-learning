"""Keep the published two-hour lab contracts internally consistent."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LABS = [
    "05_generalization",
    "06_cnn",
    "07_transfer_learning",
    "08_embeddings",
    "09_rnn",
]


@pytest.mark.parametrize("module", LABS)
@pytest.mark.parametrize("readme_name", ["README.md", "README.pl.md"])
def test_required_lab_sections_fit_120_minutes_and_include_conclusions(module, readme_name):
    text = (REPO_ROOT / "labs" / module / readme_name).read_text()

    minutes = [int(value) for value in re.findall(r"\|[^|\n]+\|\s*(\d+)\s*min\s*\|", text)]
    assert sum(minutes) == 120, f"{module}/{readme_name}: {sum(minutes)} min"

    conclusion_match = re.search(r"^## (\d+)\. (?:Conclusions|Wnioski)", text, re.MULTILINE)
    assert conclusion_match is not None
    conclusion_section = int(conclusion_match.group(1))

    core_match = re.search(
        r"(?:Sections|Sekcje) (\d+)[-–](\d+)[^\n]*(?:core|required|rdzeń|obowiązk)",
        text,
    )
    assert core_match is not None
    assert int(core_match.group(2)) == conclusion_section
