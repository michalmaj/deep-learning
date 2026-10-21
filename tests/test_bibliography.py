"""Checks for the course-wide BibTeX bibliography."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BIBLIOGRAPHY_PATH = REPO_ROOT / "references" / "references.bib"
MODULE_HEADING_RE = re.compile(r"^% --- Module (\d{2}): .+ ---$", re.MULTILINE)
ENTRY_RE = re.compile(r"^@\w+\{([^,\s]+),", re.MULTILINE)


def test_bibliography_has_a_section_for_every_course_module():
    bibliography = BIBLIOGRAPHY_PATH.read_text()
    module_numbers = MODULE_HEADING_RE.findall(bibliography)

    assert module_numbers == [f"{number:02}" for number in range(1, 16)]


def test_every_bibliography_section_contains_at_least_one_entry():
    bibliography = BIBLIOGRAPHY_PATH.read_text()
    headings = list(MODULE_HEADING_RE.finditer(bibliography))
    empty_sections = []

    for index, heading in enumerate(headings):
        section_end = headings[index + 1].start() if index + 1 < len(headings) else None
        section = bibliography[heading.end() : section_end]
        if ENTRY_RE.search(section) is None:
            empty_sections.append(heading.group(1))

    assert empty_sections == []


def test_bibliography_keys_are_unique():
    bibliography = BIBLIOGRAPHY_PATH.read_text()
    keys = ENTRY_RE.findall(bibliography)

    assert len(keys) == len(set(keys))
