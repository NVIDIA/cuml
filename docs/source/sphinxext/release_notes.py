# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the repository changelog as the Sphinx release-notes page."""

from __future__ import annotations

import re
from pathlib import Path

_CHANGELOG = Path(__file__).resolve().parents[3] / "CHANGELOG.md"
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_RELEASE = re.compile(
    r"cuml\s+(?:v?\d+(?:\.\d+)+(?:[^\s]*)?|unreleased)\b", re.I
)
_TITLE = re.compile(r"release notes", re.I)


def render_release_notes(
    changelog: str, policy_url: str = "deprecation_policy.html"
) -> str:
    """Add the page introduction and normalize historic changelog headings."""
    lines = [
        "# Release notes",
        "",
        (
            "See the [compatibility and deprecation policy]"
            f"({policy_url}) for cuML's API stability guarantees."
        ),
        "",
    ]
    source_lines = changelog.splitlines()
    headings = [
        (len(match.group(1)), match.group(2))
        for line in source_lines
        if (match := _HEADING.match(line))
    ]
    sectioned = any(
        level == 2 and not _RELEASE.match(heading)
        for level, heading in headings
    ) and any(
        level == 3 and _RELEASE.match(heading) for level, heading in headings
    )
    found_release = False

    for line in source_lines:
        match = _HEADING.match(line)
        if not match:
            lines.append(line)
            continue

        level = len(match.group(1))
        heading = match.group(2)
        if _TITLE.fullmatch(heading.strip()):
            # The integration supplies one page title for both the canonical
            # changelog and old branches whose changelog lacks a document title.
            continue
        if _RELEASE.match(heading):
            found_release = True
            release_level = 3 if sectioned else 2
            lines.append(f"{'#' * release_level} {heading}")
        elif sectioned and level <= 2:
            # Section headings remain above releases in the canonical,
            # sectioned changelog.
            lines.append(f"## {heading}")
        elif sectioned and found_release and level <= 4:
            lines.append(f"#### {heading}")
        elif found_release and level <= 3:
            # Historic generated notes use both H2 and H3 for categories.
            # Keep the prose untouched while presenting a consistent tree.
            lines.append(f"### {heading}")
        elif not found_release and level <= 2:
            # Reserve H1 for the page title even on older or hand-edited
            # changelogs with introductory sections.
            lines.append(f"## {heading}")
        else:
            lines.append(line)

    return "\n".join(lines).rstrip() + "\n"


def _read_release_notes(app, docname: str, source: list[str]) -> None:
    if docname != "release_notes":
        return
    app.env.note_dependency(str(_CHANGELOG))
    try:
        changelog = _CHANGELOG.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            f"Unable to read the canonical cuML changelog at {_CHANGELOG}"
        ) from error

    policy_url = app.builder.get_relative_uri(docname, "deprecation_policy")
    source[0] = render_release_notes(changelog, policy_url)


def setup(app):
    app.connect("source-read", _read_release_notes)
    return {
        "version": "1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
