# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

_EXTENSION = Path(__file__).parents[1] / "release_notes.py"
_DOCS_SOURCE = Path(__file__).parents[2]
_SPEC = importlib.util.spec_from_file_location("release_notes", _EXTENSION)
assert _SPEC is not None and _SPEC.loader is not None
release_notes = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_notes)


def test_render_release_notes_normalizes_legacy_heading_levels():
    # rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
    changelog = """# cuML 26.08.00 (5 Aug 2026)

## New Features

- Added a feature.

# cuML 26.06.00 (3 Jun 2026)

### Fixed

#### Details
"""

    rendered = release_notes.render_release_notes(
        changelog, "../deprecation_policy/"
    )

    assert rendered.count("# Release notes\n") == 1
    # rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
    assert "## cuML 26.08.00 (5 Aug 2026)" in rendered
    assert "### New Features" in rendered
    assert "## cuML 26.06.00 (3 Jun 2026)" in rendered
    assert "### Fixed" in rendered
    assert "#### Details" in rendered
    assert "](../deprecation_policy/)" in rendered


def test_render_release_notes_preserves_sectioned_hierarchy():
    changelog = """# Release notes

## Recent releases

Introductory prose.

# rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
### cuML 26.10.00 (unreleased)

#### Added

## Archive

Archive prose.

### cuml 25.08.00 (6 Aug 2025)

#### Fixed
"""

    rendered = release_notes.render_release_notes(changelog)

    assert rendered.count("# Release notes\n") == 1
    assert "## Recent releases\n" in rendered
    # rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
    assert "### cuML 26.10.00 (unreleased)\n" in rendered
    assert "#### Added\n" in rendered
    assert "## Archive\n" in rendered
    assert "### cuml 25.08.00 (6 Aug 2025)\n" in rendered
    assert "#### Fixed\n" in rendered


def test_render_release_notes_replaces_existing_title():
    changelog = """# Release notes

## Before the releases

## cuML unreleased

### Added
"""

    rendered = release_notes.render_release_notes(changelog)

    assert rendered.count("# Release notes\n") == 1
    assert "## Before the releases\n" in rendered
    assert rendered.count("## cuML unreleased\n") == 1
    assert "### Added\n" in rendered


class _Environment:
    def __init__(self):
        self.dependencies = []

    def note_dependency(self, dependency):
        self.dependencies.append(dependency)


class _Builder:
    def get_relative_uri(self, from_docname, to_docname):
        assert from_docname == "release_notes"
        assert to_docname == "deprecation_policy"
        return "../deprecation_policy/"


class _Application:
    def __init__(self):
        self.env = _Environment()
        self.builder = _Builder()


def test_read_release_notes_registers_dependency(monkeypatch, tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    # rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
    changelog.write_text("# cuML 26.10.00 (unreleased)\n", encoding="utf-8")
    monkeypatch.setattr(release_notes, "_CHANGELOG", changelog)
    app = _Application()
    source = ["placeholder"]

    release_notes._read_release_notes(app, "release_notes", source)

    assert app.env.dependencies == [str(changelog)]
    assert source[0].startswith("# Release notes\n")
    assert "](../deprecation_policy/)" in source[0]


def test_read_release_notes_ignores_other_documents(monkeypatch, tmp_path):
    missing = tmp_path / "CHANGELOG.md"
    monkeypatch.setattr(release_notes, "_CHANGELOG", missing)
    app = _Application()
    source = ["unchanged"]

    release_notes._read_release_notes(app, "index", source)

    assert source == ["unchanged"]
    assert app.env.dependencies == []


def test_read_release_notes_reports_missing_changelog(monkeypatch, tmp_path):
    missing = tmp_path / "CHANGELOG.md"
    monkeypatch.setattr(release_notes, "_CHANGELOG", missing)
    app = _Application()

    with pytest.raises(RuntimeError, match=str(Path(missing))):
        release_notes._read_release_notes(app, "release_notes", [""])

    assert app.env.dependencies == [str(missing)]


def test_deprecation_policy_uses_resolvable_developer_policy_link():
    policy = (_DOCS_SOURCE / "deprecation_policy.md").read_text(
        encoding="utf-8"
    )

    assert "(developer_guide/python/development.md)" in policy
    assert "development.md#" not in policy
