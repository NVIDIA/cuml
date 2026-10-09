# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide GPU benchmark execution and cuML source provenance."""

import contextlib
import subprocess
from pathlib import Path
from typing import Any

from .base import GPUBackend


def _git_output(root: Path, *args: str) -> str:
    """Read Git metadata from the given checkout."""
    return subprocess.run(
        ["git", "--literal-pathspecs", *args],
        cwd=root,
        text=True,
        errors="surrogateescape",
        capture_output=True,
        check=True,
        timeout=5,
    ).stdout


def _source() -> dict[str, Any]:
    """Collect provenance from the executing cuML checkout or packaged commit."""
    module_path = Path(__file__).resolve()
    try:
        root = Path(
            _git_output(
                module_path.parent, "rev-parse", "--show-toplevel"
            ).strip()
        )
        # An installed package may live in a virtualenv inside another Git
        # repository. Only use that repository if it tracks this source file.
        _git_output(
            root,
            "ls-files",
            "--error-unmatch",
            "--",
            module_path.relative_to(root).as_posix(),
        )
        revision = _git_output(root, "rev-parse", "HEAD").strip()
        repository = None
        with contextlib.suppress(subprocess.CalledProcessError):
            repository = (
                _git_output(
                    root, "config", "--get", "remote.origin.url"
                ).strip()
                or None
            )
        dirty = bool(
            _git_output(root, "status", "--porcelain", "--untracked-files=all")
        )
        return {
            "repository": repository,
            "revision": revision,
            "dirty": dirty,
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    # Fall back to the commit recorded in the installed package.
    commit_file = module_path.parents[2] / "GIT_COMMIT"
    try:
        revision = commit_file.read_text(encoding="utf-8").strip()
    except OSError:
        revision = None
    return {
        "repository": "https://github.com/nvidia/cuml.git",
        "revision": revision,
        "dirty": None,
    }


class CumlBackend(GPUBackend):
    """Provide cuML package metadata and GPU synchronization."""

    extra_packages = ("cuml",)

    def package_source(self, package: str) -> dict[str, Any] | None:
        """Return source provenance for cuML packages.

        Parameters
        ----------
        package : str
            Installed package name.
        """
        # Record cuML source provenance alongside its version in the artifact.
        return (
            _source() if package == "cuml" else super().package_source(package)
        )
