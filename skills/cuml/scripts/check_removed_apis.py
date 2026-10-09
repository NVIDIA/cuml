# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find uses of cuML APIs that have been deprecated or removed.

Scans Python files, Jupyter notebooks, Markdown (fenced Python blocks are parsed
as Python) and text files (shell scripts, Dockerfiles) for the APIs listed in
``removed_apis.json`` and prints where they are used together with the
replacement.

The installed cuML version is detected from package metadata (cuML is not
imported, so no GPU is needed). Only APIs that are deprecated or removed in that
version are reported, unless ``--all`` is given or no cuML install is found.

Usage::

    python check_removed_apis.py [--all] [--json] PATH [PATH ...]

Exits with status 1 if any removed API is found, otherwise 0.
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

TABLE = Path(__file__).with_name("removed_apis.json")
PYTHON_SUFFIXES = {".py"}
NOTEBOOK_SUFFIXES = {".ipynb"}
MARKDOWN_SUFFIXES = {".md"}
TEXT_SUFFIXES = {".sh", ".txt", ".rst", ".cfg", ".toml", ".yaml", ".yml"}
TEXT_NAMES = {"Dockerfile", "Makefile"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build"}


@dataclass
class Finding:
    path: str
    line: int
    api: str
    status: str
    replacement: str
    deprecated_in: str | None
    removed_in: str | None


def parse_version(text):
    """Return a (year, month) tuple from a version string such as 'YY.MM.PP'."""
    match = re.match(r"(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def installed_cuml_version():
    """Return the installed cuML version as (year, month), or None."""
    for dist in ("cuml-cu13", "cuml-cu12", "cuml"):
        try:
            version = parse_version(metadata.version(dist))
        except metadata.PackageNotFoundError:
            continue
        # The unrelated "cuml" placeholder on PyPI has version 0.x.
        if version is not None and version[0] >= 20:
            return version
    return None


def entry_status(entry, installed, show_all):
    """Return 'removed', 'deprecated' or 'removed later' for an entry, or None to skip it."""
    removed = (
        parse_version(entry["removed_in"]) if entry["removed_in"] else None
    )
    deprecated = (
        parse_version(entry["deprecated_in"])
        if entry["deprecated_in"]
        else None
    )
    if installed is None:
        return "removed" if removed else "deprecated"
    if removed and installed >= removed:
        return "removed"
    if deprecated and installed >= deprecated:
        return "deprecated"
    return "removed later" if show_all else None


class Scanner(ast.NodeVisitor):
    """Collect (line, entry) pairs for removed cuML APIs used in a Python module."""

    def __init__(self, entries):
        self.entries = entries
        self.aliases = {}  # local name -> dotted cuml path
        self.hits = []

    def resolve(self, node):
        """Return the dotted path of a Name/Attribute chain, with cuml aliases expanded."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.append(self.aliases.get(node.id, node.id))
        return ".".join(reversed(parts))

    def check_path(self, node, path):
        for entry in self.entries:
            name = entry["name"]
            if entry["kind"] == "module" and (
                path == name or path.startswith(name + ".")
            ):
                self.hits.append((node.lineno, entry))

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split(".")[0] == "cuml":
                self.aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else "cuml"
                )
                self.check_path(node, alias.name)

    def visit_ImportFrom(self, node):
        if node.module and node.module.split(".")[0] == "cuml":
            self.check_path(node, node.module)
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                self.aliases[alias.asname or alias.name] = full
                self.check_path(node, full)

    def visit_Attribute(self, node):
        path = self.resolve(node)
        if path and path.startswith("cuml."):
            self.check_path(node, path)
        self.generic_visit(node)

    def visit_Call(self, node):
        path = self.resolve(node.func) or ""
        is_cuml = path.startswith("cuml.")
        func_name = path.rsplit(".", 1)[-1]
        for kw in node.keywords:
            for entry in self.entries:
                if (
                    entry["kind"] == "kwarg"
                    and kw.arg == entry["name"]
                    and is_cuml
                    and func_name in entry["classes"]
                ):
                    self.hits.append((node.lineno, entry))
        self.generic_visit(node)

    def visit_Dict(self, node):
        for key in node.keys:
            if isinstance(key, ast.Constant):
                for entry in self.entries:
                    if (
                        entry["kind"] == "build_kwds_key"
                        and key.value == entry["name"]
                    ):
                        self.hits.append((key.lineno, entry))
        self.generic_visit(node)

    def visit_Constant(self, node):
        # Module names embedded in strings, e.g. subprocess.run("python -c 'import cuml.fil'")
        if isinstance(node.value, str):
            for offset, entry in scan_text(node.value, self.entries):
                self.hits.append((node.lineno + offset, entry))


def text_patterns(entries):
    """Regexes for modules named in shell commands and prose."""
    patterns = []
    for entry in entries:
        name = re.escape(entry["name"])
        if entry["kind"] == "module":
            patterns.append((re.compile(rf"\b{name}\b"), entry))
    return patterns


def scan_text(text, entries):
    """Return (line offset, entry) pairs for APIs found in plain text."""
    hits = []
    for regex, entry in text_patterns(entries):
        for match in regex.finditer(text):
            hits.append((text.count("\n", 0, match.start()), entry))
    return hits


def scan_python(source, entries):
    scanner = Scanner(entries)
    scanner.visit(ast.parse(source))
    return scanner.hits


def scan_notebook(text, entries):
    hits = []
    cells = json.loads(text).get("cells", [])
    for index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Shell escapes and magics are not Python; check them as text and blank them out.
        lines = source.splitlines()
        python_lines = []
        for lineno, line in enumerate(lines):
            if line.lstrip().startswith(("!", "%")):
                hits.extend(
                    (index, entry) for _, entry in scan_text(line, entries)
                )
                python_lines.append("")
            else:
                python_lines.append(line)
        try:
            hits.extend(
                (index, entry)
                for _, entry in scan_python("\n".join(python_lines), entries)
            )
        except SyntaxError:
            hits.extend(
                (index, entry) for _, entry in scan_text(source, entries)
            )
    return hits


FENCE = re.compile(
    r"^```(?:python|py)?[ \t]*\n(.*?)^```", re.MULTILINE | re.DOTALL
)


def scan_markdown(text, entries):
    """Scan fenced Python code blocks as Python and everything else as text."""
    hits = [(offset + 1, e) for offset, e in scan_text(text, entries)]
    for block in FENCE.finditer(text):
        first_line = text.count("\n", 0, block.start(1)) + 1
        try:
            block_hits = scan_python(block.group(1), entries)
        except SyntaxError:
            continue
        hits.extend((first_line + line - 1, e) for line, e in block_hits)
    return hits


def iter_files(paths):
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and not SKIP_DIRS.intersection(child.parts):
                    yield child
        elif path.is_file():
            yield path


def scan_file(path, entries):
    """Return (line, entry) pairs for a file; notebook hits report the cell number."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    if path.suffix in PYTHON_SUFFIXES:
        try:
            return scan_python(text, entries)
        except SyntaxError:
            return [(offset + 1, e) for offset, e in scan_text(text, entries)]
    if path.suffix in NOTEBOOK_SUFFIXES:
        return scan_notebook(text, entries)
    if path.suffix in MARKDOWN_SUFFIXES:
        return scan_markdown(text, entries)
    if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
        return [(offset + 1, e) for offset, e in scan_text(text, entries)]
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--all",
        action="store_true",
        help="also report APIs removed after the installed version",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)

    entries = json.loads(TABLE.read_text())["entries"]
    installed = installed_cuml_version()

    findings = []
    for path in iter_files(args.paths):
        seen = set()
        for line, entry in scan_file(path, entries):
            status = entry_status(entry, installed, args.all)
            key = (line, entry["kind"], entry["name"])
            if status is None or key in seen:
                continue
            seen.add(key)
            findings.append(
                Finding(
                    path=str(path),
                    line=line,
                    api=entry["name"],
                    status=status,
                    replacement=entry["replacement"],
                    deprecated_in=entry["deprecated_in"],
                    removed_in=entry["removed_in"],
                )
            )

    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        version = (
            ".".join(f"{v:02d}" for v in installed) if installed else None
        )
        print(
            f"cuML version: {version or 'not installed (reporting everything)'}"
        )
        for f in findings:
            where = f"removed in {f.removed_in}" if f.removed_in else ""
            if f.status == "deprecated":
                where = f"deprecated in {f.deprecated_in}" + (
                    f", removed in {f.removed_in}" if f.removed_in else ""
                )
            print(f"{f.path}:{f.line}: {f.api} ({where}) -> {f.replacement}")
        if not findings:
            print("None of the APIs this checker knows about were found.")
        print(
            "Note: this checker only covers a few removed APIs whose replacement cannot be "
            "found by inspecting the installed cuML. It does not check other deprecations "
            "or whether cuml.accel runs the code on the GPU (use `python -m cuml.accel -v`)."
        )
    return 1 if any(f.status == "removed" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
