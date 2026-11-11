# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect benchmark metadata and persist JSON artifacts."""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib
import importlib.metadata
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def _now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    """Convert arbitrary values to JSON-compatible representations."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return repr(value)


def _version(package: str) -> str:
    """Return an installed package version or an unknown marker."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _gpu_components() -> list[dict[str, Any]]:
    """Collect visible GPU names and counts when CUDA is available."""
    try:
        cp = importlib.import_module("cupy")
        count = cp.cuda.runtime.getDeviceCount()
        devices: dict[str, int] = {}
        for index in range(count):
            props = cp.cuda.runtime.getDeviceProperties(index)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            devices[str(name)] = devices.get(str(name), 0) + 1
        return [
            {
                "type": "accelerator",
                "name": name,
                "count": count,
                "attributes": {"vendor": "NVIDIA"},
            }
            for name, count in devices.items()
        ]
    except Exception:
        return []


def atomic_write(
    path: str | Path, artifact: dict[str, Any], *, overwrite: bool = True
) -> None:
    """Atomically publish a JSON artifact, optionally requiring a new path.

    Parameters
    ----------
    path : str or Path
        Destination artifact path.
    artifact : dict
        JSON-compatible artifact to write.
    overwrite : bool, default=True
        Replace an existing destination. If False, fail if it already exists.
    """
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                artifact, stream, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Linking publishes the complete file atomically without replacing
            # an existing path, even if another process creates it meanwhile.
            os.link(temporary, destination)
            os.unlink(temporary)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _failure(exc: BaseException, phase: str) -> dict[str, Any]:
    """Build a failed outcome from an exception and execution phase."""
    message = str(exc) or f"{type(exc).__name__} raised without a message"
    return {
        "status": "failed",
        "last_phase": phase,
        "error": {"type": type(exc).__name__, "message": message},
    }
