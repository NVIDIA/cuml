#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import enum
import importlib.metadata
import os
from typing import Literal

from cuda.bindings import runtime
from packaging.requirements import Requirement

import cuml
from cuml.accel.accelerator import Accelerator
from cuml.internals.outputs import set_global_output_type


class Logger:
    """A simple logger for use in `cuml.accel` only.

    This outputs logs on the same stream (stdout) that `cuml.internals.logger` uses, but
    critically lets us set the log level for `cuml.accel` logs separately from the level
    used by the rest of cuml. This could use python's `logging` directly, but then
    we'd have to setup a formatter and handler within our application, and the user
    experience would be identical, just with added complexity. Using a simple wrapper
    around `print` calls (what `cuml.internals.logger` effectively is) should be sufficient
    for our current needs.
    """

    class _Level(enum.IntEnum):
        ERROR = 40
        WARN = 30
        INFO = 20
        DEBUG = 10

    def __init__(self):
        self.level = Logger._Level.WARN

    def __repr__(self):
        return "<Logger level={self.level.name!r}>"

    def _log(self, msg):
        print(f"[cuml.accel] {msg}", flush=True)

    def set_level(self, level: str) -> None:
        """Set the logger level.

        Parameters
        ----------
        level : {'error', 'warn', 'info', 'debug'}
            The log level to set.
        """
        try:
            level = Logger._Level[level.upper()]
        except KeyError:
            raise ValueError(
                f"level should be one of ['error', 'warn', 'info', 'debug']; "
                f"got {level!r} instead."
            ) from None
        self.level = level

    def error(self, msg: str) -> None:
        """Log a message at ERROR level."""
        if self.level <= Logger._Level.ERROR:
            self._log(msg)

    def warn(self, msg: str) -> None:
        """Log a message at WARN level."""
        if self.level <= Logger._Level.WARN:
            self._log(msg)

    def info(self, msg: str) -> None:
        """Log a message at INFO level."""
        if self.level <= Logger._Level.INFO:
            self._log(msg)

    def debug(self, msg: str) -> None:
        """Log a message at DEBUG level."""
        if self.level <= Logger._Level.DEBUG:
            self._log(msg)


logger = Logger()


class CheckConstraint:
    """A version constraint checker.

    A warning will be logged if the constraint is not satisfied.

    Parameters
    ----------
    constraint: str
        A constraint on the installed version of the package, in the form
        of a valid requirements string.
    """

    def __init__(self, requirement: str) -> None:
        self.requirement = Requirement(requirement)

    def __call__(self) -> bool:
        """Check the constraint.

        Returns whether the constraint was satisfied, and logs a warning if
        it wasn't.
        """
        version = importlib.metadata.version(self.requirement.name)
        if not self.requirement.specifier.contains(version):
            logger.warn(
                f"Warning: `cuml.accel` version {cuml.__version__} was tested "
                f"with {self.requirement!s}, but {self.requirement.name}={version} "
                f"was found. Things may work, but you may also encounter "
                f"potentially subtle breakage."
            )
            return False
        return True


_CONSTRAINTS = {
    "sklearn": CheckConstraint("scikit-learn>=1.6.0,<=1.9.1"),
    "hdbscan": CheckConstraint("hdbscan>=0.8.39,<=0.8.44"),
    "umap": CheckConstraint("umap-learn>=0.5.7,<=0.5.12"),
}

_OVERRIDES = {
    "hdbscan",
    "sklearn.cluster",
    "sklearn.covariance",
    "sklearn.decomposition",
    "sklearn.ensemble",
    "sklearn.kernel_ridge",
    "sklearn.linear_model",
    "sklearn.manifold",
    "sklearn.neighbors",
    "sklearn.preprocessing",
    "sklearn.svm",
    "umap",
}

_PATCHES = {
    "sklearn.pipeline",
    "sklearn.compose",
    "sklearn.utils",
    "sklearn.utils._array_api",
    "sklearn.utils.discovery",
}

ACCELERATED_MODULES = sorted(_OVERRIDES.union(_PATCHES).union(_CONSTRAINTS))


def _exclude_from_acceleration(module: str) -> bool:
    """Whether a module should be excluded from acceleration"""
    parts = module.split(".")
    name = parts[0]
    if name in ("sklearn", "hdbscan", "umap"):
        # Exclude all of sklearn/hdbscan/umap _except_ for modules like
        # `{lib}.*.tests.*` since we want to be able to run the upstream
        # test suites with cuml.accel enabled
        return len(parts) < 2 or parts[-2] != "tests"

    # Exclude any module under these packages
    return name in ("cuml", "treelite")


ACCEL = Accelerator(exclude=_exclude_from_acceleration)
for module in ACCELERATED_MODULES:
    ACCEL.register(
        module,
        override=(
            f"cuml.accel._overrides.{module}" if module in _OVERRIDES else None
        ),
        patch=(
            f"cuml.accel._patches.{module}" if module in _PATCHES else None
        ),
        callback=_CONSTRAINTS.get(module),
    )


def _is_concurrent_managed_access_supported():
    """Check the availability of concurrent managed access (UVM).

    Note that WSL2 does not support managed memory.
    """

    # Ensure CUDA is initialized before checking cudaDevAttrConcurrentManagedAccess
    runtime.cudaFree(0)

    device_id = 0
    err, supports_managed_access = runtime.cudaDeviceGetAttribute(
        runtime.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess, device_id
    )
    if err != runtime.cudaError_t.cudaSuccess:
        logger.error(
            f"Failed to check cudaDevAttrConcurrentManagedAccess with error {err}"
        )
        return False
    return supports_managed_access != 0


def enabled() -> bool:
    """Returns whether the accelerator is enabled."""
    return ACCEL.enabled


def install(
    disable_uvm: bool = False,
    log_level: Literal["error", "warn", "info", "debug", None] = None,
) -> None:
    """Enable `cuml.accel`.

    Parameters
    ----------
    disable_uvm : bool, optional
        Whether to disable UVM.
    log_level : {"error", "warn", "info", "debug"}, optional
        The log level to set for the `cuml.accel` logger. Defaults to `None` to
        check the value of the `CUML_ACCEL_LOG_LEVEL` environment variable,
        falling back to `"warn"` if not configured. Set to `"info"` or
        `"debug"` to get more information about what methods `cuml.accel`
        accelerated for a given run.
    """
    if enabled():
        # Already enabled, no-op
        return

    if log_level is None:
        log_level = os.environ.get("CUML_ACCEL_LOG_LEVEL", "warn")

    logger.set_level(log_level)
    # Set the environment variable if not already set so cuml.accel will
    # be automatically enabled in subprocesses
    os.environ.setdefault("CUML_ACCEL_ENABLED", "1")
    os.environ.setdefault("CUML_ACCEL_LOG_LEVEL", log_level)

    if not disable_uvm:
        if _is_concurrent_managed_access_supported():
            import rmm

            mr = rmm.mr.get_current_device_resource()
            if isinstance(mr, rmm.mr.PrefetchResourceAdaptor) and isinstance(
                mr.upstream_mr, rmm.mr.ManagedMemoryResource
            ):
                # Nothing to do
                pass
            elif not isinstance(mr, rmm.mr.CudaMemoryResource):
                logger.debug(
                    "A non-default memory resource is already configured, "
                    "skipping enabling managed memory."
                )
            else:
                rmm.mr.set_current_device_resource(
                    rmm.mr.PrefetchResourceAdaptor(
                        rmm.mr.ManagedMemoryResource()
                    )
                )
                logger.debug("Enabled managed memory.")
        else:
            logger.debug("Could not enable managed memory on this platform.")

    ACCEL.install()
    set_global_output_type("numpy")

    logger.info("Accelerator installed.")
