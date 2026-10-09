# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manage isolated benchmark commands and callable workers."""

from __future__ import annotations

import codecs
import contextlib
import locale
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, TypeVar

logger = logging.getLogger("cuml.benchmark")

_Result = TypeVar("_Result")


def run_command(command: list[str], environment: dict[str, str]) -> None:
    """Run a command, forwarding stderr and retaining failure diagnostics.

    Stdout is inherited. On POSIX, the command starts in a new process group
    so an interruption or failed command can also stop its descendants. After
    the command exits, stderr draining is bounded in case descendants still
    hold the pipe open.

    Parameters
    ----------
    command : list of str
        Executable and arguments to launch.
    environment : dict of str to str
        Environment variables for the command.

    Raises
    ------
    OSError
        The command could not be launched.
    RuntimeError
        The command exited with a nonzero status; includes captured stderr.
    """
    process = subprocess.Popen(
        command,
        env=environment,
        stderr=subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    decoder = codecs.getincrementaldecoder(locale.getpreferredencoding(False))(
        errors="replace"
    )
    stderr_lines, received = [], 0
    drain_deadline = None
    try:
        while True:
            complete = False
            try:
                _, diagnostics = process.communicate(timeout=0.1)
                complete = True
            except subprocess.TimeoutExpired as exc:
                diagnostics = exc.stderr or b""
            text = decoder.decode(diagnostics[received:], final=complete)
            received = len(diagnostics)
            sys.stderr.write(text)
            stderr_lines.append(text)
            status = process.poll()
            if status is not None and drain_deadline is None:
                # Descendants may inherit stderr. A dead command can no longer
                # enforce their deadlines, so stop its group independently of
                # pipe EOF.
                if status and os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                drain_deadline = time.monotonic() + 1
            if complete:
                break
            if (
                drain_deadline is not None
                and time.monotonic() >= drain_deadline
            ):
                text = decoder.decode(b"", final=True)
                sys.stderr.write(text)
                stderr_lines.append(text)
                break
    except BaseException:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        finally:
            # The leader may have exited while a descendant survived.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
        raise
    finally:
        process.stderr.close()
    stderr = "".join(stderr_lines)
    if process.returncode:
        raise RuntimeError(
            f"worker exited with status {process.returncode}"
            + (f":\n{stderr.strip()}" if stderr else "")
        )


class _PipeHandler(logging.Handler):
    """Forward worker log records through a multiprocessing pipe."""

    def __init__(self, sender: Connection) -> None:
        """Initialize the pipe used to send log records."""
        super().__init__()
        self.sender = sender

    def emit(self, record: logging.LogRecord) -> None:
        """Send a serialized log record to the parent process.

        Parameters
        ----------
        record : logging.LogRecord
            Worker log record to forward.
        """
        self.sender.send(
            (
                "log",
                {
                    "name": record.name,
                    "levelno": record.levelno,
                    "levelname": record.levelname,
                    "msg": self.format(record),
                    "pathname": record.pathname,
                    "lineno": record.lineno,
                    "funcName": record.funcName,
                    "created": record.created,
                    "process": record.process,
                    "processName": record.processName,
                },
            )
        )


class SubprocessTimeout(TimeoutError):
    """Indicate that a benchmark worker exceeded its execution deadline."""

    pass


class SubprocessExited(RuntimeError):
    """Indicate that a benchmark worker exited without returning a result."""

    def __init__(self, exitcode: int | None) -> None:
        """Record the worker's exit code in the exception."""
        self.exitcode = exitcode
        super().__init__(f"Worker exited with code {exitcode}")


def _worker(
    sender: Connection,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    log_level: int,
) -> None:
    """Run a target while forwarding logs, phases, and its result."""
    worker_logger = logging.getLogger("cuml.benchmark")
    worker_logger.handlers = [_PipeHandler(sender)]
    worker_logger.setLevel(log_level)
    worker_logger.propagate = False
    try:
        result = target(
            *args,
            **kwargs,
            report_phase=lambda phase: sender.send(("phase", phase)),
        )
        sender.send(("result", result))
    finally:
        sender.close()


@contextlib.contextmanager
def _worker_process(
    target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Iterator[tuple[BaseProcess, Connection]]:
    """Manage a spawned worker process and its result pipe."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = None
    try:
        process = context.Process(
            target=_worker,
            args=(sender, target, args, kwargs, logger.getEffectiveLevel()),
        )
        process.start()
        sender.close()
        yield process, receiver
    finally:
        sender.close()
        receiver.close()
        if process is not None:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()


def run_in_subprocess(
    target: Callable[..., _Result],
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    timeout: float,
    report_phase: Callable[[str], None] | None = None,
) -> _Result:
    """Run a callable in a spawned worker with an execution deadline.

    Parameters
    ----------
    target : callable
        Worker callable accepting a report_phase callback.
    args : tuple, default=()
        Positional arguments passed to the target.
    kwargs : dict, optional
        Keyword arguments passed to the target.
    timeout : float
        Maximum execution time in seconds, including process startup.
    report_phase : callable, optional
        Callback receiving worker execution phase names.
    """
    deadline = time.monotonic() + timeout
    with _worker_process(target, args, dict(kwargs or {})) as (
        process,
        receiver,
    ):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SubprocessTimeout(
                    f"Execution exceeded {timeout:g} seconds"
                )
            if receiver.poll(min(remaining, 0.1)):
                try:
                    kind, value = receiver.recv()
                except EOFError:
                    # The sender is closed: reap before reading its exit code.
                    process.join(timeout=max(0, deadline - time.monotonic()))
                    if process.is_alive():
                        raise SubprocessTimeout(
                            f"Execution exceeded {timeout:g} seconds"
                        )
                    raise SubprocessExited(process.exitcode) from None
                if kind == "result":
                    return value
                if kind == "phase" and report_phase is not None:
                    report_phase(value)
                elif kind == "log":
                    record = logging.makeLogRecord(value)
                    if logger.isEnabledFor(record.levelno):
                        logger.handle(record)
            if not process.is_alive() and not receiver.poll():
                process.join()
                raise SubprocessExited(process.exitcode)
